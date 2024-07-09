from argparse import ArgumentParser
from pathlib import Path
from warnings import filterwarnings

filterwarnings("ignore", category=SyntaxWarning)
filterwarnings("ignore", category=UserWarning)

import ffmpeg
import pycountry
import torch
import webvtt
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Cache_Q6,
    ExLlamaV2Cache_Q8,
    ExLlamaV2Config,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import (
    ExLlamaV2DynamicGenerator,
    ExLlamaV2DynamicJob,
    ExLlamaV2Sampler,
)
from jinja2 import Template
from PIL import Image
from rich import print
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from transformers import AutoProcessor, AutoModelForCausalLM

parser = ArgumentParser()
parser.add_argument("-c", "--captioning_model", type=Path, required=True)
parser.add_argument("-m", "--translation_model", type=Path, required=True)
parser.add_argument("-i", "--input", type=Path, required=True)
parser.add_argument("-v", "--video", type=Path, required=True)
parser.add_argument("-d", "--device", type=str, default="cuda")
parser.add_argument("-f", "--lang_from", type=str, default="Chinese")
parser.add_argument("-t", "--lang_to", type=str, default="English")
parser.add_argument("-b", "--cache_bits", type=int, default=4, choices=(4, 6, 8, 16))
parser.add_argument("-l", "--line_len", type=int, default=128)
parser.add_argument("-s", "--seq_len", type=int, default=8192)
args = parser.parse_args()

progress = Progress(
    TextColumn("{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
)

lang_from = args.lang_from.capitalize().strip()
lang_to = args.lang_to.capitalize().strip()
lang_code = pycountry.languages.get(name=lang_to)
lang_code = lang_code.alpha_2 if lang_code else lang_to.lower()[:2]

suffixes = {
    ".sbv": lambda i: webvtt.from_sbv(i),
    ".srt": lambda i: webvtt.from_srt(i),
    ".vtt": lambda i: webvtt.read(i),
}

subs = suffixes[args.input.suffix](args.input)

with progress as p:
    loading = p.add_task("Loading captioning model", total=3)

    captioner = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=args.captioning_model,
        trust_remote_code=True,
    )

    p.advance(loading)
    captioner.to(device=args.device, dtype=torch.bfloat16).eval()
    p.advance(loading)

    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=args.captioning_model,
        trust_remote_code=True,
    )

    p.advance(loading)

    task = "<MORE_DETAILED_CAPTION>"
    inputs = []

    probe = ffmpeg.probe(args.video)
    info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    width = int(info["width"])
    height = int(info["height"])

    captioning = p.add_task("Captioning video frames", total=len(subs))

    for index, line in enumerate(subs):
        frame, _ = (
            ffmpeg.input(args.video, ss=line.start)
            .output("pipe:", vframes=1, format="rawvideo", pix_fmt="rgb24")
            .run(capture_stdout=True, capture_stderr=True)
        )

        image = Image.frombytes("RGB", (width, height), frame)
        ids = processor(text=task, images=image, return_tensors="pt")
        ids.to(device=args.device, dtype=torch.bfloat16)

        output = captioner.generate(
            input_ids=ids["input_ids"],
            pixel_values=ids["pixel_values"],
            max_new_tokens=args.line_len,
        )[0]

        desc = processor.decode(output, skip_special_tokens=True)
        inputs.append({"line": line.text.strip(), "desc": desc.strip()})
        p.advance(captioning)

    del captioner, processor
    torch.cuda.empty_cache()

    config = ExLlamaV2Config(str(args.translation_model))
    config.max_seq_len = args.seq_len
    config.fasttensors = True

    tokenizer = ExLlamaV2Tokenizer(config)
    template = tokenizer.tokenizer_config_dict["chat_template"]
    template = Template(template)

    settings = ExLlamaV2Sampler.Settings()
    settings.greedy()

    caches = {
        4: lambda m: ExLlamaV2Cache_Q4(m, lazy=True),
        6: lambda m: ExLlamaV2Cache_Q6(m, lazy=True),
        8: lambda m: ExLlamaV2Cache_Q8(m, lazy=True),
        16: lambda m: ExLlamaV2Cache(m, lazy=True),
    }

    model = ExLlamaV2(config)
    cache = caches[args.cache_bits](model)

    loading = p.add_task("Loading translation model", total=len(model.modules) + 1)
    model.load_autosplit(cache, callback=lambda _, __: p.advance(loading))

    generator = ExLlamaV2DynamicGenerator(model, cache, tokenizer)
    stop = [tokenizer.eos_token_id, "\n"]
    result = ""

for index, input in enumerate(inputs):
    previous = f"Previous translation: {result}\n" if result else ""

    instruction = (
        f"Translate the following caption from {lang_from} to {lang_to}. "
        "Use the provided screenshot description and previously translated caption, "
        "if available and relevant, as context for the current translation. "
        "Respond with the translation only, without adding anything.\n"
        f"Image description: {input["desc"]}\n{previous}"
        f"Caption: {input["line"]}"
    )

    prompt = template.render(
        messages=[{"role": "user", "content": instruction}],
        bos_token=tokenizer.bos_token,
        add_generation_prompt=True,
    )

    ids = tokenizer.encode(prompt, encode_special_tokens=True)

    job = ExLlamaV2DynamicJob(
        input_ids=ids,
        gen_settings=settings,
        max_new_tokens=args.line_len,
        stop_conditions=stop,
    )

    generator.enqueue(job)
    chunks = []
    eos = False

    while not eos:
        for result in generator.iterate():
            if result["stage"] == "streaming":
                chunk = result.get("text", "")
                chunks.append(chunk)
                eos = result["eos"]

    result = "".join(chunks).strip()
    subs[index].text = result

    print(
        f'\nDescription: "{input['desc']}"\n'
        f'Caption: "{input['line']}"\n'
        f'Translation: "{result}"',
    )

name = args.input.name.replace("".join(args.input.suffixes), "")
name = f"{name}.{lang_code}"
output = args.input.parent / f"{name}.vtt"
index = 1

while output.exists():
    output = args.input.parent / f"{name}.{index}.vtt"
    index += 1

subs.save(output)
