from argparse import ArgumentParser
from pathlib import Path
from warnings import filterwarnings

filterwarnings("ignore", category=SyntaxWarning)
filterwarnings("ignore", category=UserWarning)

import cv2
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
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler
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
parser.add_argument("-a", "--captioning_model", type=Path, required=True)
parser.add_argument("-m", "--translation_model", type=Path, required=True)
parser.add_argument("-i", "--input", type=Path, required=True)
parser.add_argument("-v", "--video", type=Path, required=True)
parser.add_argument("-d", "--device", type=str, default="cuda")
parser.add_argument("-f", "--lang_from", type=str, default="Chinese")
parser.add_argument("-t", "--lang_to", type=str, default="English")
parser.add_argument("-b", "--batch", type=int, default=8)
parser.add_argument("-c", "--cache_bits", type=int, default=6, choices=(4, 6, 8, 16))
parser.add_argument("-l", "--line_len", type=int, default=128)
parser.add_argument("-s", "--seq_len", type=int, default=4096)
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

lines = suffixes[args.input.suffix](args.input)

with progress as p, torch.inference_mode():
    extracting = p.add_task("Extracting video frames", total=len(lines))
    video = cv2.VideoCapture(args.video)
    fps = video.get(cv2.CAP_PROP_FPS)
    images = []

    for line in lines:
        h, m, s, ms = map(int, line.start.replace(".", ":").split(":"))
        timestamp = int(h * 3600 + m * 60 + s + ms / 1000) * fps
        video.set(cv2.CAP_PROP_POS_FRAMES, timestamp)

        _, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        if not images or len(images[-1]) == args.batch:
            images.append([])

        images[-1].append(image)
        p.advance(extracting)

    video.release()

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

    captioning = p.add_task("Captioning video frames", total=len(images))
    task = "<MORE_DETAILED_CAPTION>"
    captions = []

    for batch in images:
        ids = processor(text=[task] * len(batch), images=batch, return_tensors="pt")
        ids.to(device=args.device, dtype=torch.bfloat16)

        outputs = captioner.generate(
            input_ids=ids["input_ids"],
            pixel_values=ids["pixel_values"],
            max_new_tokens=args.line_len,
        )

        outputs = processor.batch_decode(outputs, skip_special_tokens=True)
        captions.extend(outputs)
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
    loading = p.add_task("Loading translation model", total=len(model.modules) + 1)

    cache = caches[args.cache_bits](model)
    model.load_autosplit(cache, callback=lambda _, __: p.advance(loading))
    generator = ExLlamaV2DynamicGenerator(model, cache, tokenizer)
    stop = [tokenizer.eos_token_id, "\n"]
    result = ""

for index, line in enumerate(lines):
    previous = f"Previous translation: {result}\n" if result else ""
    caption = captions[index].strip()
    line = line.text.strip()

    instruction = (
        f"Translate the following caption from {lang_from} to {lang_to}. "
        "Use the provided screenshot description and previously translated captions, "
        "if available and relevant, as context for the current translation. "
        "Respond with the translation only, without adding anything.\n"
        f"Image description: {caption}\n{previous}"
        f"Caption: {line}"
    )

    prompt = template.render(
        messages=[{"role": "user", "content": instruction}],
        bos_token=tokenizer.bos_token,
        add_generation_prompt=True,
    )

    output = generator.generate(
        prompt=prompt,
        max_new_tokens=args.line_len,
        gen_settings=settings,
        encode_special_tokens=True,
        stop_conditions=stop,
        completion_only=True,
    ).strip()

    lines[index].text = output
    print(f'\nImage: "{caption}"\nLine: "{line}"\nTranslation: "{output}"')

name = args.input.name.replace("".join(args.input.suffixes), "")
name = f"{name}.{lang_code}"
output = args.input.parent / f"{name}.vtt"
index = 1

while output.exists():
    output = args.input.parent / f"{name}.{index}.vtt"
    index += 1

lines.save(output)
