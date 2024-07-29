import json
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

progress = Progress(
    TextColumn("{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
)

parser = ArgumentParser()
parser.add_argument("-c", "--config", type=Path, default="config.json")
parser.add_argument("-i", "--input", type=Path, required=True)
parser.add_argument("-v", "--video", type=Path, required=True)
parser.add_argument("-f", "--lang_from", type=str, default="")
parser.add_argument("-t", "--lang_to", type=str, default="")
args = parser.parse_args()

config = json.loads(args.config.read_text())
lang_from = args.lang_from.capitalize().strip()
lang_to = args.lang_to.capitalize().strip()
lang_code = pycountry.languages.get(name=lang_to) if lang_to else ""
lang_code = f".{lang_code.alpha_2}" if lang_code else ""

suffixes = {
    ".sbv": lambda i: webvtt.from_sbv(i),
    ".srt": lambda i: webvtt.from_srt(i),
    ".vtt": lambda i: webvtt.read(i),
}

lines = suffixes[args.input.suffix](args.input)
total = len(lines)
digits = len(str(total))

with progress as p, torch.inference_mode():
    extracting = p.add_task("Extracting video frames", total=total)
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

        if not images or len(images[-1]) == config["captioning"]["batch_size"]:
            images.append([])

        images[-1].append(image)
        p.advance(extracting)

    video.release()

    loading = p.add_task("Loading captioning model", total=3)

    captioner = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=config["captioning"]["model"],
        trust_remote_code=True,
    )

    p.advance(loading)

    captioner.to(device=config["captioning"]["device"], dtype=torch.bfloat16).eval()
    p.advance(loading)

    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=config["captioning"]["model"],
        trust_remote_code=True,
    )

    p.advance(loading)

    captioning = p.add_task("Captioning video frames", total=len(images))
    task = "<MORE_DETAILED_CAPTION>"
    captions = []

    for batch in images:
        inputs = processor(text=[task] * len(batch), images=batch, return_tensors="pt")
        inputs.to(device=config["captioning"]["device"], dtype=torch.bfloat16)

        outputs = captioner.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=config["captioning"]["max_new_tokens"],
        )

        outputs = processor.batch_decode(outputs, skip_special_tokens=True)
        captions.extend(outputs)
        p.advance(captioning)

    del captioner, processor
    torch.cuda.empty_cache()

    model_config = ExLlamaV2Config(config["translation"]["model"])
    model_config.max_seq_len = config["translation"]["max_seq_len"]
    model_config.fasttensors = True

    tokenizer = ExLlamaV2Tokenizer(model_config)
    template = tokenizer.tokenizer_config_dict["chat_template"]
    template = Template(template)

    gen_settings = ExLlamaV2Sampler.Settings()
    gen_settings.greedy()

    caches = {
        4: lambda m: ExLlamaV2Cache_Q4(m, lazy=True),
        6: lambda m: ExLlamaV2Cache_Q6(m, lazy=True),
        8: lambda m: ExLlamaV2Cache_Q8(m, lazy=True),
        16: lambda m: ExLlamaV2Cache(m, lazy=True),
    }

    model = ExLlamaV2(model_config)
    cache = caches[config["translation"]["cache_bits"]](model)

    loading = p.add_task("Loading translation model", total=len(model.modules) + 1)
    model.load_autosplit(cache, callback=lambda _, __: p.advance(loading))

    generator = ExLlamaV2DynamicGenerator(model, cache, tokenizer)
    stop_conditions = [tokenizer.eos_token_id, "\n"]
    output = ""

for index, line in enumerate(lines):
    previous = f"Previous translation: {output}\n" if output else ""
    from_lang_from = f" from {lang_from}" if lang_from else ""
    to_lang_to = f" to {lang_to}" if lang_to else ""
    caption = captions[index].strip()
    line = line.text.strip()

    system = (
        f"Translate the following caption{from_lang_from}{to_lang_to}. "
        "Use the provided screenshot description and previously translated captions, "
        "if available and relevant, as context for the current translation. "
        "Respond with the translation only, without adding anything."
    )

    user = (
        f"Screenshot description: {caption}\n{previous}"
        f"Caption: {line}"
    )

    try:
        prompt = template.render(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            bos_token=tokenizer.bos_token,
            add_generation_prompt=True,
        )
    except:
        prompt = template.render(
            messages=[
                {"role": "user", "content": f"{system}\n{user}"}
            ],
            bos_token=tokenizer.bos_token,
            add_generation_prompt=True,
        )

    output = generator.generate(
        prompt=prompt,
        max_new_tokens=config["translation"]["max_new_tokens"],
        gen_settings=gen_settings,
        encode_special_tokens=True,
        stop_conditions=stop_conditions,
        add_bos=True,
        completion_only=True,
    ).strip()

    lines[index].text = output

    print(
        f"\nIndex: {index + 1:0{digits}}/{total}"
        f'\nCaption: "{caption}"'
        f'\nLine: "{line}"'
        f'\nOutput: "{output}"'
    )

name = args.input.name.replace("".join(args.input.suffixes), "")
name = f"{name}{lang_code}"
output = args.input.parent / f"{name}.vtt"
index = 1

while output.exists():
    output = args.input.parent / f"{name}.{index}.vtt"
    index += 1

lines.save(output)
