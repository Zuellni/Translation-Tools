import json
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Annotated, Dict

file_dir = Path(__file__).parent
base_dir = Path(r"D:\AI\Models\LLM")
base_dir = base_dir if base_dir.is_dir() else file_dir
models = {}

for model in base_dir.glob("*/"):
    if (model / "config.json").is_file():
        models[model.name] = model

choices = list(models.keys())
default = choices[0] if choices else None

parser = ArgumentParser()
parser.add_argument("-m", "--model", type=str, default=default, choices=choices)
parser.add_argument("-c", "--cache_bits", type=int, default=4, choices=(4, 16))
parser.add_argument("-f", "--lang_from", type=str, default="Chinese")
parser.add_argument("-t", "--lang_to", type=str, default="English")
parser.add_argument("-i", "--input", type=Path, default=".")
parser.add_argument("-l", "--line_len", type=int, default=128)
parser.add_argument("-s", "--seed", type=int, default=-1)
args = parser.parse_args()

import pycountry
import torch
import webvtt
from exllamav2 import *
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2Sampler
from exllamav2.generator.filters import ExLlamaV2PrefixFilter
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.exllamav2 import ExLlamaV2TokenEnforcerFilter
from pydantic import Field, RootModel, StringConstraints
from rich import print
from rich.progress import *

progress = Progress(
    TextColumn("{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
)

lang_from = args.lang_from.capitalize().strip()
lang_to = args.lang_to.capitalize().strip()
lang_code = pycountry.languages.get(name=lang_to)
lang_code = lang_code.alpha_2 if lang_code else lang_to.lower()[:2]
inputs = []

if (input := args.input).is_dir():
    for suffix in ("srt", "vtt"):
        inputs.extend(input.glob(f"*.{suffix}"))
elif input.is_file():
    inputs.append(input)

if (seed := args.seed) != -1:
    torch.manual_seed(seed)
    random.seed(seed)

with progress as p:
    for input in inputs:
        subs = webvtt.from_srt(input) if input.suffix == ".srt" else webvtt.read(input)
        subs_len = len(subs)

        class Translation(RootModel):
            root: Annotated[
                Dict[
                    Annotated[int, Field(ge=1, le=subs_len)],
                    Annotated[str,
                        StringConstraints(
                            min_length=1,
                            max_length=args.line_len,
                            strip_whitespace=True,
                        )
                    ],
                ],
                subs_len,
            ]

        config = ExLlamaV2Config(models[args.model])
        tokenizer = ExLlamaV2Tokenizer(config)
        schema = JsonSchemaParser(Translation.schema())

        filters = [
            ExLlamaV2TokenEnforcerFilter(schema, tokenizer),
            ExLlamaV2PrefixFilter(model, tokenizer, "{"),
        ]

        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = 1.0
        settings.top_k = 0
        settings.top_p = 1.0
        settings.typical = 1.0
        settings.min_p = 0.0
        settings.top_a = 0.1
        settings.token_repetition_penalty = 1.0
        settings.temperature_last = True

        init_len = config.max_seq_len
        init_alpha = config.scale_alpha_value

        print(
            f'Model: "{args.model.lower()}"\n'
            f"Flash Attention: {'\"true\"' if attn.has_flash_attn else '\"false\"'}\n"
            f"Initial Sequence: {init_len}\n"
            f"Initial Alpha: {init_alpha}\n"
        )

        lines = [f'{index + 1}: "{line.text}"' for index, line in enumerate(subs)]
        lines = ", ".join(lines)

        system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        assistant = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        prompt = (
            f"{system}Translate each line from {lang_from} to {lang_to}. "
            "Take all lines into consideration. "
            "Return the translation in JSON format."
            f"{user}{{{lines}}}{assistant}"
        )

        ids = tokenizer.encode(prompt, encode_special_tokens=True)
        prompt_len = ids.shape[-1]

        max_len = -(max(init_len, subs_len * args.line_len / 4) // -256) * 256
        alpha = init_alpha if max_len <= init_len else max_len / init_len
        remaining = max_len - prompt_len

        config.max_seq_len = max_len
        config.scale_alpha_value = alpha

        print(
            f"Lines: {subs_len}\n"
            f"Prompt: {prompt_len}\n"
            f"Free: {remaining}\n"
            f"Sequence: {max_len}\n"
            f"Alpha: {alpha}\n"
        )

        model = ExLlamaV2(config)
        loading = p.add_task("Loading", total=len(model.modules) + 1)

        cache = (
            ExLlamaV2Cache_Q4(model, lazy=True)
            if args.cache_bits == 4
            else ExLlamaV2Cache(model, lazy=True)
        )

        model.load_autosplit(cache, callback=lambda _, __: p.advance(loading))
        generator = ExLlamaV2DynamicGenerator(model, cache, tokenizer)

        memory = torch.cuda.get_device_properties("cuda").total_memory // 1024**2
        reserved = torch.cuda.memory_reserved("cuda") // 1024**2
        allocated = torch.cuda.memory_allocated("cuda") // 1024**2

        print(
            f"Memory: {memory}\n"
            f"Reserved: {reserved}\n"
            f"Allocated: {allocated}\n"
        )

        translating = p.add_task("Translating", total=remaining)

        job = ExLlamaV2DynamicJob(
            input_ids=ids,
            gen_settings=settings,
            filters=filters,
            filter_prefer_eos=True,
            max_new_tokens=remaining,
            stop_conditions=[tokenizer.eos_token_id],
        )

        generator.enqueue(job)
        eos = False
        chunks = []

        while not eos:
            for result in generator.iterate():
                if result["stage"] == "streaming":
                    chunk = result.get("text", "")
                    chunks.append(chunk)
                    eos = result["eos"]

                    print(chunk, end="", flush=True)

        result = json.loads(chunks)
        print(result)

        for index, key in enumerate(result):
            subs[index].text = result[key]

        input_name = input.name.replace("".join(input.suffixes), "")
        input_name = f"{input_name}.{lang_code}"
        output = input.parent / f"{input_name}.vtt"
        index = 1

        while output.exists():
            output = input.parent / f"{input_name}.{index}.vtt"
            index += 1

        subs.save(output)
        model.unload()
