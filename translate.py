import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Annotated, Dict

import pycountry
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
from exllamav2.generator.filters import ExLlamaV2PrefixFilter
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.exllamav2 import ExLlamaV2TokenEnforcerFilter
from pydantic import DirectoryPath, Field, RootModel, StringConstraints
from rich import print

parser = ArgumentParser()
parser.add_argument("-m", "--model", type=DirectoryPath, required=True)
parser.add_argument("-c", "--cache_bits", type=int, default=16, choices=(4, 6, 8, 16))
parser.add_argument("-f", "--lang_from", type=str, default="Chinese")
parser.add_argument("-t", "--lang_to", type=str, default="English")
parser.add_argument("-i", "--input", type=Path, default=".")
parser.add_argument("-l", "--line_len", type=int, default=128)
args = parser.parse_args()

lang_from = args.lang_from.capitalize().strip()
lang_to = args.lang_to.capitalize().strip()
lang_code = pycountry.languages.get(name=lang_to)
lang_code = lang_code.alpha_2 if lang_code else lang_to.lower()[:2]

inputs = (
    [i for i in args.input.glob("*.*") if i.suffix in (".srt", ".vtt")]
    if args.input.is_dir()
    else [args.input]
    if args.input.is_file()
    else []
)

for input in inputs:
    subs = webvtt.from_srt(input) if input.suffix == ".srt" else webvtt.read(input)
    subs_len = len(subs)

    class Translation(RootModel):
        root: Annotated[
            Dict[
                Annotated[int, Field(ge=0, lt=subs_len)],
                Annotated[
                    str,
                    StringConstraints(
                        min_length=1,
                        max_length=args.line_len,
                        strip_whitespace=True,
                    ),
                ],
            ],
            subs_len,
        ]

    config = ExLlamaV2Config(str(args.model))
    tokenizer = ExLlamaV2Tokenizer(config)
    schema = JsonSchemaParser(Translation.schema())

    settings = ExLlamaV2Sampler.Settings()
    settings.greedy()

    init_len = config.max_seq_len
    init_alpha = config.scale_alpha_value

    lines = [f'{index}: "{line.text}"' for index, line in enumerate(subs)]
    lines = ", ".join(lines)

    system = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    user = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
    assistant = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    prompt = (
        f"{system}Translate each line from {lang_from} to {lang_to}. "
        "Consider the meaning of all lines when translating. "
        "Return the translation in JSON format."
        f"{user}{{{lines}}}{assistant}"
    )

    ids = tokenizer.encode(prompt, encode_special_tokens=True)
    prompt_len = ids.shape[-1]

    max_len = -(max(init_len, subs_len * args.line_len / 4) // -256) * 256
    alpha = init_alpha if max_len <= init_len else max_len / init_len
    remaining = max_len - prompt_len

    config.fasttensors = True
    config.max_seq_len = max_len
    config.scale_alpha_value = alpha

    print(
        f'File: "{input.name}"\n'
        f"Lines: {subs_len}\n"
        f"Prompt: {prompt_len}\n"
        f"Remaining: {remaining}\n"
        f"Sequence: {max_len}\n"
        f"Alpha: {alpha}\n"
    )

    model = ExLlamaV2(config)

    filters = [
        ExLlamaV2TokenEnforcerFilter(schema, tokenizer),
        ExLlamaV2PrefixFilter(model, tokenizer, "{"),
    ]

    cache = (
        ExLlamaV2Cache_Q4(model, lazy=True)
        if args.cache_bits == 4
        else ExLlamaV2Cache_Q6(model, lazy=True)
        if args.cache_bits == 6
        else ExLlamaV2Cache_Q8(model, lazy=True)
        if args.cache_bits == 8
        else ExLlamaV2Cache(model, lazy=True)
    )

    model.load_autosplit(cache, progress=True)
    generator = ExLlamaV2DynamicGenerator(model, cache, tokenizer)

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
                print(chunk, end="", flush=True)
                chunks.append(chunk)
                eos = result["eos"]

    result = "".join(chunks)
    result = json.loads(result)
    result = Translation(**result).dict()

    for key, value in result.items():
        subs[key].text = value

    output_name = input.name.replace("".join(input.suffixes), "")
    output_name = f"{output_name}.{lang_code}"
    output = input.parent / f"{output_name}.vtt"
    index = 1

    while output.exists():
        output = input.parent / f"{output_name}.{index}.vtt"
        index += 1

    subs.save(output)
    model.unload()
