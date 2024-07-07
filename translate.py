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
from jinja2 import Template
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.exllamav2 import ExLlamaV2TokenEnforcerFilter
from pydantic import DirectoryPath, Field, RootModel, StringConstraints
from rich import print

parser = ArgumentParser()
parser.add_argument("-m", "--model", type=DirectoryPath, required=True)
parser.add_argument("-c", "--cache_bits", type=int, default=16, choices=(4, 6, 8, 16))
parser.add_argument("-s", "--seq_len", type=int, default=8192)
parser.add_argument("-l", "--line_len", type=int, default=128)
parser.add_argument("-f", "--lang_from", type=str, default="Chinese")
parser.add_argument("-t", "--lang_to", type=str, default="English")
parser.add_argument("-i", "--input", type=Path, default=".")
args = parser.parse_args()

lang_from = args.lang_from.capitalize().strip()
lang_to = args.lang_to.capitalize().strip()
lang_code = pycountry.languages.get(name=lang_to)
lang_code = lang_code.alpha_2 if lang_code else lang_to.lower()[:2]

inputs = (
    [i for i in args.input.glob("*.*") if i.suffix in (".srt", ".vtt")]
    if args.input.is_dir()
    else [args.input] if args.input.is_file() else []
)

config = ExLlamaV2Config(str(args.model))
config.fasttensors = True
init_len = args.seq_len or config.max_seq_len

tokenizer = ExLlamaV2Tokenizer(config)
chat_template = tokenizer.tokenizer_config_dict["chat_template"]
template = Template(chat_template)

settings = ExLlamaV2Sampler.Settings()
settings.greedy()

caches = {
    4: lambda m: ExLlamaV2Cache_Q4(m, lazy=True),
    6: lambda m: ExLlamaV2Cache_Q6(m, lazy=True),
    8: lambda m: ExLlamaV2Cache_Q8(m, lazy=True),
    16: lambda m: ExLlamaV2Cache(m, lazy=True),
}

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

    schema = JsonSchemaParser(Translation.schema())
    lines = {index: line.text for index, line in enumerate(subs)}
    lines = json.dumps(lines, ensure_ascii=False, separators=(",", ":"))

    instruction = (
        f"Translate each line from {lang_from} to {lang_to}. "
        "Consider the meaning of all lines when translating. "
        f"Keep the line length under {args.line_len} characters. "
        "Return the translation in JSON format."
    )

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": lines},
    ]

    try:
        prompt = template.render(
            messages=messages,
            bos_token=tokenizer.bos_token,
            add_generation_prompt=True,
        )
    except:
        prompt = template.render(
            messages=[{"role": "user", "content": f"{instruction}\n{lines}"}],
            bos_token=tokenizer.bos_token,
            add_generation_prompt=True,
        )

    input_ids = tokenizer.encode(prompt, encode_special_tokens=True)
    input_len = input_ids.shape[-1]
    config.max_seq_len = int(max(init_len, subs_len * args.line_len / 4) // 256 * 256)

    model = ExLlamaV2(config)
    cache = caches[args.cache_bits](model)

    model.load_autosplit(cache, progress=True)
    generator = ExLlamaV2DynamicGenerator(model, cache, tokenizer)

    filters = [
        ExLlamaV2TokenEnforcerFilter(schema, tokenizer),
        ExLlamaV2PrefixFilter(model, tokenizer, "{"),
    ]

    job = ExLlamaV2DynamicJob(
        input_ids=input_ids,
        gen_settings=settings,
        filters=filters,
        filter_prefer_eos=True,
        max_new_tokens=config.max_seq_len - input_len,
        stop_conditions=[tokenizer.eos_token_id],
    )

    generator.enqueue(job)
    eos = False
    chunks = []

    print(
        f'Input: "{input.name}"\n'
        f"Lines: {subs_len}\n"
        f"Tokens: {input_len}\n"
        f"Sequence: {config.max_seq_len}\n"
    )

    while not eos:
        for result in generator.iterate():
            if result["stage"] == "streaming":
                chunk = result.get("text", "")
                print(chunk, end="", flush=True)
                chunks.append(chunk)
                eos = result["eos"]

    print("\n")
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
