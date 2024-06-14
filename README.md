# Translation Tools
LLM translation toolkit for srt/vtt subtitle files.

## Installation
Create a new environment with conda/mamba:
```
conda create -n tools git python pytorch pytorch-cuda -c conda-forge -c nvidia -c pytorch
conda activate tools
```

Install requirements, use wheels for [ExLlamaV2](https://github.com/turboderp/exllamav2/releases/latest) and [Flash Attention](https://github.com/bdashore3/flash-attention/releases/latest) on Windows:
```
pip install -r requirements.txt
```

## Usage
Clone the repository, download a model ([Llama-3-8B-Instruct](https://huggingface.co/turboderp/Llama-3-8B-Instruct-exl2) recommended) and translate something:
```
git lfs install
git clone https://github.com/zuellni/translation-tools -b main --depth 1 tools & cd tools
git clone https://huggingface.co/turboderp/Llama-3-8B-Instruct-exl2 -b 6.0bpw --depth 1 model
python translate.py -m model -i test.zh.vtt
```
