# Translation Tools
LLM translation toolkit for subtitle files.

## Installation
Create a new environment with conda/mamba:
```
conda create -n tools git python pytorch pytorch-cuda -c conda-forge -c nvidia -c pytorch
conda activate tools
```

Clone the repository and install requirements:
```
git clone https://github.com/zuellni/translation-tools tools
pip install -r tools/requirements.txt
```

Use wheels for [ExLlamaV2](https://github.com/turboderp/exllamav2/releases/latest) and [FlashAttention](https://github.com/bdashore3/flash-attention/releases/latest) on Windows:
```
pip install exllamav2-X.X.X+cuXXX.torch2.X.X-cp3XX-cp3XX-win_amd64.whl
pip install flash_attn-X.X.X+cuXXX.torch2.X.X-cp3XX-cp3XX-win_amd64.whl
```

## Usage
Download a model ([Llama-3-8B-Instruct](https://huggingface.co/turboderp/Llama-3-8B-Instruct-exl2) recommended) and translate something:
```
git lfs install
git clone https://huggingface.co/turboderp/Llama-3-8B-Instruct-exl2 -b 6.0bpw tools/model
python tools/translate.py -m tools/model -i tools/test.zh.vtt
```
