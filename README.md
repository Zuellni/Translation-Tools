# Translation Tools
LLM translation toolkit for subtitle files.

## Installation
Create a new environment with mamba:
```
mamba create -n tools git python pytorch pytorch-cuda -c conda-forge -c nvidia -c pytorch
mamba activate tools
```

Clone the repository and install requirements:
```
git clone https://github.com/zuellni/translation-tools --branch main --depth 1
cd translations-tools
pip install -r requirements.txt
```

Use wheels for [ExLlamaV2](https://github.com/turboderp/exllamav2/releases/latest) and [FlashAttention](https://github.com/bdashore3/flash-attention/releases/latest) on Windows:
```
pip install exllamav2-X.X.X+cuXXX.torchX.X.X-cp3XX-cp3XX-win_amd64.whl
pip install flash_attn-X.X.X+cuXXX.torchX.X.X-cp3XX-cp3XX-win_amd64.whl
```

## Usage
Download a model ([Gemma-2-9B-It](https://huggingface.co/turboderp/gemma-2-9b-it-exl2) recommended) and translate something:
```
cd translations-tools
git lfs install
git clone https://huggingface.co/turboderp/gemma-2-9b-it-exl2 --branch 6.0bpw --depth 1
python translate.py -h
```
