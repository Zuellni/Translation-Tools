# Translation Tools
LLM translation toolkit for srt/vtt subtitle files.

## Installation
Create a new environment with conda/mamba:
```
conda create -n tools git python pytorch pytorch-cuda -c conda-forge -c nvidia -c pytorch
conda activate tools
```

Install requirements, use wheels for [ExLlama](https://github.com/turboderp/exllamav2/releases/latest) and [Flash Attention](https://github.com/bdashore3/flash-attention/releases/latest) on Windows:
```
pip install requirements.txt
```
