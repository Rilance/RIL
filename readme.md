# RIL

RIL is an AI model loader and interactive chat launcher developed by the rilance team. The full name is Rilance Intelligence Launcher.
A launcher that lets you effortlessly harness the power of AI, supporting loading, managing, and real-time interaction with multiple large language models.

## Quick Start

### Environment Requirements
1. Python 3.10+
2. NVIDIA GPU (recommended) + CUDA 12.4
3. Windows

Personal tested environment:

1. Python 3.10.11
2. CUDA 12.4
3. Windows 11

### Installation Steps
1. Clone the repository
``` bash
git clone https://github.com/
cd PetAI-Grok-Launcher
```
2. Create a virtual environment
```bash
python -m venv venv
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Launch the application
```bash
python app-zh.py
```
## 常见问题

1. Q: Failed to install flash-attn
A: Ensure PyTorch and CUDA are installed. Visit[flash-attention github](https://github.com/bdashore3/flash-attention/releases/) to download the appropriate version, then install manually via pip. For example:
```bash
pip install flash_attn-2.7.1.post1+cu124torch2.5.1cxx11abiFALSE-cp311-cp311-win_amd64.whl
```

2. Q: How to load custom models?
A: Place your model files in the models folder. RIL will automatically detect them, and you can load the models manually afterward.