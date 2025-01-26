# RIL

RIL 是 rilance 团队开发的 AI 模型加载与交互式对话启动器。全称 Rilance Intelligence Launcher
一个让你轻松玩转AI的启动器,支持多种大语言模型的加载、管理和实时交互。

## 快速开始

### 环境要求
- Python 3.10+
- NVIDIA GPU (推荐) + CUDA 12.4
- Windows

个人使用的环境是：
- Python 3.10.11
- CUDA 12.4
- Windows 11

### 安装步骤

1. 克隆仓库
```bash
git clone https://github.com/
cd PetAI-Grok-Launcher
```
2. 创建虚拟环境
```bash
python -m venv venv
```
3. 安装依赖
```bash
pip install -r requirements.txt
```
4. 启动程序
```bash
python app-zh.py
```
## 常见问题

1. Q: 安装 flash-attn 失败
A: 请确保安装了 PyTorch 和 CUDA。然后前往[flash-attention github](https://github.com/bdashore3/flash-attention/releases/)下载适合的版本，然后通过pip进行手动安装。比如：
```bash
pip install flash_attn-2.7.1.post1+cu124torch2.5.1cxx11abiFALSE-cp311-cp311-win_amd64.whl
```

2. Q: 如何加载自己的模型？
A: 请将模型文件放入 `models` 文件夹中, RIL 会自动识别。然后你可以自行加载模型。