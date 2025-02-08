#!/usr/bin/env python3
import argparse
import sys
import time
import os
from pathlib import Path

try:
    from TTS.api import TTS
except ImportError:
    sys.exit("请确保已安装 TTS 包（pip install TTS）")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ------------------------- FastAPI 服务初始化 -------------------------
app = FastAPI()

# 全局变量，用于保存加载后的 TTS 模型和参考音频路径
tts = None
ref_audio = None

# ------------------------- 加载 TTS 模型 -------------------------
def load_tts_model(model_name, gpu, model_dir_root="./models/tts"):
    """
    根据模型名称加载 TTS 模型。
    如果 model_name 指定的是一个路径，则直接使用该路径；否则与 model_dir_root 拼接生成路径。
    然后从该目录中分别加载模型文件、配置文件。
    """
    # 先尝试直接将 model_name 当作路径使用
    model_path = Path(model_name)
    if not model_path.exists():
        # 如果直接使用不存在，则与 model_dir_root 拼接
        model_path = Path(model_dir_root) / model_name

    if not model_path.exists():
        print(f"错误：模型目录 {model_path} 不存在，请先下载模型。")
        sys.exit(1)

    try:
        print(f"加载 TTS 模型：{model_name}，模型路径：{model_path}")
        # 根据你的模型文件夹结构设定正确的配置文件名
        config_file = model_path / "config.json"
        # 注意：不再生成单独的 model_file，而是直接使用模型目录作为 model_path
        tts_model = TTS(
            model_path=str(model_path),  # 直接传入模型目录
            config_path=str(config_file),
            progress_bar=False,
            gpu=gpu
        )
        print("模型加载成功。")
        return tts_model
    except Exception as e:
        print(f"模型加载错误: {str(e)}")
        sys.exit(1)

# ------------------------- 生成 TTS 音频 -------------------------
def generate_tts_file(tts_model, text, ref_audio_path, language):
    """
    根据输入文本生成 TTS 音频并保存到代码根目录下的 audio 文件夹中，
    每次生成的文件名固定为 tts_output.wav，从而覆盖之前的文件。
    
    参数说明：
      - tts_model: 已加载的 TTS 模型对象
      - text: 待合成的文本
      - ref_audio_path: 参考音频文件路径（用于指定说话人）
      - language: 语言代码，由调用时传入（部署后动态提交语言类型）
    """
    # 确保 audio 文件夹存在
    print(f"正在使用参考音频进行克隆：{ref_audio_path}")
    audio_dir = Path("./audio")
    audio_dir.mkdir(exist_ok=True)
    output_file = audio_dir / "tts_output.wav"
    try:
        print(f"正在生成 TTS 音频：{text}")
        tts_model.tts_to_file(
            text=text,
            file_path=str(output_file),
            speaker_wav=str(ref_audio_path),
            language=language,
            speed=1.0,  # 语速
            temperature=0.6  # 控制音色变化程度（0.3 ~ 0.7 之间调整） 值越高，音色变化更大；值越低，更接近参考音色。
        )
        print(f"TTS 音频生成成功：{output_file}")
        return output_file
    except Exception as e:
        print(f"TTS 生成错误: {str(e)}")
        raise e

# ------------------------- FastAPI 接口定义 -------------------------
class TTSRequest(BaseModel):
    text: str
    language: str

@app.post("/tts")
async def tts_endpoint(request: TTSRequest):
    """
    POST 请求示例：
      {
         "text": "你好，世界！",
         "language": "zh-cn"
      }
    接口将调用 TTS 模型生成音频文件（固定保存为 audio/tts_output.wav），并返回生成结果。
    """
    try:
        output_file = generate_tts_file(tts, request.text, ref_audio, request.language)
        return {"status": "success", "output_file": str(output_file)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ------------------------- 主函数 -------------------------
def main():
    parser = argparse.ArgumentParser(description="TTS FastAPI 服务")
    parser.add_argument("--model", type=str, required=True, help="TTS 模型文件夹名称（位于 ./models/tts/ 下）")
    parser.add_argument("--audio_path", type=str, required=True, help="参考音频文件路径")
    parser.add_argument("--gpu", action="store_true", help="启用 GPU 加速")
    
    args = parser.parse_args()
    
    ref_audio_path = Path(args.audio_path)
    if not ref_audio_path.exists():
        print(f"错误：参考音频文件 {ref_audio_path} 不存在。")
        sys.exit(1)
    
    global tts, ref_audio
    tts = load_tts_model(args.model, args.gpu)
    ref_audio = ref_audio_path
    
    print("TTS FastAPI 服务已启动。")
    uvicorn.run(app, host="127.0.0.1", port=8010)

if __name__ == "__main__":
    main()
