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

from pydub import AudioSegment
from pydub.playback import play

def load_tts_model(model_name, gpu, model_dir_root="./models/tts"):
    """
    根据模型名称加载 TTS 模型，模型目录为 ./models/tts/<model_name>
    """
    model_dir = Path(model_dir_root) / model_name
    if not model_dir.exists():
        print(f"错误：模型目录 {model_dir} 不存在，请先下载模型。")
        sys.exit(1)
    try:
        print(f"加载 TTS 模型：{model_name}，模型路径：{model_dir}")
        tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False,
            gpu=gpu,
            model_dir=str(model_dir)
        )
        print("模型加载成功。")
        return tts
    except Exception as e:
        print(f"模型加载错误: {str(e)}")
        sys.exit(1)

def generate_and_play_tts(tts, text, ref_audio, language):
    """
    根据输入文本生成 TTS 音频并播放。
    生成音频存储到临时文件，播放完成后删除该文件。
    """
    # 生成临时文件名
    temp_file = Path("./cache") / f"temp_{int(time.time())}.wav"
    temp_file.parent.mkdir(exist_ok=True)
    try:
        print(f"正在生成 TTS 音频：{text}")
        tts.tts_to_file(
            text=text,
            file_path=str(temp_file),
            speaker_wav=ref_audio,
            language=language
        )
        print(f"TTS 音频生成成功：{temp_file}\n正在播放……")
        # 播放音频
        audio = AudioSegment.from_file(str(temp_file))
        play(audio)
    except Exception as e:
        print(f"TTS 生成错误: {str(e)}")
    finally:
        # 尝试清理临时文件
        try:
            if temp_file.exists():
                os.remove(temp_file)
                print(f"已清理临时文件：{temp_file}")
        except Exception as e:
            print(f"清理临时文件失败: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="持续运行的 TTS 服务")
    parser.add_argument("--model", type=str, required=True, help="TTS 模型文件夹名称（位于 ./models/tts/ 下）")
    parser.add_argument("--audio_path", type=str, required=True, help="参考音频文件路径")
    parser.add_argument("--language", type=str, default="zh-cn", help="语言代码，默认为 zh-cn")
    parser.add_argument("--gpu", action="store_true", help="启用 GPU 加速")
    
    args = parser.parse_args()
    
    # 检查参考音频是否存在
    ref_audio = Path(args.audio_path)
    if not ref_audio.exists():
        print(f"错误：参考音频文件 {ref_audio} 不存在。")
        sys.exit(1)
    
    # 加载 TTS 模型（仅加载一次，服务期间一直存在）
    tts = load_tts_model(args.model, args.gpu)
    
    print("TTS 服务已启动，等待输入文本（每行一条文本，输入 'quit' 退出）")
    
    # 持续监听标准输入，等待大语言模型返回的文本\n"
    for line in sys.stdin:
        text = line.strip()
        if text.lower() == "quit":
            print("退出 TTS 服务。")
            break
        if not text:
            continue
        generate_and_play_tts(tts, text, str(ref_audio), args.language)
        print("等待下一条文本……")
    
if __name__ == "__main__":
    main()
