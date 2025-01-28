# ====================== stt_api_client.py 已弃用 ======================
import argparse
import requests
import torch
import logging
from pathlib import Path
from typing import Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEFAULT_API_ENDPOINT = os.getenv("STT_API_ENDPOINT", "http://127.0.0.1:8000/transcribe")

def local_transcribe(audio_path: str, model_path: Optional[str] = None) -> str:
    """通用本地模型推理"""
    try:
        from transformers import pipeline
        
        # 自动检测模型路径
        resolved_path = Path(model_path) if model_path else detect_model_path()
        if not resolved_path or not resolved_path.exists():
            logger.error(f"无效的模型路径: {resolved_path}")
            return ""

        logger.info(f"正在加载模型: {resolved_path}")
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=str(resolved_path),
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        result = pipe(audio_path)
        return result.get("text", "")
    
    except Exception as e:
        logger.error(f"本地识别失败: {str(e)}", exc_info=True)
        return f"ERROR: {str(e)}"

def api_transcribe(audio_path: str, endpoint: str = "http://127.0.0.1:8000") -> str:
    try:
        with open(audio_path, "rb") as f:
            response = requests.post(
                endpoint,
                files={"file": (Path(audio_path).name, f, "audio/wav")},
                timeout=20
            )
        
        response.raise_for_status()
        return response.json().get("text", "")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"API请求失败: {str(e)}")
        return f"API_ERROR: {str(e)}"
    except Exception as e:
        logger.error(f"API处理异常: {str(e)}")
        return f"ERROR: {str(e)}"

def detect_model_path() -> Optional[Path]:
    """自动检测模型路径"""
    default_path = Path(os.getenv("STT_MODEL_DIR", "models/stt"))
    logger.info(f"检测模型路径: {default_path}")
    
    candidates = [
        default_path / "whisper-large-v3",
        default_path / "wav2vec2-large-960h",
        default_path
    ]
    
    for path in candidates:
        if path.exists() and (path.is_dir() or path.suffix in (".gguf", ".bin")):
            logger.info(f"找到模型路径: {path}")
            return path
    
    logger.error(f"未找到有效模型路径: {default_path}")
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STT客户端")
    parser.add_argument("--mode", choices=["local", "api"], required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model-path", help="本地模型路径（仅限local模式）")
    parser.add_argument("--api-endpoint", default="http://127.0.0.1:8000/transcribe", 
                       help="API端点地址（仅限api模式）")
    args = parser.parse_args()

    try:
        if args.mode == "local":
            result = local_transcribe(args.audio, args.model_path)
            print(f"本地识别结果: {result}")
        else:
            result = api_transcribe(args.audio, args.api_endpoint)
            print(f"API识别结果: {result}")
    except Exception as e:
        logger.error(f"主程序异常: {str(e)}")
        exit(1)