from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from transformers import pipeline, AutoConfig, AutoProcessor, AutoModelForSpeechSeq2Seq, AutoModelForCTC
from faster_whisper import WhisperModel
import torch
import logging
import os
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from transformers import WhisperProcessor, WhisperForConditionalGeneration

@asynccontextmanager
async def lifespan(app: FastAPI):
    stt_service.load_model()
    yield  

app = FastAPI(lifespan=lifespan)
logger = logging.getLogger("uvicorn.error")

# 配置项
DEVICE = "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
MODEL_DIR = Path("models/stt")

class STTModelLoader:
    @staticmethod
    def load_model(model_path: Path):
        try:
            config = AutoConfig.from_pretrained(model_path)
            
            if config.model_type == "whisper":
                # 使用专用 Whisper 组件
                processor = WhisperProcessor.from_pretrained(model_path)
                model = WhisperForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=TORCH_DTYPE,
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                ).to(DEVICE)
                
                return pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    device=DEVICE,
                    torch_dtype=TORCH_DTYPE,
                    generate_kwargs={"language": "chinese", "task": "transcribe"}
                )

            elif config.model_type == "wav2vec2":
                processor = AutoProcessor.from_pretrained(model_path)
                model = AutoModelForCTC.from_pretrained(
                    model_path,
                    torch_dtype=TORCH_DTYPE
                ).to(DEVICE)
                return pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    feature_extractor=processor.feature_extractor,
                    tokenizer=processor.tokenizer,
                    device=DEVICE,
                    torch_dtype=TORCH_DTYPE
                )

            else:
                logger.warning(f"使用通用 pipeline 加载 {config.model_type} 模型")
                return pipeline(
                    "automatic-speech-recognition",
                    model=str(model_path),
                    device=DEVICE,
                    torch_dtype=TORCH_DTYPE
                )

        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
    
    @staticmethod
    def load_faster_whisper(model_path: str):
        try:
            # 根据设备支持动态选择 compute_type
            if torch.cuda.is_available():
                compute_type = "float16"  # 如果有 GPU，优先尝试 float16
            else:
                compute_type = "int8"  # CPU 环境更适合使用 int8
    
            model = WhisperModel(
                model_path,
                device=DEVICE,
                compute_type=compute_type
            )
            logger.info(f"成功加载 Faster Whisper 模型: {model_path}")
            return model
        except ValueError as e:
            logger.warning(f"默认 compute_type 加载失败，尝试切换为 'float32': {str(e)}")
            # 如果 float16 或 int8 加载失败，尝试使用 float32
            try:
                model = WhisperModel(
                    model_path,
                    device=DEVICE,
                    compute_type="float32"
                )
                logger.info(f"成功加载 Faster Whisper 模型（使用 float32）: {model_path}")
                return model
            except Exception as inner_e:
                logger.error(f"加载 Faster Whisper 模型失败: {str(inner_e)}")
                raise

class STTService:
    _instance = None
    
    def __init__(self):
        self.pipeline = None
        self.model_loaded = False
        self.faster_whisper_model = None  # 用于加载 Faster Whisper
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = STTService()
        return cls._instance
    
    def _find_model_path(self) -> Optional[Path]:
        logger.info(f"正在查找模型目录: {MODEL_DIR}")
        if not MODEL_DIR.exists():
            logger.error(f"模型目录不存在: {MODEL_DIR}")
            return None
    
        for entry in MODEL_DIR.iterdir():
            logger.info(f"检查路径: {entry}")
            if entry.is_dir() and (entry / "config.json").exists():
                logger.info(f"找到有效模型目录: {entry}")
                return entry
    
        logger.error("未找到有效的 STT 模型")
        return None
    
    def load_model(self):
        try:
            model_path = self._find_model_path()
            if not model_path:
                raise FileNotFoundError(f"未找到包含 config.json 的有效模型目录: {MODEL_DIR}")
            
            logger.info(f"检测到模型类型路径: {model_path}")
            
            # 判断是否是 Faster Whisper 模型
            if "faster-whisper" in str(model_path).lower():
                self.faster_whisper_model = STTModelLoader.load_faster_whisper(str(model_path))
                self.model_loaded = True
            else:
                self.pipeline = STTModelLoader.load_model(model_path)
                self.model_loaded = True
            
            logger.info(f"{model_path.name} 模型加载成功 @ {DEVICE}")
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}", exc_info=True)
            self.model_loaded = False

stt_service = STTService.get_instance()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if not stt_service.model_loaded:
        raise HTTPException(503, "服务未就绪")
    
    try:
        with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        if stt_service.faster_whisper_model:
            # 使用 Faster Whisper 模型处理
            segments, info = stt_service.faster_whisper_model.transcribe(
                tmp_path, beam_size=5, language="zh"
            )
            result_text = " ".join([segment.text for segment in segments])
        else:
            # 使用原有 pipeline 模型处理
            result = stt_service.pipeline(tmp_path)
            result_text = result["text"]
        
        os.unlink(tmp_path)
        return {"text": result_text}
    
    except Exception as e:
        logger.error(f"识别失败: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "语音识别失败"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
