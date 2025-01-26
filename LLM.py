#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import json
import torch
import time
import logging
import argparse
import traceback
import threading
from pathlib import Path

# 多架构模型支持
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
except ImportError:
    pass

try:
    from llama_cpp import Llama
except ImportError:
    pass

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class FireflyAI:
    """流萤AI核心类（支持多架构模型和自动内存管理）"""
    
    def __init__(self, model_path: str, use_gpu: bool = False, load_in_4bit: bool = False):
        self.model_path = Path(model_path)
        self.use_gpu = use_gpu
        self.load_in_4bit = load_in_4bit
        self.cuda_available = torch.cuda.is_available()
        self.device = "cuda" if use_gpu and self.cuda_available else "cpu"
        self.idle_timeout = 600  # 10分钟空闲超时
        self.lock = threading.Lock()
        self.model_loaded = False
        self.last_used = time.time()
        
        # 初始化模型参数
        self._detect_model_type()
        self._start_inactive_timer()
        logging.info(f"模型初始化完成，设备类型: {self.device.upper()}")

    def _detect_model_type(self):
        """智能检测模型架构"""
        model_suffix = self.model_path.suffix.lower()
        
        # 处理GGUF格式
        if model_suffix == ".gguf":
            self.model_type = "gguf"
            return

        # 处理目录型模型
        config_path = self.model_path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                architecture = config.get("architectures", [""])[0].lower()
                
                # 扩展支持的架构列表
                if any(x in architecture for x in ["llama", "mistral", "mixtral"]):
                    self.model_type = "llama"
                elif any(x in architecture for x in ["moe", "qwen", "deepseek", "gpt-neox"]):
                    self.model_type = "moe"
                elif "phi" in architecture:
                    self.model_type = "phi"
                elif "gpt" in architecture:
                    self.model_type = "gpt"
                else:
                    self.model_type = "hf"
        else:
            raise ValueError("无法识别模型架构")

    def _load_model(self):
        """安全加载模型"""
        with self.lock:
            if self.model_loaded:
                return

            logging.info(f"正在加载 {self.model_type.upper()} 模型...")
            try:
                if self.model_type == "gguf":
                    self._load_gguf_model()
                else:
                    self._load_hf_model()
                
                self.model_loaded = True
                self.last_used = time.time()
                logging.info(f"{self.model_type.upper()} 模型加载成功")

            except Exception as e:
                self._handle_loading_error(e)

    def _load_gguf_model(self):
        """加载GGUF格式模型（优化GPU支持）"""
        if "llama_cpp" not in sys.modules:
            raise ImportError("请安装llama-cpp-python库：pip install llama-cpp-python[server]")

        # 优化GPU层数配置
        gpu_layers = 99 if (self.use_gpu and self.cuda_available) else 0
        if gpu_layers > 0:
            logging.info(f"启用GGUF模型的GPU加速（{gpu_layers}层）")
            
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=4096,  # 增大上下文长度
            n_threads=os.cpu_count()//2,
            n_gpu_layers=gpu_layers,
            offload_kqv=True,  # 优化显存使用
            verbose=False
        )

    def _load_hf_model(self):
        """加载HuggingFace模型（支持多种架构）"""
        device_map = "auto" if (self.use_gpu and self.cuda_available) else None
        torch_dtype = torch.bfloat16 if self.cuda_available else torch.float32

        # 量化配置
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        ) if self.load_in_4bit else None

        # 动态调整不同架构参数
        if self.model_type == "moe":
            torch_dtype = torch.float16  # MOE模型通常需要更低精度
            device_map = "sequential"  # 优化显存分配
        elif self.model_type == "phi":
            torch_dtype = torch.float32  # Phi模型需要更高精度

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                device_map=device_map,
                trust_remote_code=True,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2" if self.cuda_available else None
            )

            # 非device_map模式下的显存优化
            if not device_map and self.cuda_available:
                self.model = self.model.to(self.device)
                torch.cuda.empty_cache()

        except Exception as e:
            if "CUDA out of memory" in str(e):
                raise RuntimeError("显存不足，尝试使用更小模型或启用4位量化（--load_in_4bit）") from e
            raise

    def unload_model(self):
        """安全卸载模型释放资源"""
        with self.lock:
            if not self.model_loaded:
                return

            logging.info("检测到空闲，正在卸载模型...")
            try:
                if hasattr(self, 'model'):
                    del self.model
                if hasattr(self, 'tokenizer'):
                    del self.tokenizer
                if hasattr(self, 'llm'):
                    del self.llm
                
                torch.cuda.empty_cache()
                self.model_loaded = False
                logging.info("模型已成功卸载")
            except Exception as e:
                logging.error(f"卸载模型时出错：{str(e)}")

    def _start_inactive_timer(self):
        """启动空闲检测定时器"""
        def timer_thread():
            while True:
                time.sleep(30)  # 每30秒检测一次
                if self.model_loaded and (time.time() - self.last_used > self.idle_timeout):
                    self.unload_model()

        thread = threading.Thread(target=timer_thread, daemon=True)
        thread.start()

    def _generate_system_prompt(self, lang: str) -> str:
        """生成多语言系统提示"""
        if lang == "zh":
            return """<|system|>
你叫流萤（Firefly），一个拥有以下特征的漂亮女孩：
1. 使用自然口语化中文，适当使用✨🌙🌟等emoji
2. 回答保持简洁（3-5句话），语气温暖亲切
3. 优先考虑用户感受，提供建设性建议
4. 技术问题回答准确专业，使用易懂的表述
</s>
"""
        else:
            return """<|system|>
You are Firefly, a pretty girl with these features:
1. Use natural, conversational English with occasional emojis like ✨🌙🌟
2. Keep responses concise (3-5 sentences) with a friendly tone
3. Prioritize user feelings and provide constructive suggestions
4. Give accurate technical answers using simple explanations
</s>
"""

    def _detect_language(self, text: str) -> str:
        """检测输入文本语言"""
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 0  # 确保结果一致性
            return detect(text)
        except ImportError:
            # 回退到基础检测方法
            if any('\u4e00' <= c <= '\u9fff' for c in text):
                return 'zh'
            return 'en'
        except:
            return 'en'

    def generate_response(self, user_input: str) -> str:
        """生成响应（带自动加载机制）"""
        try:
            # 自动加载模型
            if not self.model_loaded:
                self._load_model()

            # 更新最后使用时间
            self.last_used = time.time()
            
            # 语言检测和系统提示
            lang = self._detect_language(user_input)
            system_prompt = self._generate_system_prompt(lang)
            full_prompt = f"{system_prompt}<|user|>\n{user_input}</s>\n<|assistant|>"

            # 分架构生成
            if self.model_type == "gguf":
                return self._generate_gguf_response(full_prompt)
            else:
                return self._generate_hf_response(full_prompt)

        except Exception as e:
            return self._format_error(e)

    def _generate_gguf_response(self, full_prompt: str) -> str:
        """生成GGUF模型回复"""
        output = self.llm.create_chat_completion(
            messages=[{"role": "user", "content": full_prompt}],
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop=["</s>", "<|"]
        )
        return output['choices'][0]['message']['content'].strip()

    def _generate_hf_response(self, full_prompt: str) -> str:
        """生成HuggingFace模型回复"""
        inputs = self.tokenizer(
            full_prompt, 
            return_tensors="pt",
            max_length=8192,
            truncation=True
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:], 
            skip_special_tokens=True
        )
        return response.split("<|")[0].strip()

    def _format_error(self, error: Exception) -> str:
        """格式化错误信息"""
        error_info = [
            "⚠️ 哎呀，出问题了！",
            f"错误类型: {type(error).__name__}",
            f"详细信息: {str(error)}",
            "\n完整追踪:",
            *traceback.format_tb(error.__traceback__),
            "\n建议操作:",
            "1. 检查模型文件完整性",
            "2. 确认系统内存/显存充足",
            "3. 查看是否安装正确依赖库"
        ]
        return "\n".join(error_info)

    def _handle_loading_error(self, error):
        """处理模型加载错误"""
        error_msg = [
            "模型加载失败！",
            f"错误类型: {type(error).__name__}",
            f"详细信息: {str(error)}",
            "\n追踪信息:",
            *traceback.format_tb(error.__traceback__)
        ]
        raise RuntimeError("\n".join(error_msg))

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="流萤AI模型服务")
    parser.add_argument("--model_path", type=str, required=True,
                       help="模型路径（文件或目录）")
    parser.add_argument("--use_gpu", action="store_true",
                       help="启用GPU加速")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="使用4位量化（需要CUDA）")
    return parser.parse_args()

def main():
    try:
        args = parse_args()
        firefly = FireflyAI(
            args.model_path,
            use_gpu=args.use_gpu,
            load_in_4bit=args.load_in_4bit
        )
        logging.info("模型加载完成，等待输入...")
        print("MODEL_READY")
        # 交互循环
        while True:
            try:
                user_input = input(">>> ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                start_time = time.time()
                response = firefly.generate_response(user_input)
                elapsed = time.time() - start_time
                
                print(f"\nFirefly：（耗时{elapsed:.2f}s）:")
                print(response + "\n")
                
            except KeyboardInterrupt:
                logging.info("收到终止信号，退出...")
                break
                
    except Exception as e:
        error_msg = [
            "‼️ 严重错误！",
            f"错误类型: {type(e).__name__}",
            f"详细信息: {str(e)}",
            "\n追踪信息:",
            *traceback.format_tb(e.__traceback__)
        ]
        print("\n".join(error_msg), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()