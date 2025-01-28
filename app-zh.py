# ====================== 导入依赖 ======================
import os
import sys
import markdown
import requests
import shutil
import time
import subprocess
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QUrl, QObject, QSize, QProcess, QPoint, QFileInfo, QRect, QRectF, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QListWidget, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QLineEdit, QPushButton, QLabel,
    QListWidgetItem, QMessageBox, QSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QTabWidget, QButtonGroup, QComboBox,
    QDialog, QProgressBar, QPlainTextEdit, QTabBar, QSizeGrip, QScrollArea
)
from PyQt5.QtWidgets import QStyle
from PyQt5.QtGui import (QColor, QPalette, QFont, QDesktopServices, 
                        QTextCursor, QCursor, QIcon, QPainter, QBrush, 
                        QPen, QPainterPath)
from PyQt5.QtWebEngineWidgets import QWebEngineView
from pathlib import Path
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtWebEngineWidgets import QWebEnginePage
from PyQt5.QtCore import QProcessEnvironment  
from PyQt5.QtWidgets import QLayout, QSizePolicy
from PyQt5.QtCore import QSize, QRect, QPoint
import sounddevice as sd
from scipy.io.wavfile import write
import requests
from tempfile import NamedTemporaryFile
import json
import numpy as np
from scipy.io.wavfile import write
import requests
import noisereduce as nr
import librosa

# ====================== 前置配置 ======================
import warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"sipPyTypeDict\(\) is deprecated.*",
)

# 获取资源绝对路径
def resource_path(relative_path):
    """ 获取资源的绝对路径 """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

# ====================== 常量定义 ======================
APP_NAME = "AIToolkitPro"
APP_VERSION = "1.0.0"
MAX_CONCURRENT = 3
BORDER_WIDTH = 5 
MODEL_EXTENSIONS = ('.safetensors', '.bin', '.pth', '.pt', '.gguf')
APP_ICON_PATH = resource_path("assets/icons/icon.ico") 
user_prefix = "User: "
model_prefix = "Firefly: "
raw_audio_path = "./audio/recorded_audio.wav"
denoised_audio_path = "./audio/denoised_audio.wav"

# ====================== 全局函数类 ======================
def call_stt_api(audio_path, api_endpoint="http://127.0.0.1:8000/transcribe"):
    """
    调用 STT API 进行语音转文本
    :param audio_path: 音频文件路径
    :param api_endpoint: API 端点地址
    :return: 转录的文本或错误信息
    """
    try:
        with open(audio_path, "rb") as audio_file:
            response = requests.post(
                api_endpoint,
                files={"file": (os.path.basename(audio_path), audio_file, "audio/wav")},
                timeout=20
            )
        response.raise_for_status()
        # 确认 API 返回了正确的 JSON 数据
        return response.json().get("text", "无返回文本")
    except requests.exceptions.RequestException as e:
        return f"API_ERROR: {str(e)}"
    except Exception as e:
        return f"ERROR: {str(e)}"

# ====================== 配置管理类 ======================
class ConfigManager:
    _instance = None
    
    def __init__(self):
        self.settings = QSettings("Rilance", "RIL")
        self.dark_mode = False
        self.proxy = {"http": "", "https": ""}
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.font_size = 12
        self.download_history = []
        self.terminal_type = "cmd"
        self.font_family = "Microsoft Yahei"
        self.font_size = 12
        self.stt_mode = "api"
        self.api_endpoint = "http://127.0.0.1:8000/transcribe" 
        self.load_settings()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = ConfigManager()
        return cls._instance

    def load_settings(self):
        self.dark_mode = self.settings.value("dark_mode", False, type=bool)
        self.proxy["http"] = self.settings.value("proxy/http", "")
        self.proxy["https"] = self.settings.value("proxy/https", "")
        self.model_path = self.settings.value("model_path", self.model_path)
        self.font_size = self.settings.value("font_size", 12, type=int)
        self.download_history = self.settings.value("download_history", [])
        self.terminal_type = self.settings.value("terminal_type", "cmd")
        self.font_family = self.settings.value("font_family", "Microsoft Yahei")
        self.font_size = self.settings.value("font_size", 12, type=int)
        self.stt_mode = self.settings.value("stt_mode", "api")
        self.api_endpoint = self.settings.value("api_endpoint", "http://127.0.0.1:8000/transcribe")

    def save_settings(self):
        self.settings.setValue("dark_mode", self.dark_mode)
        self.settings.setValue("proxy/http", self.proxy["http"])
        self.settings.setValue("proxy/https", self.proxy["https"])
        self.settings.setValue("model_path", self.model_path)
        self.settings.setValue("font_size", self.font_size)
        self.settings.setValue("download_history", self.download_history)
        self.settings.setValue("terminal_type", self.terminal_type)
        self.settings.setValue("font_family", self.font_family)
        self.settings.setValue("font_size", self.font_size)
        self.settings.setValue("stt_mode", self.stt_mode)
        self.settings.setValue("api_endpoint", self.api_endpoint)

# ====================== 首页组件 ======================
class HomePage(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 背景图片
        bg_path = os.path.abspath("./assets/bk/114793143.png")
        if os.path.exists(bg_path):
            self.bg_label = QLabel()
            self.bg_label.setPixmap(QIcon(bg_path).pixmap(800, 400))
            self.bg_label.setAlignment(Qt.AlignCenter)
            self.bg_label.setStyleSheet("background: transparent;")
            layout.addWidget(self.bg_label)
        else:
            error_label = QLabel("背景图片未找到")
            error_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(error_label)
        
        # 欢迎文字
        welcome_label = QLabel("欢迎来到RIL\n(Rilance Intelligence Launcher)")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #4CAF50;
            margin: 20px 0;
        """)
        layout.addWidget(welcome_label)
        
        self.setLayout(layout)

# ====================== 对话页面 ======================
class ChatPage(QWidget):
    user_prefix = "User: "
    model_prefix = "Firefly: "

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_process = None  # 模型加载进程
        self.record_btn = None
        self.init_ui()
        self.scan_local_models()
        self.output_buffer = ""
        self.output_timer = QTimer()
        self.output_timer.timeout.connect(self.flush_buffer)
        self.current_message = ""
        self.is_recording = False

        self.stream = None  # 显式初始化
        self.audio_frames = []
        self.sample_rate = 16000

    def init_ui(self):
        if self.layout() is not None:
            QWidget().setLayout(self.layout()) 
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        self.record_btn = QPushButton()
        self.record_btn.setVisible(False)

        # 顶部控制栏
        control_layout = QHBoxLayout()
        
        # 模型选择下拉框
        self.model_combo = QComboBox()
        self.model_combo.setFixedWidth(300)
        self.model_combo.setStyleSheet("QComboBox { padding: 5px; }")
        
        # 刷新按钮
        refresh_btn = QPushButton("刷新")
        refresh_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_BrowserReload))
        refresh_btn.clicked.connect(self.scan_local_models)
        
        # 加载模型按钮
        self.load_btn = QPushButton("加载模型")
        self.load_btn.setIcon(QIcon("./assets/icons/load.png"))
        self.load_btn.clicked.connect(self.load_model)
        self.load_btn.setEnabled(False)

        # 硬件加速选项
        self.gpu_check = QCheckBox("GPU加速")
        self.quant_check = QCheckBox("4位量化")

        control_layout.addWidget(QLabel("选择模型:"))
        control_layout.addWidget(self.model_combo)
        control_layout.addWidget(refresh_btn)
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.gpu_check)
        control_layout.addWidget(self.quant_check)
        main_layout.addLayout(control_layout)

        # 对话历史
        self.chat_history = QPlainTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("""
            QPlainTextEdit {
                font-family: Consolas;
                font-size: 12px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)
        main_layout.addWidget(self.chat_history)

        # 输入区域
        input_layout = QHBoxLayout()
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("输入您的问题...")
        self.user_input.returnPressed.connect(self.send_message)
        send_btn = QPushButton("发送")
        send_btn.clicked.connect(self.send_message)
        self.record_btn = QPushButton() 
        self.record_btn.setIcon(QIcon(resource_path("assets/icons/mic.png")))
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self.toggle_recording)
        self.record_btn.setStyleSheet("""
            QPushButton { 
                padding: 5px;
                border: none;
                background: transparent;
            }
            QPushButton:checked {
                background: #ff0000;
                border-radius: 8px;
            }
        """)
        
        input_layout.addWidget(self.record_btn)

        self.setLayout(main_layout)
        
        input_layout.addWidget(self.user_input, 4)
        input_layout.addWidget(send_btn, 1)
        main_layout.addLayout(input_layout)

        self.setLayout(main_layout)

    def scan_local_models(self):
        self.model_combo.clear()
        model_dir = Path(self.config.model_path) 
        
        if not model_dir.exists():
            QMessageBox.warning(self, "警告", "模型目录不存在！已自动创建。")
            try:
                model_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"创建目录失败: {str(e)}")
            return

        model_exts = ('.safetensors', '.bin', '.pth', '.pt', '.gguf')
        valid_models = []

        for entry in model_dir.iterdir():
            if entry.is_file() and entry.suffix.lower() == '.gguf':
                valid_models.append(entry.stem) 
            elif entry.is_dir():
                has_config = (entry / 'config.json').exists()
                has_model_file = any(f.suffix in model_exts for f in entry.iterdir())
                if has_config or has_model_file:
                    valid_models.append(entry.name)

        valid_models = sorted(list(set(valid_models)))
        
        if not valid_models:
            QMessageBox.information(self, "提示", "未找到可用模型")
            self.load_btn.setEnabled(False)
            return

        self.model_combo.addItems(valid_models)
        self.load_btn.setEnabled(True)

    def load_model(self):
        model_name = self.model_combo.currentText()
        if not model_name:
            return

        venv_python = VirtualEnvManager.get_python_path()
        process_env = VirtualEnvManager.get_env_with_venv()
        
        # 构建模型路径
        base_path = Path(self.config.model_path)
        model_path = base_path / model_name
        if not model_path.exists():
            model_path = model_path.with_suffix('.gguf')

        if not model_path.exists():
            QMessageBox.critical(self, "错误", f"找不到模型文件: {model_path}")
            return

        # 准备参数
        args = [
            "LLM.py",
            "--model_path", str(model_path)
        ]
    
        if self.gpu_check.isChecked():
            args.append("--use_gpu")
        if self.quant_check.isChecked():
            args.append("--load_in_4bit")

        # 启动进程
        self.model_process = QProcess()
        self.model_process.setProcessEnvironment(process_env)  # 设置完整环境变量
        self.model_process.start(str(venv_python), args)
        self.model_process.readyReadStandardOutput.connect(self.handle_initial_output)
        
        # 连接信号
        self.model_process.readyReadStandardOutput.connect(self.handle_output)
        self.model_process.readyReadStandardError.connect(self.handle_error)
        self.model_process.finished.connect(self.handle_finish)
        
        self.model_process.start()
        self.append_message(f"[系统] 启动模型加载: {' '.join(args)}", is_system=True)

    def handle_initial_output(self):
        data = self.model_process.readAllStandardOutput().data().decode()
        if "MODEL_READY" in data: 
            self.append_message("[系统] 模型已就绪，可以开始对话", is_system=True)
            self.model_process.readyReadStandardOutput.disconnect(self.handle_initial_output)

    def send_message(self):
        if not self.model_process or self.model_process.state() != QProcess.Running:
            QMessageBox.warning(self, "错误", "请先加载模型")
            return

        question = self.user_input.text().strip()
        if not question:
            return

        self.model_process.write(f"{question}\n".encode())
        self.append_message(f"User: {question}")
        self.user_input.clear()

    def handle_output(self):
        data = self.model_process.readAllStandardOutput().data().decode(errors='ignore')
        if data:
            self.output_buffer += data
            if not self.output_timer.isActive():
                self.output_timer.start(50) 

    def handle_error(self):
        err = self.model_process.readAllStandardError().data().decode()
        if err.strip():
            self.append_message(f"[系统] {err.strip()}", is_system=True)

    def handle_finish(self, exit_code, exit_status):
        status = "正常" if exit_code == 0 else f"异常 ({exit_code})"
        self.append_message(f"[系统] 模型进程已退出 [{status}]", is_system=True)

    def append_message(self, message, is_system=False):
        """添加消息到对话历史"""
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        if is_system:  
            cursor.insertHtml(f'<span style="color:#666;">{message}</span><br>')
        else:
            # 用户消息处理
            if message.startswith(user_prefix):
                clean_msg = message[len(user_prefix):].lstrip() 
                cursor.insertText(f"{user_prefix}{clean_msg}\n")
            # 模型消息处理
            elif message.startswith(model_prefix):
                clean_msg = message[len(model_prefix):].lstrip()
                cursor.insertText(f"{model_prefix}{clean_msg}\n")
            else:  # 未知类型默认处理
                cursor.insertText(f"{message}\n")
        
        # 自动滚动
        self.chat_history.ensureCursorVisible()

    def flush_buffer(self):
        if self.output_buffer:
            # 合并字符到当前消息
            self.current_message += self.output_buffer
            self.output_buffer = ""
        
            # 直接插入文本（不逐字换行）
            cursor = self.chat_history.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.select(QTextCursor.BlockUnderCursor)
        
            # 替换原有内容
            cursor.removeSelectedText()
            cursor.insertText(f"Firefly: {self.current_message}")
            self.chat_history.ensureCursorVisible()

    def toggle_recording(self):
        if self.record_btn.isChecked():
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        try:
            self.is_recording = True
            self.audio_frames = []
            
            def callback(indata, frames, time, status):
                if self.is_recording:
                    self.audio_frames.append(indata.copy())

            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=callback
            )
            self.stream.start()
        except Exception as e:
            self.is_recording = False
            self.record_btn.setChecked(False)
            QMessageBox.critical(self, "录音错误", f"无法启动录音设备: {str(e)}")

    def stop_recording(self):
        self.is_recording = False
        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
                self.stream = None
        except Exception as e:
            QMessageBox.warning(self, "录音错误", f"停止录音失败: {str(e)}")
            self.record_btn.setChecked(False)
            return

        if not self.audio_frames:
            self.record_btn.setChecked(False) 
            return
        
        output_dir = "./audio"
        os.makedirs(output_dir, exist_ok=True)
        raw_audio_path = os.path.join(output_dir, "recorded_audio.wav")
        denoised_audio_path = os.path.join(output_dir, "denoised_audio.wav")

        try:
            # 拼接录音帧并保存原始音频文件
            audio_data = np.concatenate(self.audio_frames, axis=0)
            write(raw_audio_path, self.sample_rate, audio_data.astype(np.int16))

            # 加载音频并应用降噪
            reduced_audio = self.apply_noise_reduction(raw_audio_path)
            denoised_file_path = os.path.join(output_dir, "denoised_audio.wav")
            write(denoised_file_path, self.sample_rate, reduced_audio.astype(np.int16))

            # 调用 STT API，使用降噪后的音频文件
            transcribed_text = call_stt_api(denoised_audio_path)
            self.user_input.setText(transcribed_text)
        except Exception as e:
            QMessageBox.warning(self, "处理错误", f"音频处理失败: {str(e)}")
            self.record_btn.setChecked(False)

        try:
            if os.path.exists(raw_audio_path):
                os.remove(raw_audio_path)
            if os.path.exists(denoised_audio_path):
                os.remove(denoised_audio_path)
            QMessageBox.information(None, "清理完成", "临时音频文件已删除。")
        except Exception as e:
            QMessageBox.critical(
                None, 
                "清理错误",  
                f"清理过程中出现异常: {str(e)}" 
            )
    
    def transcribe_api(self, filename):
        try:
            process = subprocess.Popen(
                ["python", "stt_api_client.py", 
                 "--mode", "api",
                 "--audio", filename,
                 "--api-endpoint", f"{self.config.api_endpoint}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate(timeout=20)
        
            if process.returncode != 0:
                err_msg = stderr.decode().strip() or "API请求失败"
                return f"API错误: {err_msg}"
            
            return stdout.decode().strip()
        
        except subprocess.TimeoutExpired:
            process.kill()
            return "API请求超时"
        except Exception as e:
            return f"API调用错误: {str(e)}"
        
    def transcribe_audio(self, filename):
        """根据配置选择转录模式"""
        if self.config.stt_mode == "local":
            return self.transcribe_local(filename)
        else:
            return self.transcribe_api(filename)
        
    def apply_noise_reduction(self, audio_path):
        """
        加载音频文件并进行降噪处理。
        
        Args:
            file_path (str): 输入的音频文件路径。
        
        Returns:
            np.ndarray: 降噪后的音频数据。
        """
        
        audio, rate = librosa.load(audio_path, sr=self.sample_rate)
        epsilon = 1e-8  # 根据实际效果调整
        audio += epsilon * np.random.randn(len(audio))

        try:
            # 加载音频文件
            audio_data, sr = librosa.load(audio_path, sr=self.sample_rate)
        
            # 计算背景噪声 (取音频前 1 秒作为噪声样本)
            noise_sample = audio_data[:sr]
        
            # 应用降噪
            reduced_audio = nr.reduce_noise(
                y=audio_data, 
                sr=sr, 
                y_noise=noise_sample,
                stationary=False,
                n_fft=2048,
                win_length=1024,
                n_std_thresh=2,
                verbose=False,
                n_std_thresh_stationary=1.5 
            )
            return reduced_audio
        except Exception as e:
            QMessageBox.warning(self, "降噪错误", f"无法完成降噪处理: {str(e)}")
            return np.zeros(1) 

# ====================== 虚拟环境管理器 ======================
class VirtualEnvManager:
    @staticmethod
    def get_python_path() -> Path:
        """获取虚拟环境Python路径（兼容打包模式）"""
        # 获取可执行文件所在目录
        if getattr(sys, 'frozen', False):
            # 打包模式下，sys.executable是exe文件的路径
            base_path = Path(sys.executable).parent.resolve()  # 获取exe所在目录
            venv_path = base_path / "venv"
            
            # 如果同级目录不存在，尝试在exe所在目录的上层目录查找
            if not venv_path.exists():
                venv_path = base_path.parent / "venv"
        else:
            # 开发模式下的项目根目录
            base_path = Path(__file__).parent.resolve()
            venv_path = base_path / "venv"

        # 检查虚拟环境目录是否存在
        if not venv_path.exists():
            error_msg = (
                f"虚拟环境目录未找到：\n{venv_path}\n"
                "请将venv目录放置在可执行文件同级目录"
            )
            QMessageBox.critical(
                None,
                "虚拟环境错误",
                error_msg
            )
            sys.exit(1)

        # 确定Python解释器路径
        if sys.platform.startswith("win"):
            venv_python = venv_path / "Scripts" / "python.exe"
        else:
            venv_python = venv_path / "bin" / "python"

        # 再次验证Python解释器是否存在
        if not venv_python.exists():
            error_msg = (
                f"Python解释器未找到：\n{venv_python}\n"
                "请确保虚拟环境已正确安装"
            )
            QMessageBox.critical(
                None,
                "虚拟环境错误",
                error_msg
            )
            sys.exit(1)
            
        return venv_python
    
    @staticmethod
    def get_env_with_venv():
        """获取包含虚拟环境路径的环境变量"""
        venv_python = VirtualEnvManager.get_python_path()
        venv_bin = venv_python.parent
        
        env = QProcessEnvironment.systemEnvironment()
        original_path = env.value("PATH", "")

        # 仅修改PATH环境变量
        if sys.platform.startswith("win"):
            new_path = f"{str(venv_bin)};{original_path}"
        else:
            new_path = f"{str(venv_bin)}:{original_path}"
        
        env.insert("PATH", new_path)
        
        # 删除手动设置的Python相关变量
        env.remove("PYTHONHOME")
        env.remove("PYTHONPATH")
        
        return env
    
    @staticmethod
    def validate_venv(venv_path):
        required = [
            venv_path / "Scripts" / "python.exe" if sys.platform == "win32" 
            else venv_path / "bin" / "python",
            venv_path / "Lib" / "site-packages"
        ]
        return all(p.exists() for p in required)

# ====================== 文件列表项组件 ======================
class FileListItem(QWidget):
    def __init__(self, filename, size):
        super().__init__()
        self._filename = filename
        self._size = int(size) if str(size).isdigit() else 0  # 强化类型转换
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 2, 5, 2)
        
        self.name_label = QLabel(self._filename)
        self.size_label = QLabel(self.format_size())
        self.size_label.setStyleSheet("color: #666;")
        
        layout.addWidget(self.name_label, 70)
        layout.addWidget(self.size_label, 30)
        self.setLayout(layout)

    def format_size(self):
        size = self._size
        # 增加单位换算容错
        try:
            if size >= 1024**3:
                return f"{size/1024**3:.1f} GB"
            elif size >= 1024**2:
                return f"{size/1024**2:.1f} MB"
            elif size >= 1024:
                return f"{size/1024:.1f} KB"
            return f"{size} B"
        except:
            return "未知大小"

    @property
    def filename(self):
        return self._filename

    @property
    def size(self):
        return self._size

# ====================== 数据模型类 ======================
class HuggingFaceModel:
    def __init__(self, data):
        self.modelId = data.get('modelId', '')
        self.tags = data.get('tags', [])
        self.downloads = data.get('downloads', 0)
        self.likes = data.get('likes', 0)
        self.pipeline_tag = data.get('pipeline_tag', 'other')
        self.siblings = data.get('siblings', [])

        self._total_size = 0
        for f in self.siblings:
            raw_size = f.get('size', 0)
            try:
                self._total_size += int(raw_size)
            except (ValueError, TypeError):
                pass

    @property
    def formatted_size(self):
        size = self._total_size
        if size >= 1024**3:
            return f"{size/1024**3:.1f} GB"
        elif size >= 1024**2:
            return f"{size/1024**2:.1f} MB"
        elif size >= 1024:
            return f"{size/1024:.1f} KB"
        return f"{size} B"

    @property
    def main_category(self):
        categories = [
            'text-generation', 
            'image-classification',
            'speech-recognition',
            'object-detection',
            'text-to-image',
            'text-to-audio'
        ]
        return next((tag for tag in self.tags if tag in categories), 'other')

# ====================== 线程类 ======================
class ModelSearchThread(QThread):
    search_complete = pyqtSignal(list)
    search_failed = pyqtSignal(str)

    def __init__(self, search_text, config, sort_by="downloads"):
        super().__init__()
        self.search_text = search_text
        self.config = config
        self.sort_by = sort_by

    def run(self):
        try:
            proxies = {}
            if self.config.proxy["http"]:
                proxies = {
                    "http": self.config.proxy["http"],
                    "https": self.config.proxy["https"]
                }

            params = {
                "search": self.search_text,
                "sort": self.sort_by,
                "direction": "-1",
                "limit": 100
            }
            response = requests.get(
                "https://huggingface.co/api/models",
                params=params,
                proxies=proxies,
                timeout=15
            )
            response.raise_for_status()
            models = [HuggingFaceModel(model) for model in response.json()]
            self.search_complete.emit(models)
        except Exception as e:
            self.search_failed.emit(str(e))

class ModelDetailThread(QThread):
    detail_loaded = pyqtSignal(dict)
    detail_failed = pyqtSignal(str)

    def __init__(self, model_id, config):
        super().__init__()
        self.model_id = model_id
        self.config = config

    def run(self):
        try:
            proxies = {}
            if self.config.proxy["http"]:
                proxies = {
                    "http": self.config.proxy["http"],
                    "https": self.config.proxy["https"]
                }

            model_url = f"https://huggingface.co/api/models/{self.model_id}"
            response = requests.get(model_url, proxies=proxies, timeout=10)
            response.raise_for_status()
            model_data = response.json()

            readme_url = f"https://huggingface.co/{self.model_id}/raw/main/README.md"
            readme_response = requests.get(readme_url, proxies=proxies, timeout=10)
            readme_content = readme_response.text if readme_response.status_code == 200 else ""

            result = {
                "metadata": model_data,
                "readme": readme_content,
                "model_size": sum(f.get('size', 0) for f in model_data.get('siblings', []))
            }
            self.detail_loaded.emit(result)
        except Exception as e:
            self.detail_failed.emit(str(e))

# ====================== 下载管理类 ======================
class DownloadTask(QThread):
    progress_updated = pyqtSignal(int)
    speed_updated = pyqtSignal(str)
    status_updated = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, model_id, file_info, config):
        super().__init__()
        self.model_id = model_id
        self.file_info = file_info
        self.config = config
        self._is_paused = False
        self._is_canceled = False
        self.downloaded_bytes = 0
        self.total_size = file_info.get("size", 0)
        self.response = None

    def run(self):
        save_path = ""
        try:
            file_url = f"https://huggingface.co/{self.model_id}/resolve/main/{self.file_info['rfilename']}"
            proxies = {}
            if self.config.proxy["http"]:
                proxies = {
                    "http": self.config.proxy["http"],
                    "https": self.config.proxy["https"]
                }

            headers = {"User-Agent": "AI-Toolkit-Pro/1.0"}
            self.response = requests.get(
                file_url, 
                stream=True, 
                proxies=proxies, 
                headers=headers,
                timeout=10
            )
            self.response.raise_for_status()
            
            self.total_size = int(self.response.headers.get("content-length", self.total_size))
            if "stt/" in self.file_info['rfilename']:  # 根据模型类型调整路径
                save_path = os.path.join(self.config.model_path, "stt", self.file_info["rfilename"])
            else:
                save_path = os.path.join(self.config.model_path, self.file_info["rfilename"])
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            start_time = time.time()
            last_update_time = start_time
            downloaded_since_last = 0
            
            with open(save_path, "wb") as f:
                for chunk in self.response.iter_content(chunk_size=8192*4):
                    if self._is_canceled:
                        break
                    while self._is_paused:
                        time.sleep(0.5)
                    if not chunk:
                        continue
                    
                    f.write(chunk)
                    self.downloaded_bytes += len(chunk)
                    downloaded_since_last += len(chunk)
                    
                    progress = int((self.downloaded_bytes / self.total_size) * 100)
                    self.progress_updated.emit(progress)
                    
                    current_time = time.time()
                    if current_time - last_update_time >= 0.5:
                        elapsed = current_time - last_update_time
                        speed = downloaded_since_last / elapsed
                        
                        if speed >= 1024*1024:
                            speed_str = f"{speed/(1024*1024):.1f} MB/s"
                        elif speed >= 1024:
                            speed_str = f"{speed/1024:.1f} KB/s"
                        else:
                            speed_str = f"{speed:.1f} B/s"
                        
                        self.speed_updated.emit(speed_str)
                        
                        last_update_time = current_time
                        downloaded_since_last = 0

            if not self._is_canceled:
                self.status_updated.emit("下载完成")
                self.finished.emit()
                self.config.download_history.append({
                    "model_id": self.model_id,
                    "file": self.file_info["rfilename"],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                self.config.save_settings()

        except Exception as e:
            self.status_updated.emit(f"下载失败: {str(e)}")
            if save_path and os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except Exception as delete_error:
                    print(f"清理失败文件时出错: {delete_error}")
        finally:
            if self.response:
                self.response.close()

    def pause(self):
        self._is_paused = True
        self.status_updated.emit("已暂停")

    def resume(self):
        self._is_paused = False
        self.status_updated.emit("恢复下载...")

    def cancel(self):
        self._is_canceled = True
        self.status_updated.emit("已取消")
        
        if self.response:
            try:
                self.response.close()
            except Exception as e:
                print(f"关闭连接时出错: {e}")
        
        save_path = os.path.join(self.config.model_path, self.file_info["rfilename"])
        if not os.path.exists(save_path):
            return

        max_retries = 3
        for attempt in range(max_retries):
            try:
                os.remove(save_path)
                break
            except PermissionError:
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt+1))
                else:
                    print(f"最终删除失败: {save_path}")
            except Exception as e:
                print(f"删除文件时发生意外错误: {e}")
                break

class DownloadManager(QObject):
    task_added = pyqtSignal(object)
    total_progress = pyqtSignal(int)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.active_tasks = []
        self.completed_tasks = []
        self.total_bytes = 0
        self.downloaded_bytes = 0

    def add_task(self, model_id, file_info):
        if len(self.active_tasks) >= MAX_CONCURRENT:
            return None
        
        save_path = os.path.join(self.config.model_path, file_info["rfilename"])
        if os.path.exists(save_path):
            return None
        
        task = DownloadTask(model_id, file_info, self.config)
        task.progress_updated.connect(self.update_total_progress)
        task.finished.connect(lambda: self._move_to_completed(task))
        self.active_tasks.append(task)
        self.total_bytes += task.total_size
        self.task_added.emit(task)
        task.start()
        return task

    def update_total_progress(self, progress):
        current_downloaded = sum(t.downloaded_bytes for t in self.active_tasks)
        total_progress = int((current_downloaded / self.total_bytes) * 100) if self.total_bytes else 0
        self.total_progress.emit(total_progress)

    def _move_to_completed(self, task):
        try:
            self.active_tasks.remove(task)
            self.completed_tasks.append(task)
            self.total_bytes -= task.total_size
        except ValueError:
            pass

# ====================== 终端模拟器组件 ======================
class TerminalTab(QWidget):
    def __init__(self, terminal_type):
        super().__init__()
        self.terminal_type = terminal_type
        self.process = QProcess()
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont("Consolas", 10))
        
        self.input = QLineEdit()
        self.input.returnPressed.connect(self.execute_command)
        
        layout.addWidget(self.output)
        layout.addWidget(self.input)
        self.setLayout(layout)
        
        self.process.readyReadStandardOutput.connect(self.handle_output)
        self.process.readyReadStandardError.connect(self.handle_error)
        if sys.platform == "win32":
            self.process.start(self.terminal_type)
        else:
            self.process.start("/bin/bash")

    def execute_command(self):
        cmd = self.input.text()
        self.process.write(cmd.encode() + b"\n")
        self.input.clear()

    def handle_output(self):
        data = self.process.readAllStandardOutput().data().decode()
        self.output.appendPlainText(data)
        self.output.moveCursor(QTextCursor.End)

    def handle_error(self):
        data = self.process.readAllStandardError().data().decode()
        self.output.appendPlainText(f"Error: {data}")
        self.output.moveCursor(QTextCursor.End)

class CommandLinePage(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.terminals = QTabWidget()
        self.terminals.setTabsClosable(True)
        self.terminals.tabCloseRequested.connect(self.close_terminal)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        control_layout = QHBoxLayout()
        self.terminal_combo = QComboBox()
        self.terminal_combo.addItems(["cmd", "PowerShell", "bash"])
        self.terminal_combo.setCurrentText(self.config.terminal_type)
        
        new_btn = QPushButton("新建终端")
        new_btn.clicked.connect(self.new_terminal)
        control_layout.addWidget(QLabel("终端类型:"))
        control_layout.addWidget(self.terminal_combo)
        control_layout.addWidget(new_btn)
        
        layout.addLayout(control_layout)
        layout.addWidget(self.terminals)
        self.setLayout(layout)

    def new_terminal(self):
        term_type = self.terminal_combo.currentText()
        tab = TerminalTab(term_type)
        self.terminals.addTab(tab, f"{term_type} {self.terminals.count()+1}")
        self.config.terminal_type = term_type
        self.config.save_settings()

    def close_terminal(self, index):
        widget = self.terminals.widget(index)
        widget.process.terminate()
        self.terminals.removeTab(index)

# ====================== 设置页面 ======================
class SettingsPage(QWidget):
    config_updated = pyqtSignal()

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init_ui()
        self.setMinimumWidth(400)

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)

        storage_group = QGroupBox("存储设置")
        storage_layout = QFormLayout()
        
        self.path_edit = QLineEdit()
        self.path_edit.setText(self.config.model_path)
        path_btn = QPushButton("浏览...")
        path_btn.clicked.connect(self.select_path)
        
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.path_edit, stretch=4)
        path_layout.addWidget(path_btn, stretch=1)
        storage_layout.addRow("模型存储路径：", path_layout)

        self.storage_info = QLabel()
        self.update_storage_info()
        storage_layout.addRow("存储空间：", self.storage_info)
        
        storage_group.setLayout(storage_layout)
        main_layout.addWidget(storage_group)

        network_group = QGroupBox("网络设置")
        network_layout = QFormLayout()

        self.http_proxy_input = QLineEdit()
        self.http_proxy_input.setPlaceholderText("http://proxy.example.com:8080")
        self.http_proxy_input.setText(self.config.proxy["http"])
        
        self.https_proxy_input = QLineEdit()
        self.https_proxy_input.setPlaceholderText("https://proxy.example.com:8080")
        self.https_proxy_input.setText(self.config.proxy["https"])

        test_proxy_btn = QPushButton("测试代理")
        test_proxy_btn.clicked.connect(self.test_proxy)
        
        network_layout.addRow("HTTP 代理：", self.http_proxy_input)
        network_layout.addRow("HTTPS 代理：", self.https_proxy_input)
        network_layout.addRow("", test_proxy_btn)
        
        network_group.setLayout(network_layout)
        main_layout.addWidget(network_group)

        appearance_group = QGroupBox("外观设置")
        appearance_layout = QVBoxLayout()

        self.theme_combo = QComboBox()
        self.theme_combo.addItem("🌞 浅色模式", "light")
        self.theme_combo.addItem("🌙 深色模式", "dark")
        self.theme_combo.setCurrentIndex(1 if self.config.dark_mode else 0)
        
        self.font_size = QSpinBox()
        self.font_size.setRange(10, 24)
        self.font_size.setValue(self.config.font_size)
        
        appearance_form = QFormLayout()
        appearance_form.addRow("界面主题：", self.theme_combo)
        appearance_form.addRow("字体大小：", self.font_size)
        
        appearance_group.setLayout(appearance_form)
        main_layout.addWidget(appearance_group)

        btn_layout = QHBoxLayout()
        save_btn = QPushButton("💾 保存设置")
        save_btn.clicked.connect(self.save_settings)
        reset_btn = QPushButton("🔄 恢复默认")
        reset_btn.clicked.connect(self.reset_settings)
        
        btn_layout.addWidget(reset_btn)
        btn_layout.addWidget(save_btn)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    def select_path(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "选择存储目录",
            self.path_edit.text(),
            QFileDialog.ShowDirsOnly
        )
        if path:
            self.path_edit.setText(os.path.normpath(path))
            self.update_storage_info()

    def update_storage_info(self):
        path = self.path_edit.text()
        try:
            usage = shutil.disk_usage(path)
            total = usage.total / (1024**3)
            used = (usage.total - usage.free) / (1024**3)
            self.storage_info.setText(
                f"已用 {used:.1f}GB / 总共 {total:.1f}GB "
                f"({usage.used/usage.total:.0%})"
            )
        except Exception as e:
            self.storage_info.setText("无法获取存储信息")

    def test_proxy(self):
        proxies = {}
        if self.http_proxy_input.text():
            proxies = {
                "http": self.http_proxy_input.text(),
                "https": self.https_proxy_input.text()
            }

        try:
            start = time.time()
            response = requests.get(
                "https://huggingface.co/api/status",
                proxies=proxies,
                timeout=5
            )
            latency = (time.time() - start) * 1000
            if response.status_code == 200:
                QMessageBox.information(
                    self, 
                    "代理测试成功",
                    f"连接成功！响应时间：{latency:.0f}ms\n"
                    f"服务状态：{response.json().get('status')}"
                )
            else:
                QMessageBox.warning(
                    self,
                    "代理测试失败",
                    f"服务器返回错误：{response.status_code}"
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "代理测试失败",
                f"无法连接到服务器：\n{str(e)}"
            )

    def save_settings(self):
        if not os.access(self.path_edit.text(), os.W_OK):
            QMessageBox.critical(self, "错误", "存储路径不可写！")
            return

        self.config.model_path = self.path_edit.text()
        self.config.dark_mode = self.theme_combo.currentData() == "dark"
        self.config.font_size = self.font_size.value()
        self.config.proxy = {
            "http": self.http_proxy_input.text(),
            "https": self.https_proxy_input.text()
        }
        self.config.save_settings()
        self.config_updated.emit()
        QMessageBox.information(self, "成功", "设置已保存")

    def reset_settings(self):
        default_path = os.path.abspath("models")
        self.path_edit.setText(default_path)
        self.theme_combo.setCurrentIndex(0)
        self.font_size.setValue(12)
        self.http_proxy_input.clear()
        self.https_proxy_input.clear()
        self.update_storage_info()

# ====================== 模型列表项 ======================
class ModelListItem(QWidget):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.init_ui()
        self.apply_style()

    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(10, 5, 10, 5)
        
        text_layout = QVBoxLayout()
        
        self.title_label = QLabel(f"<b>{self.model.modelId}</b>")
        self.stats_label = QLabel(f"♥️ {self.model.likes}  ↓ {self.model.downloads}")
        self.size_label = QLabel(f"📦 {self.model.formatted_size}")
        
        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.stats_label)
        text_layout.addWidget(self.size_label)
        
        layout.addLayout(text_layout)
        self.setLayout(layout)

    def apply_style(self):
        palette = self.palette()
        text_color = palette.text().color().name()
        self.title_label.setStyleSheet(f"""
            font-size: 14px;
            color: {text_color};
        """)
        self.stats_label.setStyleSheet(f"""
            color: {text_color};
            font-size: 12px;
            opacity: 0.8;
        """)
        self.size_label.setStyleSheet(f"""
            color: {text_color};
            font-size: 12px;
            opacity: 0.8;
        """)

# ====================== 模型详情对话框 ======================
class ModelDetailDialog(QDialog):
    def __init__(self, model_id, config, download_manager):
        super().__init__()
        self.model_id = model_id
        self.config = config
        self.download_manager = download_manager
        self.model_data = None
        self.setWindowTitle("模型详情")
        self.setMinimumSize(800, 600)
        self.init_ui()
        self.load_data()
        self.setWindowModality(Qt.NonModal)
        self.apply_theme()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        header = QWidget()
        header.setObjectName("detailHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 10, 20, 10)
        
        info_layout = QVBoxLayout()
        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("color: #666;")
        
        self.download_btn = QPushButton("Git下载完整模型")
        self.download_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #45a049; }
        """)
        self.download_btn.clicked.connect(self.git_download)
        
        info_layout.addWidget(self.title_label)
        info_layout.addWidget(self.stats_label)
        info_layout.addWidget(self.download_btn)
        
        header_layout.addLayout(info_layout)
        
        self.tab_widget = QTabWidget()
        
        self.doc_view = QWebEngineView()
        self.tab_widget.addTab(self.doc_view, "文档")
        
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.itemDoubleClicked.connect(self.on_file_double_clicked)
        self.tab_widget.addTab(self.file_list, "文件")
        
        layout.addWidget(header)
        layout.addWidget(self.tab_widget)
        self.setLayout(layout)
        
        self.setStyleSheet("""
            QDialog {
                background-color: palette(window);
                border-radius: 8px;
            }
            #detailHeader {
                background-color: palette(base);
                border-bottom: 1px solid palette(mid);
            }
        """)

    def apply_theme(self):
        if self.config.dark_mode:
            self.setStyleSheet("""
                QDialog {
                    background-color: #333;
                    color: #fff;
                }
                #detailHeader {
                    background-color: #2d2d2d;
                    border-bottom: 1px solid #1a1a1a;
                }
                QListWidget {
                    background-color: #404040;
                    color: #fff;
                    border: none;
                }
                QWebEngineView {
                    background-color: #333;
                    color: #fff;
                }
            """)
        else:
            self.setStyleSheet("""
                QDialog {
                    background-color: white;
                    color: #333;
                }
                #detailHeader {
                    background-color: #f0f0f0;
                    border-bottom: 1px solid #ddd;
                }
                QListWidget {
                    background-color: white;
                    color: #333;
                    border: 1px solid #ddd;
                }
                QWebEngineView {
                    background-color: white;
                    color: #333;
                }
            """)

    def load_data(self):
        self.thread = ModelDetailThread(self.model_id, self.config)
        self.thread.detail_loaded.connect(self.update_ui)
        self.thread.detail_failed.connect(self.show_error)
        self.thread.start()

    def update_ui(self, data):
        self.model_data = data
        model = HuggingFaceModel(data['metadata'])
        
        self.title_label.setText(model.modelId)
        self.stats_label.setText(f"❤️ {model.likes} 下载量: {model.downloads} 大小: {model.formatted_size}")
        
        # 动态生成带样式的Markdown内容
        if self.config.dark_mode:
            css = """
                <style>
                    body { 
                        background-color: #333;
                        color: #fff; 
                        font-family: Arial;
                        padding: 20px;
                    }
                    h1, h2, h3 { color: #4CAF50; }
                    code { 
                        background-color: #404040;
                        padding: 2px 4px;
                        border-radius: 3px;
                    }
                    pre {
                        background-color: #404040;
                        padding: 10px;
                        border-radius: 5px;
                        overflow-x: auto;
                    }
                    a { color: #4CAF50; }
                </style>
            """
        else:
            css = """
                <style>
                    body { 
                        background-color: white;
                        color: #333; 
                        font-family: Arial;
                        padding: 20px;
                    }
                    code { 
                        background-color: #f0f0f0;
                        padding: 2px 4px;
                        border-radius: 3px;
                    }
                    pre {
                        background-color: #f0f0f0;
                        padding: 10px;
                        border-radius: 5px;
                        overflow-x: auto;
                    }
                </style>
            """
        
        html_content = markdown.markdown(data['readme'])
        full_html = f"<html><head>{css}</head><body>{html_content}</body></html>"
        self.doc_view.setHtml(full_html)
        
        self.file_list.clear()
        for file in data['metadata'].get('siblings', []):
            # 强化文件大小处理
            file_size = file.get('size', 0)
            try:
                file_size = int(file_size)
            except (ValueError, TypeError):
                file_size = 0
                
            item = QListWidgetItem()
            widget = FileListItem(file['rfilename'], file_size)  # 传入处理后的size
            item.setSizeHint(widget.sizeHint())
            self.file_list.addItem(item)
            self.file_list.setItemWidget(item, widget)

        self.file_list.clear()
        for file in data['metadata'].get('siblings', []):
            # 强化文件大小处理
            file_size = file.get('size', 0)
            try:
                file_size = int(file_size)
            except (ValueError, TypeError):
                file_size = 0
                
            item = QListWidgetItem()
            widget = FileListItem(file['rfilename'], file_size)
            item.setSizeHint(widget.sizeHint())
            self.file_list.addItem(item)
            self.file_list.setItemWidget(item, widget)

    def show_error(self, error):
        QMessageBox.critical(self, "错误", f"加载详情失败: {error}")

    def git_download(self):
        model_url = f"https://huggingface.co/{self.model_id}"
        save_path = os.path.join(self.config.model_path, self.model_id.split('/')[-1])
    
        if os.path.exists(save_path):
            QMessageBox.warning(self, "警告", "模型目录已存在！")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Git下载进度")
        layout = QVBoxLayout()
        output = QPlainTextEdit()
        output.setReadOnly(True)
        output.setFont(QFont("Consolas", 10))
        btn = QPushButton("取消")
        layout.addWidget(output)
        layout.addWidget(btn)
        dlg.setLayout(layout)
    
        process = QProcess()
        process.setWorkingDirectory(self.config.model_path)
        process.start("git", ["clone", "--progress", model_url])
    
        def update_output():
            out_data = process.readAllStandardOutput().data().decode(errors='ignore')
            err_data = process.readAllStandardError().data().decode(errors='ignore')
            if out_data:
                output.appendPlainText(out_data.strip())
            if err_data:
                output.appendPlainText(err_data.strip())
            output.moveCursor(QTextCursor.End)
            QApplication.processEvents()
    
        process.readyReadStandardOutput.connect(update_output)
        process.readyReadStandardError.connect(update_output)
    
        def handle_finish():
            if process.exitStatus() == QProcess.NormalExit:
                output.appendPlainText("\n✅ 下载完成！")
                self.config.download_history.append({
                    "model_id": self.model_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                self.config.save_settings()
            else:
                output.appendPlainText(f"\n❌ 错误: {process.readAllStandardError().data().decode()}")
    
        process.finished.connect(handle_finish)
        btn.clicked.connect(lambda: process.terminate())
        dlg.exec_()

    def on_file_double_clicked(self, item):
        widget = self.file_list.itemWidget(item)
        if widget:
            filename = widget.filename
            for file_info in self.model_data['metadata']['siblings']:
                if file_info['rfilename'] == filename:
                    task = self.download_manager.add_task(self.model_id, file_info)
                    if task is None:
                        QMessageBox.information(self, "提示", "该文件已在下载队列中或已存在。")
                    return
            QMessageBox.warning(self, "错误", "文件信息缺失。")

# ====================== 下载管理页面 ======================
class DownloadPage(QWidget):
    def __init__(self, download_manager):
        super().__init__()
        self.download_manager = download_manager
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        top_bar = QHBoxLayout()
        self.open_path_btn = QPushButton("打开模型路径")
        self.open_path_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.open_path_btn.clicked.connect(self.open_model_path)
        top_bar.addWidget(self.open_path_btn)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        self.task_list = QListWidget()
        layout.addWidget(QLabel("进行中的下载："))
        layout.addWidget(self.task_list)

        btn_layout = QHBoxLayout()
        self.pause_btn = QPushButton("暂停")
        self.resume_btn = QPushButton("继续")
        self.cancel_btn = QPushButton("取消")
        self.pause_btn.clicked.connect(self.pause_task)
        self.resume_btn.clicked.connect(self.resume_task)
        self.cancel_btn.clicked.connect(self.cancel_task)
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.resume_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self.download_manager.task_added.connect(self.add_task_item)

    class DownloadItemWidget(QWidget):
        def __init__(self, task_name):
            super().__init__()
            self.layout = QHBoxLayout()
            
            self.name_label = QLabel(task_name)
            self.name_label.setMinimumWidth(150)
            self.layout.addWidget(self.name_label, stretch=4)
            
            self.progress_bar = QProgressBar()
            self.progress_bar.setMaximum(100)
            self.progress_bar.setTextVisible(False)
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #ddd;
                    border-radius: 3px;
                    height: 16px;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                    border-radius: 2px;
                }
            """)
            self.layout.addWidget(self.progress_bar, stretch=4)
            
            self.speed_label = QLabel("0 KB/s")
            self.speed_label.setAlignment(Qt.AlignRight)
            self.speed_label.setStyleSheet("color: #666;")
            self.layout.addWidget(self.speed_label, stretch=2)
            
            self.setLayout(self.layout)

    def add_task_item(self, task):
        item_widget = self.DownloadItemWidget(task.model_id)
        item = QListWidgetItem()
        item.setSizeHint(item_widget.sizeHint())
        
        task.progress_updated.connect(item_widget.progress_bar.setValue)
        task.speed_updated.connect(
            lambda speed: item_widget.speed_label.setText(
                f"{speed} " if "MB" in speed else f"{speed} "  # 保持对齐
            )
        )
        task.status_updated.connect(
            lambda status: item_widget.name_label.setText(
                f"{task.model_id} - {status}"
            )
        )
        
        self.task_list.addItem(item)
        self.task_list.setItemWidget(item, item_widget)

    def open_model_path(self):
        model_path = self.download_manager.config.model_path
        if not os.path.exists(model_path):
            QMessageBox.critical(self, "错误", f"路径不存在: {model_path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(model_path))

    def pause_task(self):
        if self.task_list.currentRow() >= 0:
            task = self.download_manager.active_tasks[self.task_list.currentRow()]
            task.pause()

    def resume_task(self):
        if self.task_list.currentRow() >= 0:
            task = self.download_manager.active_tasks[self.task_list.currentRow()]
            task.resume()

    def cancel_task(self):
        if self.task_list.currentRow() >= 0:
            task = self.download_manager.active_tasks[self.task_list.currentRow()]
            task.cancel()

# ====================== 分类导航栏 ======================
class ModelCategoryBar(QWidget):
    category_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 5, 0, 5)
        
        categories = [
            ("全部", "all"),
            ("热门模型", "hot"),
            ("文本生成", "text-generation"),
            ("文生图", "text-to-image"),
            ("文生语音", "text-to-audio"),
            ("图像分类", "image-classification"),
            ("语音识别", "speech-recognition"),
            ("目标检测", "object-detection")
        ]

        self.btn_group = QButtonGroup(self)
        for name, tag in categories:
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setProperty('category', tag)
            btn.setStyleSheet("""
                QPushButton {
                    padding: 8px 12px;
                    margin: 2px;
                    border-radius: 4px;
                    border: 1px solid #ddd;
                }
                QPushButton:checked {
                    background-color: #4CAF50;
                    color: white;
                }
            """)
            btn.clicked.connect(self.on_category_click)
            self.btn_group.addButton(btn)
            layout.addWidget(btn)

        self.btn_group.buttons()[0].setChecked(True)
        layout.addStretch()
        self.setLayout(layout)

    def on_category_click(self):
        sender = self.sender()
        self.category_changed.emit(sender.property('category'))

# ====================== 模型中心页面 ======================
class ModelDownloadPage(QWidget):
    def __init__(self, config, download_manager):
        super().__init__()
        self.config = config
        self.download_manager = download_manager
        self.current_models = []
        self.current_category = "all"
        self.init_ui()
        self.load_hot_models()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.category_bar = ModelCategoryBar()
        self.category_bar.category_changed.connect(self.filter_models)
        layout.addWidget(self.category_bar)
        
        search_layout = QHBoxLayout()
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("输入模型名称搜索...")
        self.search_bar.returnPressed.connect(self.search_models)
        self.search_btn = QPushButton("搜索")
        self.search_btn.clicked.connect(self.search_models)
        
        search_layout.addWidget(self.search_bar, stretch=4)
        search_layout.addWidget(self.search_btn, stretch=1)
        layout.addLayout(search_layout)
        
        self.model_list = QListWidget()
        self.model_list.itemDoubleClicked.connect(self.show_detail)
        layout.addWidget(self.model_list)
        
        self.status_label = QLabel("就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        self.apply_styles()

    def load_hot_models(self):
        self.status_label.setText("正在加载热门模型...")
        self.thread = ModelSearchThread("", self.config, sort_by="downloads")
        self.thread.search_complete.connect(self.handle_hot_models)
        self.thread.search_failed.connect(self.handle_search_error)
        self.thread.start()

    def handle_hot_models(self, models):
        self.current_models = models
        self.filter_models("hot")
        self.status_label.setText(f"找到 {len(models)} 个热门模型")

    def apply_styles(self):
        font = QFont(self.config.font_family, self.config.font_size)
        self.search_bar.setFont(font)
        self.status_label.setFont(font)
        if self.config.dark_mode:
            search_style = """
                QLineEdit {
                    padding: 8px;
                    border: 2px solid #4CAF50;
                    border-radius: 4px;
                    background-color: #333;
                    color: white;
                    font-size: 14px;
                }
            """
        else:
            search_style = """
                QLineEdit {
                    padding: 8px;
                    border: 2px solid #4CAF50;
                    border-radius: 4px;
                    background-color: white;
                    color: black;
                    font-size: 14px;
                }
            """
        self.search_bar.setStyleSheet(search_style)

    def search_models(self):
        query = self.search_bar.text().strip()
        if not query:
            return

        self.status_label.setText("搜索中...")
        self.search_btn.setEnabled(False)

        self.thread = ModelSearchThread(query, self.config)
        self.thread.search_complete.connect(self.handle_search_result)
        self.thread.search_failed.connect(self.handle_search_error)
        self.thread.start()

    def handle_search_result(self, models):
        self.current_models = models
        self.filter_models(self.current_category)
        self.status_label.setText(f"找到 {len(models)} 个模型")
        self.search_btn.setEnabled(True)

    def handle_search_error(self, error):
        QMessageBox.critical(self, "错误", f"搜索失败: {error}")
        self.status_label.setText("搜索失败")
        self.search_btn.setEnabled(True)

    def filter_models(self, category):
        self.current_category = category
        if category == "hot":
            filtered = sorted(self.current_models, 
                             key=lambda x: x.downloads, 
                             reverse=True)[:50]
        else:
            filtered = [m for m in self.current_models 
                       if category == "all" or category == m.main_category]
        
        self.model_list.clear()
        for model in filtered[:50]:
            item = QListWidgetItem()
            widget = ModelListItem(model, self.config)
            item.setSizeHint(widget.sizeHint())
            self.model_list.addItem(item)
            self.model_list.setItemWidget(item, widget)

    def show_detail(self, item):
        widget = self.model_list.itemWidget(item)
        dialog = ModelDetailDialog(widget.model.modelId, self.config, self.download_manager)
        dialog.setParent(self.window(), Qt.Dialog)
        dialog.show()

#===================== 语音转文本页面 ======================
class STTPage(QWidget):
    def __init__(self, config, download_manager):
        super().__init__()
        self.config = config
        self.download_manager = download_manager
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout()
        
        # 模型控制栏
        control_layout = QHBoxLayout()
        self.model_combo = QComboBox()
        self.gpu_check = QCheckBox("GPU加速")
        self.load_btn = QPushButton("加载模型")
        self.load_btn.clicked.connect(self.load_model)
        
        control_layout.addWidget(QLabel("选择模型:"))
        control_layout.addWidget(self.model_combo)
        control_layout.addWidget(self.gpu_check)
        control_layout.addWidget(self.load_btn)
        layout.addLayout(control_layout)
        
        # 日志显示
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view)
        
        self.setLayout(layout)
        self.scan_models()
        
    def scan_models(self):
        self.model_combo.clear()
        model_dir = Path(self.config.model_path) / "stt"
        if model_dir.exists():
            for model in model_dir.glob("*"):
                if model.is_dir() and (model / "model.bin").exists():
                    self.model_combo.addItem(model.name)

    def load_model(self):
        model_name = self.model_combo.currentText()
        use_gpu = self.gpu_check.isChecked()
        model_path = Path(self.config.model_path) / "stt" / model_name
        
        process = QProcess()
        process.setProgram(sys.executable)
        process.setArguments([
            "stt_api.py",
            "--model_path", str(model_path),
            "--use_gpu" if use_gpu else "--use_cpu"
        ])
        
        process.readyReadStandardOutput.connect(
            lambda: self.log_view.appendPlainText(process.readAllStandardOutput().data().decode()))
        process.readyReadStandardError.connect(
            lambda: self.log_view.appendPlainText(process.readAllStandardError().data().decode()))
            
        process.start()

# ====================== 主窗口 ======================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        model_path = ConfigManager.instance().model_path
        if not os.path.exists(model_path):
            try:
                os.makedirs(model_path, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"无法创建模型目录: {str(e)}")
        VirtualEnvManager.get_python_path()
        self.config = ConfigManager.instance()
        self.download_manager = DownloadManager(self.config)
        self.init_ui()
        self.setWindowTitle("RIL")
        self.setGeometry(100, 100, 1000, 600) 
        self.apply_theme()
        self.settings_page.config_updated.connect(self.on_config_updated)
        self.setWindowIcon(QIcon("./assets/icons/placeholder.png"))
        self.oldPos = None
        self.draggable = True
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setWindowIcon(QIcon(APP_ICON_PATH))
        self.stt_page = STTPage(self.config, self.download_manager)
        self.add_module("🎤 STT", self.stt_page)

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.title_bar = QWidget()
        self.title_bar.setFixedHeight(35)
        self.title_bar.setMouseTracking(True)
        title_bar_layout = QHBoxLayout(self.title_bar)
        title_bar_layout.setContentsMargins(10, 0, 10, 0)

        self.title_label = QLabel("RIL")
        self.title_label.setStyleSheet("font-weight: bold;")

        self.min_btn = QPushButton("—")
        self.max_btn = QPushButton("□")
        self.close_btn = QPushButton("×")

        self.min_btn.setFixedSize(30, 30)
        self.max_btn.setFixedSize(30, 30)
        self.close_btn.setFixedSize(30, 30)

        self.min_btn.clicked.connect(self.showMinimized)
        self.max_btn.clicked.connect(self.toggle_maximized)
        self.close_btn.clicked.connect(self.close)

        title_bar_layout.addWidget(self.title_label)
        title_bar_layout.addStretch()
        title_bar_layout.addWidget(self.min_btn)
        title_bar_layout.addWidget(self.max_btn)
        title_bar_layout.addWidget(self.close_btn)

        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.nav_list = QListWidget()
        self.nav_list.setFixedWidth(200)
        self.nav_list.setFocusPolicy(Qt.NoFocus)

        self.stacked_widget = QStackedWidget()

        self.toggle_btn = QPushButton("☰")
        self.toggle_btn.setFixedSize(30, 30)
        self.toggle_btn.clicked.connect(self.toggle_sidebar)
        title_bar_layout.insertWidget(0, self.toggle_btn)

        self.home_page = HomePage(self.config)
        self.model_page = ModelDownloadPage(self.config, self.download_manager)
        self.settings_page = SettingsPage(self.config)
        self.download_page = DownloadPage(self.download_manager)
        self.command_line_page = CommandLinePage(self.config)
        self.chat_page = ChatPage(self.config) 
        
        self.add_module("🏠 首页", self.home_page)
        self.add_module("💬 模型对话", self.chat_page) 
        self.add_module("🏗️ 模型中心", self.model_page)
        self.add_module("⚙️ 系统设置", self.settings_page)
        self.add_module("⬇️ 下载管理", self.download_page)
        self.add_module("💻 命令行终端", self.command_line_page)
        
        self.nav_list.currentRowChanged.connect(self.stacked_widget.setCurrentIndex)
        
        content_layout.addWidget(self.nav_list)
        content_layout.addWidget(self.stacked_widget)

        main_layout.addWidget(self.title_bar)
        main_layout.addWidget(content_widget)

        self.status_bar = self.statusBar()
        self.theme_status = QLabel()
        self.proxy_status = QLabel()
        self.status_bar.addPermanentWidget(self.theme_status)
        self.status_bar.addPermanentWidget(self.proxy_status)

        self.setCentralWidget(main_widget)
        self.update_status()

        self.size_grips = [
            SizeGrip(self, Qt.LeftEdge | Qt.TopEdge),
            SizeGrip(self, Qt.RightEdge | Qt.TopEdge),
            SizeGrip(self, Qt.LeftEdge | Qt.BottomEdge),
            SizeGrip(self, Qt.RightEdge | Qt.BottomEdge),
            SizeGrip(self, Qt.TopEdge),
            SizeGrip(self, Qt.BottomEdge),
            SizeGrip(self, Qt.LeftEdge),
            SizeGrip(self, Qt.RightEdge)
        ]

    def add_module(self, name, widget):
        item = QListWidgetItem(name)
        item.setFont(QFont("Microsoft Yahei", 10))
        item.setSizeHint(QSize(200, 50))
        self.nav_list.addItem(item)
        self.stacked_widget.addWidget(widget)
        if self.nav_list.count() == 1:
            self.nav_list.setCurrentRow(0)

    def apply_theme(self):
        font = QFont(self.config.font_family, self.config.font_size)
        QApplication.setFont(font)
        if self.config.dark_mode:
            self.set_dark_theme()
            self.title_bar.setStyleSheet("""
                background-color: #2d2d2d;
                color: white;
                border-bottom: 1px solid #1a1a1a;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            """)
            btn_style = """
                QPushButton {
                    background: transparent;
                    border: none;
                    color: white;
                }
                QPushButton:hover {
                    background: #404040;
                }
            """
        else:
            self.set_light_theme()
            self.title_bar.setStyleSheet("""
                background-color: #f0f0f0;
                color: black;
                border-bottom: 1px solid #ddd;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            """)
            btn_style = """
                QPushButton {
                    background: transparent;
                    border: none;
                    color: black;
                }
                QPushButton:hover {
                    background: #e0e0e0;
                }
            """
        self.min_btn.setStyleSheet(btn_style)
        self.max_btn.setStyleSheet(btn_style)
        self.close_btn.setStyleSheet(btn_style)
        self.update_fonts()

    def update_fonts(self):
        font = QFont(self.config.font_family, self.config.font_size)
        self.setFont(font)
        for i in range(self.nav_list.count()):
            item = self.nav_list.item(i)
            item.setFont(font)
        self.update()
        QApplication.processEvents()

    def set_dark_theme(self):
        palette = QApplication.palette()
        palette.setColor(QPalette.Window, QColor(45, 45, 45))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.Highlight, QColor(76, 175, 80))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        QApplication.setPalette(palette)
        
        self.setStyleSheet("""
            QMainWindow {
                background: #2d2d2d;
            }
            QStatusBar {
                background: #1a1a1a;
                color: white;
            }
            QStatusBar::item { 
                border: none; 
            }
        """)

    def set_light_theme(self):
        palette = QApplication.palette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, Qt.black)
        palette.setColor(QPalette.Base, Qt.white)
        palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
        palette.setColor(QPalette.Text, Qt.black)
        palette.setColor(QPalette.Button, QColor(240, 240, 240))
        palette.setColor(QPalette.ButtonText, Qt.black)
        palette.setColor(QPalette.Highlight, QColor(76, 175, 80))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        QApplication.setPalette(palette)
        
        self.setStyleSheet("""
            QMainWindow { 
                background: #f8f9fa; 
            }
            QStatusBar {
                background: #f8f9fa;
                color: black;
            }
            QStatusBar::item { 
                border: none; 
            }
        """)

    def update_status(self):
        theme = "深色" if self.config.dark_mode else "浅色"
        proxy = "已启用" if self.config.proxy["http"] else "未启用"
        self.theme_status.setText(f"主题: {theme}")
        self.proxy_status.setText(f"代理: {proxy}")

    def toggle_sidebar(self):
        if self.nav_list.width() > 50:
            self.nav_list.setFixedWidth(50)
            for i in range(self.nav_list.count()):
                item = self.nav_list.item(i)
                item.setText("")
        else:
            self.nav_list.setFixedWidth(200)
            texts = ["🏠", "💬", "🏗️", "⚙️", "⬇️", "💻", "🎤", "🔊"]  # 原文本简化为图标
            for i in range(self.nav_list.count()):
                item = self.nav_list.item(i)
                item.setText(texts[i])

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
    
        # 使用QRectF代替QRect
        rect = QRectF(0, 0, self.width(), self.height())  # 无需显式转换float
        path.addRoundedRect(rect, 8, 8)
    
        painter.fillPath(path, self.palette().window())
    
        pen = QPen(QColor(100, 100, 100), 1)
        painter.setPen(pen)
        painter.drawPath(path)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.oldPos = event.globalPos()
            self.draggable = self.is_draggable_area(event.pos())

    def mouseMoveEvent(self, event):
        if self.draggable and self.oldPos and event.buttons() == Qt.LeftButton:
            delta = event.globalPos() - self.oldPos
            self.move(self.pos() + delta)
            self.oldPos = event.globalPos()
        else:
            self.update_cursor(event.pos())

    def mouseReleaseEvent(self, event):
        self.oldPos = None
        self.draggable = True

    def is_draggable_area(self, pos):
        return pos.y() < self.title_bar.height()

    def update_cursor(self, pos):
        edge = self.get_edge(pos)
        if edge == Qt.LeftEdge:
            self.setCursor(Qt.SizeHorCursor)
        elif edge == Qt.RightEdge:
            self.setCursor(Qt.SizeHorCursor)
        elif edge == Qt.TopEdge:
            self.setCursor(Qt.SizeVerCursor)
        elif edge == Qt.BottomEdge:
            self.setCursor(Qt.SizeVerCursor)
        elif edge == (Qt.LeftEdge | Qt.TopEdge):
            self.setCursor(Qt.SizeFDiagCursor)
        elif edge == (Qt.RightEdge | Qt.TopEdge):
            self.setCursor(Qt.SizeBDiagCursor)
        elif edge == (Qt.LeftEdge | Qt.BottomEdge):
            self.setCursor(Qt.SizeBDiagCursor)
        elif edge == (Qt.RightEdge | Qt.BottomEdge):
            self.setCursor(Qt.SizeFDiagCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def get_edge(self, pos):
        x, y = pos.x(), pos.y()
        w, h = self.width(), self.height()
        edge = 0
        if x <= BORDER_WIDTH:
            edge |= Qt.LeftEdge
        if x >= w - BORDER_WIDTH:
            edge |= Qt.RightEdge
        if y <= BORDER_WIDTH:
            edge |= Qt.TopEdge
        if y >= h - BORDER_WIDTH:
            edge |= Qt.BottomEdge
        return edge

    def toggle_maximized(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def on_config_updated(self):
        self.apply_theme()
        self.update_status()
        self.chat_page.scan_local_models()  # 配置更新后刷新本地模型列表

class SizeGrip(QWidget):
    def __init__(self, parent, edges):
        super().__init__(parent)
        self.edges = edges
        self.setMouseTracking(True)
        self.oldPos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.oldPos = event.globalPos()

    def mouseMoveEvent(self, event):
        if self.oldPos and event.buttons() == Qt.LeftButton:
            delta = event.globalPos() - self.oldPos
            rect = self.window().geometry()
            
            if self.edges & Qt.LeftEdge:
                rect.setLeft(rect.left() + delta.x())
            if self.edges & Qt.RightEdge:
                rect.setRight(rect.right() + delta.x())
            if self.edges & Qt.TopEdge:
                rect.setTop(rect.top() + delta.y())
            if self.edges & Qt.BottomEdge:
                rect.setBottom(rect.bottom() + delta.y())
            
            self.window().setGeometry(rect.normalized())
            self.oldPos = event.globalPos()
        else:
            self.update_cursor()

    def mouseReleaseEvent(self, event):
        self.oldPos = None

    def update_cursor(self):
        if self.edges == Qt.LeftEdge | Qt.TopEdge:
            self.setCursor(Qt.SizeFDiagCursor)
        elif self.edges == Qt.RightEdge | Qt.TopEdge:
            self.setCursor(Qt.SizeBDiagCursor)
        elif self.edges == Qt.LeftEdge | Qt.BottomEdge:
            self.setCursor(Qt.SizeBDiagCursor)
        elif self.edges == Qt.RightEdge | Qt.BottomEdge:
            self.setCursor(Qt.SizeFDiagCursor)
        elif self.edges & Qt.LeftEdge or self.edges & Qt.RightEdge:
            self.setCursor(Qt.SizeHorCursor)
        elif self.edges & Qt.TopEdge or self.edges & Qt.BottomEdge:
            self.setCursor(Qt.SizeVerCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def paintEvent(self, event):
        pass 

# ====================== 程序入口 ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(APP_ICON_PATH)) 
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
