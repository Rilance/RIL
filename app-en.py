# ====================== Import Dependencies ======================
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
from PyQt5.QtCore import QProcessEnvironment  # Ensure import
from PyQt5.QtWidgets import QLayout, QSizePolicy
from PyQt5.QtCore import QSize, QRect, QPoint

# ====================== Pre-configuration ======================
import warnings
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"sipPyTypeDict\(\) is deprecated.*",
)

# Get absolute resource path
def resource_path(relative_path):
    """ Get absolute path to resource """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

# ====================== Constant Definitions ======================
APP_NAME = "AIToolkitPro"
APP_VERSION = "1.0.0"
MAX_CONCURRENT = 3
BORDER_WIDTH = 5  # Window border width
MODEL_EXTENSIONS = ('.safetensors', '.bin', '.pth', '.pt', '.gguf')
APP_ICON_PATH = resource_path("assets/icons/icon.ico") 
user_prefix = "User: "
model_prefix = "Firefly: "

# ====================== Configuration Manager Class ======================
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
        self.load_settings()
        self.font_family = "Microsoft Yahei"
        self.font_size = 12
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

# ====================== Home Page Component ======================
class HomePage(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Background image
        bg_path = os.path.abspath("./assets/bk/114793143.png")
        if os.path.exists(bg_path):
            self.bg_label = QLabel()
            self.bg_label.setPixmap(QIcon(bg_path).pixmap(800, 400))
            self.bg_label.setAlignment(Qt.AlignCenter)
            self.bg_label.setStyleSheet("background: transparent;")
            layout.addWidget(self.bg_label)
        else:
            error_label = QLabel("Background image not found")
            error_label.setAlignment(Qt.AlignCenter)
            layout.addWidget(error_label)
        
        # Welcome text
        welcome_label = QLabel("Welcome to RIL\n(Rilance Intelligence Launcher)")
        welcome_label.setAlignment(Qt.AlignCenter)
        welcome_label.setStyleSheet("""
            font-size: 28px;
            font-weight: bold;
            color: #4CAF50;
            margin: 20px 0;
        """)
        layout.addWidget(welcome_label)
        
        self.setLayout(layout)

# ====================== Chat Page ======================
class ChatPage(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_process = None  # Model loading process
        self.init_ui()
        self.scan_local_models()
        self.output_buffer = ""
        self.output_timer = QTimer()
        self.output_timer.timeout.connect(self.flush_buffer)
        self.current_message = ""

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Top control bar
        control_layout = QHBoxLayout()
        
        # Model selection dropdown
        self.model_combo = QComboBox()
        self.model_combo.setFixedWidth(300)
        self.model_combo.setStyleSheet("QComboBox { padding: 5px; }")
        
        # Refresh button
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_BrowserReload))
        refresh_btn.clicked.connect(self.scan_local_models)
        
        # Load model button
        self.load_btn = QPushButton("Load Model")
        self.load_btn.setIcon(QIcon("./assets/icons/load.png"))
        self.load_btn.clicked.connect(self.load_model)
        self.load_btn.setEnabled(False)

        # Hardware acceleration options
        self.gpu_check = QCheckBox("GPU Acceleration")
        self.quant_check = QCheckBox("4-bit Quantization")

        control_layout.addWidget(QLabel("Select Model:"))
        control_layout.addWidget(self.model_combo)
        control_layout.addWidget(refresh_btn)
        control_layout.addWidget(self.load_btn)
        control_layout.addWidget(self.gpu_check)
        control_layout.addWidget(self.quant_check)
        main_layout.addLayout(control_layout)

        # Chat history
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

        # Input area
        input_layout = QHBoxLayout()
        self.user_input = QLineEdit()
        self.user_input.setPlaceholderText("Enter your question...")
        self.user_input.returnPressed.connect(self.send_message)
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.user_input, 4)
        input_layout.addWidget(send_btn, 1)
        main_layout.addLayout(input_layout)

        self.setLayout(main_layout)

    def scan_local_models(self):
        self.model_combo.clear()
        model_dir = Path(self.config.model_path)
        
        if not model_dir.exists():
            QMessageBox.warning(self, "Warning", "Model directory does not exist! Attempting to create automatically.")
            try:
                model_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to create directory: {str(e)}")
            return

        # Supported file extensions
        model_exts = ('.safetensors', '.bin', '.pth', '.pt', '.gguf')
        valid_models = []

        # Scan directory
        for entry in model_dir.iterdir():
            # Handle GGUF single files
            if entry.is_file() and entry.suffix.lower() == '.gguf':
                valid_models.append(entry.stem)  # Remove extension
            # Handle model directories
            elif entry.is_dir():
                has_config = (entry / 'config.json').exists()
                has_model_file = any(f.suffix in model_exts for f in entry.iterdir())
                if has_config or has_model_file:
                    valid_models.append(entry.name)

        # Deduplicate and sort
        valid_models = sorted(list(set(valid_models)))
        
        if not valid_models:
            QMessageBox.information(self, "Info", "No available models found")
            self.load_btn.setEnabled(False)
            return

        self.model_combo.addItems(valid_models)
        self.load_btn.setEnabled(True)

    def load_model(self):
        model_name = self.model_combo.currentText()
        if not model_name:
            return

        # Get virtual environment Python path
        venv_python = VirtualEnvManager.get_python_path()

        # Get full virtual environment variables
        process_env = VirtualEnvManager.get_env_with_venv()
        
        # Build model path
        base_path = Path(self.config.model_path)
        model_path = base_path / model_name
        if not model_path.exists():
            model_path = model_path.with_suffix('.gguf')

        if not model_path.exists():
            QMessageBox.critical(self, "Error", f"Model file not found: {model_path}")
            return

        # Prepare arguments
        args = [
            "LLM.py",
            "--model_path", str(model_path)
        ]
    
        if self.gpu_check.isChecked():
            args.append("--use_gpu")
        if self.quant_check.isChecked():
            args.append("--load_in_4bit")

        # Start process
        self.model_process = QProcess()
        self.model_process.setProcessEnvironment(process_env)  # Set environment variables
        self.model_process.start(str(venv_python), args)
        self.model_process.readyReadStandardOutput.connect(self.handle_initial_output)
        
        # Connect signals
        self.model_process.readyReadStandardOutput.connect(self.handle_output)
        self.model_process.readyReadStandardError.connect(self.handle_error)
        self.model_process.finished.connect(self.handle_finish)
        
        self.model_process.start()
        self.append_message(f"[System] Starting model loading: {' '.join(args)}", is_system=True)

    def handle_initial_output(self):
        data = self.model_process.readAllStandardOutput().data().decode()
        if "MODEL_READY" in data:  # Check new initialization flag
            self.append_message("[System] Model ready, you can start chatting", is_system=True)
            self.model_process.readyReadStandardOutput.disconnect(self.handle_initial_output)

    def send_message(self):
        if not self.model_process or self.model_process.state() != QProcess.Running:
            QMessageBox.warning(self, "Error", "Please load model first")
            return

        question = self.user_input.text().strip()
        if not question:
            return

        # Send question to model process
        self.model_process.write(f"{question}\n".encode())
        self.append_message(f"User: {question}")
        self.user_input.clear()

    def handle_output(self):
        data = self.model_process.readAllStandardOutput().data().decode(errors='ignore')
        if data:
            self.output_buffer += data
            if not self.output_timer.isActive():
                self.output_timer.start(50)  # 50ms refresh interval

    def handle_error(self):
        err = self.model_process.readAllStandardError().data().decode()
        if err.strip():
            self.append_message(f"[System] {err.strip()}", is_system=True)

    def handle_finish(self, exit_code, exit_status):
        status = "Normal" if exit_code == 0 else f"Error ({exit_code})"
        self.append_message(f"[System] Model process exited [{status}]", is_system=True)

    def append_message(self, message, is_system=False):
        """Add message to chat history"""
        cursor = self.chat_history.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        if is_system:  # System message special handling
            cursor.insertHtml(f'<span style="color:#666;">{message}</span><br>')
        else:
            # User message handling
            if message.startswith(user_prefix):
                clean_msg = message[len(user_prefix):].lstrip()  # Remove possible duplicate prefix
                cursor.insertText(f"{user_prefix}{clean_msg}\n")
            # Model message handling
            elif message.startswith(model_prefix):
                clean_msg = message[len(model_prefix):].lstrip()
                cursor.insertText(f"{model_prefix}{clean_msg}\n")
            else:  # Default handling for unknown types
                cursor.insertText(f"{message}\n")
        
        # Auto-scroll
        self.chat_history.ensureCursorVisible()

    def flush_buffer(self):
        if self.output_buffer:
            # Merge characters to current message
            self.current_message += self.output_buffer
            self.output_buffer = ""
        
            # Directly insert text (no character-by-character line breaks)
            cursor = self.chat_history.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.select(QTextCursor.BlockUnderCursor)
        
            # Replace existing content
            cursor.removeSelectedText()
            cursor.insertText(f"Firefly: {self.current_message}")
        
            # Auto-scroll
            self.chat_history.ensureCursorVisible()

# ====================== Virtual Environment Manager ======================
class VirtualEnvManager:
    @staticmethod
    def get_python_path() -> Path:
        """Get virtual environment Python path (compatible with packaged mode)"""
        # Get executable directory
        if getattr(sys, 'frozen', False):
            # In packaged mode, sys.executable is the path to the exe file
            base_path = Path(sys.executable).parent.resolve()  # Get exe directory
            venv_path = base_path / "venv"
            
            # If not found in same directory, try parent directory
            if not venv_path.exists():
                venv_path = base_path.parent / "venv"
        else:
            # Project root in development mode
            base_path = Path(__file__).parent.resolve()
            venv_path = base_path / "venv"

        # Check virtual environment directory exists
        if not venv_path.exists():
            error_msg = (
                f"Virtual environment directory not found:\n{venv_path}\n"
                "Please place venv directory in the same directory as executable"
            )
            QMessageBox.critical(
                None,
                "Virtual Environment Error",
                error_msg
            )
            sys.exit(1)

        # Determine Python interpreter path
        if sys.platform.startswith("win"):
            venv_python = venv_path / "Scripts" / "python.exe"
        else:
            venv_python = venv_path / "bin" / "python"

        # Verify Python interpreter exists
        if not venv_python.exists():
            error_msg = (
                f"Python interpreter not found:\n{venv_python}\n"
                "Please ensure virtual environment is properly installed"
            )
            QMessageBox.critical(
                None,
                "Virtual Environment Error",
                error_msg
            )
            sys.exit(1)
            
        return venv_python
    
    @staticmethod
    def get_env_with_venv():
        """Get environment variables including virtual environment path"""
        venv_python = VirtualEnvManager.get_python_path()
        venv_bin = venv_python.parent
        
        env = QProcessEnvironment.systemEnvironment()
        original_path = env.value("PATH", "")

        # Only modify PATH environment variable
        if sys.platform.startswith("win"):
            new_path = f"{str(venv_bin)};{original_path}"
        else:
            new_path = f"{str(venv_bin)}:{original_path}"
        
        env.insert("PATH", new_path)
        
        # Remove manually set Python variables
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

# ====================== File List Item Component ======================
class FileListItem(QWidget):
    def __init__(self, filename, size):
        super().__init__()
        self._filename = filename
        self._size = int(size) if str(size).isdigit() else 0  # Enforce type conversion
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
        # Add unit conversion error handling
        try:
            if size >= 1024**3:
                return f"{size/1024**3:.1f} GB"
            elif size >= 1024**2:
                return f"{size/1024**2:.1f} MB"
            elif size >= 1024:
                return f"{size/1024:.1f} KB"
            return f"{size} B"
        except:
            return "Unknown size"

    @property
    def filename(self):
        return self._filename

    @property
    def size(self):
        return self._size

# ====================== Data Model Class ======================
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

# ====================== Thread Classes ======================
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

# ====================== Download Manager Class ======================
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
                self.status_updated.emit("Download completed")
                self.finished.emit()
                self.config.download_history.append({
                    "model_id": self.model_id,
                    "file": self.file_info["rfilename"],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                self.config.save_settings()

        except Exception as e:
            self.status_updated.emit(f"Download failed: {str(e)}")
            if save_path and os.path.exists(save_path):
                try:
                    os.remove(save_path)
                except Exception as delete_error:
                    print(f"Error cleaning failed file: {delete_error}")
        finally:
            if self.response:
                self.response.close()

    def pause(self):
        self._is_paused = True
        self.status_updated.emit("Paused")

    def resume(self):
        self._is_paused = False
        self.status_updated.emit("Resuming download...")

    def cancel(self):
        self._is_canceled = True
        self.status_updated.emit("Cancelled")
        
        if self.response:
            try:
                self.response.close()
            except Exception as e:
                print(f"Error closing connection: {e}")
        
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
                    print(f"Final deletion failed: {save_path}")
            except Exception as e:
                print(f"Unexpected error deleting file: {e}")
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

# ====================== Terminal Emulator Component ======================
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
        
        new_btn = QPushButton("New Terminal")
        new_btn.clicked.connect(self.new_terminal)
        control_layout.addWidget(QLabel("Terminal Type:"))
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

# ====================== Settings Page ======================
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

        storage_group = QGroupBox("Storage Settings")
        storage_layout = QFormLayout()
        
        self.path_edit = QLineEdit()
        self.path_edit.setText(self.config.model_path)
        path_btn = QPushButton("Browse...")
        path_btn.clicked.connect(self.select_path)
        
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.path_edit, stretch=4)
        path_layout.addWidget(path_btn, stretch=1)
        storage_layout.addRow("Model Storage Path:", path_layout)

        self.storage_info = QLabel()
        self.update_storage_info()
        storage_layout.addRow("Storage Space:", self.storage_info)
        
        storage_group.setLayout(storage_layout)
        main_layout.addWidget(storage_group)

        network_group = QGroupBox("Network Settings")
        network_layout = QFormLayout()

        self.http_proxy_input = QLineEdit()
        self.http_proxy_input.setPlaceholderText("http://proxy.example.com:8080")
        self.http_proxy_input.setText(self.config.proxy["http"])
        
        self.https_proxy_input = QLineEdit()
        self.https_proxy_input.setPlaceholderText("https://proxy.example.com:8080")
        self.https_proxy_input.setText(self.config.proxy["https"])

        test_proxy_btn = QPushButton("Test Proxy")
        test_proxy_btn.clicked.connect(self.test_proxy)
        
        network_layout.addRow("HTTP Proxy:", self.http_proxy_input)
        network_layout.addRow("HTTPS Proxy:", self.https_proxy_input)
        network_layout.addRow("", test_proxy_btn)
        
        network_group.setLayout(network_layout)
        main_layout.addWidget(network_group)

        appearance_group = QGroupBox("Appearance Settings")
        appearance_layout = QVBoxLayout()

        self.theme_combo = QComboBox()
        self.theme_combo.addItem("🌞 Light Mode", "light")
        self.theme_combo.addItem("🌙 Dark Mode", "dark")
        self.theme_combo.setCurrentIndex(1 if self.config.dark_mode else 0)
        
        self.font_size = QSpinBox()
        self.font_size.setRange(10, 24)
        self.font_size.setValue(self.config.font_size)
        
        appearance_form = QFormLayout()
        appearance_form.addRow("UI Theme:", self.theme_combo)
        appearance_form.addRow("Font Size:", self.font_size)
        
        appearance_group.setLayout(appearance_form)
        main_layout.addWidget(appearance_group)

        btn_layout = QHBoxLayout()
        save_btn = QPushButton("💾 Save Settings")
        save_btn.clicked.connect(self.save_settings)
        reset_btn = QPushButton("🔄 Reset Defaults")
        reset_btn.clicked.connect(self.reset_settings)
        
        btn_layout.addWidget(reset_btn)
        btn_layout.addWidget(save_btn)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    def select_path(self):
        path = QFileDialog.getExistingDirectory(
            self,
            "Select Storage Directory",
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
                f"Used {used:.1f}GB / Total {total:.1f}GB "
                f"({usage.used/usage.total:.0%})"
            )
        except Exception as e:
            self.storage_info.setText("Unable to get storage info")

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
                    "Proxy Test Successful",
                    f"Connection successful! Response time: {latency:.0f}ms\n"
                    f"Service status: {response.json().get('status')}"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Proxy Test Failed",
                    f"Server returned error: {response.status_code}"
                )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Proxy Test Failed",
                f"Failed to connect to server:\n{str(e)}"
            )

    def save_settings(self):
        if not os.access(self.path_edit.text(), os.W_OK):
            QMessageBox.critical(self, "Error", "Storage path not writable!")
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
        QMessageBox.information(self, "Success", "Settings saved")

    def reset_settings(self):
        default_path = os.path.abspath("models")
        self.path_edit.setText(default_path)
        self.theme_combo.setCurrentIndex(0)
        self.font_size.setValue(12)
        self.http_proxy_input.clear()
        self.https_proxy_input.clear()
        self.update_storage_info()

# ====================== Model List Item ======================
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

# ====================== Model Detail Dialog ======================
class ModelDetailDialog(QDialog):
    def __init__(self, model_id, config, download_manager):
        super().__init__()
        self.model_id = model_id
        self.config = config
        self.download_manager = download_manager
        self.model_data = None
        self.setWindowTitle("Model Details")
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
        
        self.download_btn = QPushButton("Git Clone Full Model")
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
        self.tab_widget.addTab(self.doc_view, "Documentation")
        
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.itemDoubleClicked.connect(self.on_file_double_clicked)
        self.tab_widget.addTab(self.file_list, "Files")
        
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
        self.stats_label.setText(f"❤️ {model.likes} Downloads: {model.downloads} Size: {model.formatted_size}")
        
        # Dynamically generate styled Markdown content
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
            # Enhanced file size handling
            file_size = file.get('size', 0)
            try:
                file_size = int(file_size)
            except (ValueError, TypeError):
                file_size = 0
                
            item = QListWidgetItem()
            widget = FileListItem(file['rfilename'], file_size)  # Pass processed size
            item.setSizeHint(widget.sizeHint())
            self.file_list.addItem(item)
            self.file_list.setItemWidget(item, widget)

    def show_error(self, error):
        QMessageBox.critical(self, "Error", f"Failed to load details: {error}")

    def git_download(self):
        model_url = f"https://huggingface.co/{self.model_id}"
        save_path = os.path.join(self.config.model_path, self.model_id.split('/')[-1])
    
        if os.path.exists(save_path):
            QMessageBox.warning(self, "Warning", "Model directory already exists!")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Git Download Progress")
        layout = QVBoxLayout()
        output = QPlainTextEdit()
        output.setReadOnly(True)
        output.setFont(QFont("Consolas", 10))
        btn = QPushButton("Cancel")
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
                output.appendPlainText("\n✅ Download completed!")
                self.config.download_history.append({
                    "model_id": self.model_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                self.config.save_settings()
            else:
                output.appendPlainText(f"\n❌ Error: {process.readAllStandardError().data().decode()}")
    
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
                        QMessageBox.information(self, "Info", "File already in download queue or exists.")
                    return
            QMessageBox.warning(self, "Error", "File information missing.")

# ====================== Download Management Page ======================
class DownloadPage(QWidget):
    def __init__(self, download_manager):
        super().__init__()
        self.download_manager = download_manager
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        top_bar = QHBoxLayout()
        self.open_path_btn = QPushButton("Open Model Path")
        self.open_path_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.open_path_btn.clicked.connect(self.open_model_path)
        top_bar.addWidget(self.open_path_btn)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        self.task_list = QListWidget()
        layout.addWidget(QLabel("Active Downloads:"))
        layout.addWidget(self.task_list)

        btn_layout = QHBoxLayout()
        self.pause_btn = QPushButton("Pause")
        self.resume_btn = QPushButton("Resume")
        self.cancel_btn = QPushButton("Cancel")
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
                f"{speed} " if "MB" in speed else f"{speed} "  # Maintain alignment
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
            QMessageBox.critical(self, "Error", f"Path not found: {model_path}")
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

# ====================== Category Navigation Bar ======================
class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=0, spacing=5):
        super().__init__(parent)
        self.setContentsMargins(margin, margin, margin, margin)
        self._spacing = spacing
        self._items = []

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margin, _, _, _ = self.getContentsMargins()
        size += QSize(2 * margin, 2 * margin)
        return size

    def _do_layout(self, rect):
        x = rect.x()
        y = rect.y()
        line_height = 0
        spacing = self._spacing
        
        for item in self._items:
            widget = item.widget()
            space_x = spacing + widget.style().layoutSpacing(
                QSizePolicy.PushButton, QSizePolicy.PushButton, Qt.Horizontal
            )
            space_y = spacing + widget.style().layoutSpacing(
                QSizePolicy.PushButton, QSizePolicy.PushButton, Qt.Vertical
            )
            next_x = x + item.sizeHint().width() + space_x
            if next_x - space_x > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + space_y
                next_x = x + item.sizeHint().width() + space_x
                line_height = 0
            
            item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))
            
            x = next_x
            line_height = max(line_height, item.sizeHint().height())

class ModelCategoryBar(QWidget):
    category_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = FlowLayout(spacing=8)
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        
        categories = [
            ("All", "all"),
            ("Popular Models", "hot"),
            ("Text Generation", "text-generation"),
            ("Text-to-Image", "text-to-image"),
            ("Text-to-Audio", "text-to-audio"),
            ("Image Classification", "image-classification"),
            ("Speech Recognition", "speech-recognition"),
            ("Object Detection", "object-detection")
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
                    white-space: nowrap;
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

    def on_category_click(self):
        sender = self.sender()
        self.category_changed.emit(sender.property('category'))

    def sizeHint(self):
        return QSize(160, 80) 

# ====================== Model Hub Page ======================
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
        self.search_bar.setPlaceholderText("Search model names...")
        self.search_bar.returnPressed.connect(self.search_models)
        self.search_btn = QPushButton("Search")
        self.search_btn.clicked.connect(self.search_models)
        
        search_layout.addWidget(self.search_bar, stretch=4)
        search_layout.addWidget(self.search_btn, stretch=1)
        layout.addLayout(search_layout)
        
        self.model_list = QListWidget()
        self.model_list.itemDoubleClicked.connect(self.show_detail)
        layout.addWidget(self.model_list)
        
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        self.apply_styles()

    def load_hot_models(self):
        self.status_label.setText("Loading popular models...")
        self.thread = ModelSearchThread("", self.config, sort_by="downloads")
        self.thread.search_complete.connect(self.handle_hot_models)
        self.thread.search_failed.connect(self.handle_search_error)
        self.thread.start()

    def handle_hot_models(self, models):
        self.current_models = models
        self.filter_models("hot")
        self.status_label.setText(f"Found {len(models)} popular models")

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

        self.status_label.setText("Searching...")
        self.search_btn.setEnabled(False)

        self.thread = ModelSearchThread(query, self.config)
        self.thread.search_complete.connect(self.handle_search_result)
        self.thread.search_failed.connect(self.handle_search_error)
        self.thread.start()

    def handle_search_result(self, models):
        self.current_models = models
        self.filter_models(self.current_category)
        self.status_label.setText(f"Found {len(models)} models")
        self.search_btn.setEnabled(True)

    def handle_search_error(self, error):
        QMessageBox.critical(self, "Error", f"Search failed: {error}")
        self.status_label.setText("Search failed")
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

# ====================== Main Window ======================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        model_path = ConfigManager.instance().model_path
        if not os.path.exists(model_path):
            try:
                os.makedirs(model_path, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Cannot create model directory: {str(e)}")
        VirtualEnvManager.get_python_path()
        self.config = ConfigManager.instance()
        self.download_manager = DownloadManager(self.config)
        self.init_ui()
        self.setWindowTitle("RIL")
        self.setGeometry(100, 100, 1000, 600)  # Adjust initial window size
        self.apply_theme()
        self.settings_page.config_updated.connect(self.on_config_updated)
        self.setWindowIcon(QIcon("./assets/icons/placeholder.png"))
        self.oldPos = None
        self.draggable = True
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setWindowIcon(QIcon(APP_ICON_PATH))

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Custom title bar
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

        # Main content area
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.nav_list = QListWidget()
        self.nav_list.setFixedWidth(200)
        self.nav_list.setFocusPolicy(Qt.NoFocus)

        self.stacked_widget = QStackedWidget()

        self.home_page = HomePage(self.config)
        self.model_page = ModelDownloadPage(self.config, self.download_manager)
        self.settings_page = SettingsPage(self.config)
        self.download_page = DownloadPage(self.download_manager)
        self.command_line_page = CommandLinePage(self.config)
        self.chat_page = ChatPage(self.config)  # New chat page
        
        self.add_module("🏠 Home", self.home_page)
        self.add_module("💬 Model Chat", self.chat_page)  # New chat option
        self.add_module("🏗️ Model Hub", self.model_page)
        self.add_module("⚙️ Settings", self.settings_page)
        self.add_module("⬇️ Downloads", self.download_page)
        self.add_module("💻 Terminal", self.command_line_page)
        
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
        
        # Add window edge resize controls
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
        theme = "Dark" if self.config.dark_mode else "Light"
        proxy = "Enabled" if self.config.proxy["http"] else "Disabled"
        self.theme_status.setText(f"Theme: {theme}")
        self.proxy_status.setText(f"Proxy: {proxy}")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
    
        # Use QRectF instead of QRect
        rect = QRectF(0, 0, self.width(), self.height())  # No need for explicit float conversion
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
        self.chat_page.scan_local_models()  # Refresh local model list after config update

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
        pass  # No need to draw content, used for interaction only

# ====================== Program Entry ======================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(APP_ICON_PATH)) 
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())