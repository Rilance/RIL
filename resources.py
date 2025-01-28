# resources.py
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QDir

def init_resources():
    QDir.addSearchPath('icons', 'assets/icons')  # 图标存放目录