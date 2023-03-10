import os
from PyQt5.QtWidgets import QWidget, QToolButton, QHBoxLayout, QLabel, QSpinBox, QGridLayout
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, QSize
from button import Button
import file_process as fp


class Video(QWidget):
    def __init__(self):
        super(Video, self).__init__()
        self.videomainLayout = QGridLayout()
        self.videoLayout_1 = QHBoxLayout()
        self.videoLayout_2 = QHBoxLayout()

        self.start_side_1 = QToolButton()
        Button.button_init(self.start_side_1, "开始", "./images/Icons/play-circle.png")
        self.pause_side_1 = QToolButton()
        Button.button_init(self.pause_side_1, "停止", "./images/Icons/stop.png")

        self.start_side_2 = QToolButton()
        Button.button_init(self.start_side_2, "开始", "./images/Icons/play-circle.png")

        self.pause_side_2 = QToolButton()
        Button.button_init(self.pause_side_2, "停止", "./images/Icons/stop.png")
        self.videoLayout_1.addWidget(self.start_side_1)
        self.videoLayout_1.addWidget(self.pause_side_1)
        self.videoLayout_1.addSpacing(200)
        self.videoLayout_2.addWidget(self.start_side_2)
        self.videoLayout_2.addWidget(self.pause_side_2)
        self.videoLayout_2.addSpacing(200)

        self.videomainLayout.addLayout(self.videoLayout_1, 0, 0, 1, 3, Qt.AlignCenter)
        self.videomainLayout.addLayout(self.videoLayout_2, 0, 1, 1, 10, Qt.AlignCenter)
        self.setLayout(self.videomainLayout)
