from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QSlider, QStatusBar
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt


class Information(QWidget):
    # 信息显示类显示信息显示进度条
    def __init__(self):
        super().__init__()
        self.stepLabel = QLabel(self)
        self.stepLabel.setText("第  帧")
        self.stepLabel.setFont(QFont("Microsoft Yahei", 12, QFont.Normal))
        self.stepLabel.setAlignment(Qt.AlignLeft)

        self.statusLabel = QLabel(self)
        self.statusLabel.setText("状态栏")
        self.statusLabel.setFont(QFont("Microsoft Yahei",12,  QFont.Normal))

        # 滑动条，显示播放进度
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 10)
        self.slider.setTickPosition(QSlider.TicksBelow)  # 进度条下方显示刻度
        self.slider.setTickInterval(1)  # 进度条刻度间距为1
        self.slider.setEnabled(False)  # 设置为未启用

        self.label_zone = QHBoxLayout()
        self.label_zone.addWidget(self.stepLabel)
        self.label_zone.addStretch()
        self.label_zone.addWidget(self.statusLabel)
        self.label_zone.addStretch()
        # label_zone.addWidget(self.behaviorLabel)

        self.main_zone = QVBoxLayout()
        self.main_zone.addLayout(self.label_zone)
        self.main_zone.addWidget(self.slider)
        self.setLayout(self.main_zone)

    def init_slider(self, length):
        # 初始化进度条并设置长度
        self.slider.setRange(1, length)
        self.slider.setValue(1)

    def set_slider(self, position):
        # 设置进度条位置
        self.slider.setValue(position)

    def set_step(self, message=None):
        # 设置步数显示
        if message is not None:
            try:
                self.stepLabel.setText(str(message))
            except:
                pass

    def set_status(self, message=None):
        # 设置状态栏内容及显示时间
        if message is not None:
            try:
                self.statusLabel.setText(str(message))
            except:
                pass