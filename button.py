import os
from PyQt5.QtWidgets import QWidget, QToolButton, QHBoxLayout, QLabel, QSpinBox, QVBoxLayout
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, QSize
import file_process as fp


class Button(QWidget):
    """########################################
    按钮
    功能：
        1、创建多个按钮
        2、按钮互锁，防止误操作
    ########################################"""

    def __init__(self, work_mode):
        super().__init__()

        # 公共按钮
        self.changeModeBt = QToolButton()  # 模式切换按钮
        self.button_init(self.changeModeBt, "切换模式", "./images/Icons/control.png")
        self.modeLabel = QLabel()
        self.modeLabel.setText(work_mode)
        self.machineLearningBt = QToolButton()  # 载入模型按钮
        self.button_init(self.machineLearningBt, "载入模型", "./images/Icons/AI.png")
        self.exitBt = QToolButton()  # 退出按钮
        self.button_init(self.exitBt, "退出", "./images/Icons/poweroff.png")

        # 离线模式按钮
        self.openFileBt = QToolButton()  # 打开文件按钮
        self.button_init(self.openFileBt, "打开文件", "./images/Icons/folder-open.png")
        self.startPauseBt = QToolButton()  # 开始和暂停按钮
        self.button_init(self.startPauseBt, "开始", "./images/Icons/play-circle.png")
        self.stopBt = QToolButton()  # 停止按钮
        self.button_init(self.stopBt, "停止", "./images/Icons/stop.png")
        self.backwardBt = QToolButton()  # 上一帧按钮
        self.button_init(self.backwardBt, "上一帧", "./images/Icons/left-circle.png")
        self.forwardBt = QToolButton()  # 下一帧按钮
        self.button_init(self.forwardBt, "下一帧", "./images/Icons/right-circle.png")
        self.dataBaseBt = QToolButton()  # 数据库按钮
        self.button_init(self.dataBaseBt, "数据库", "./images/Icons/database.png")
        self.combineDataBt = QToolButton()  # 合并数据按钮
        self.button_init(self.combineDataBt, "合并数据", "./images/Icons/file-copy.png")
        self.cutBt = QToolButton()  # 截取数据按钮
        self.button_init(self.cutBt, "截取数据", "./images/Icons/cut.png")
        self.savePriorBt = QToolButton()  # 保存先验热图按钮
        self.button_init(self.savePriorBt, "保存先验热图", "./images/Icons/screenshot.png")
        self.saveGroundTruthBt = QToolButton()  # 保存真值按钮
        self.button_init(self.saveGroundTruthBt, "保存真值", "./images/Icons/image.png")
        self.plotHistBt = QToolButton()  # 画直方图的按钮
        self.button_init(self.plotHistBt, "像素直方图", "./images/Icons/barchart.png")
        self.plotCurveBt = QToolButton()  # 画曲线的按钮
        self.button_init(self.plotCurveBt, "平均温度曲线", "./images/Icons/linechart.png")


        self.selectStartFrameLabel = QLabel('起始帧')
        self.selectStartFrame = QSpinBox()
        self.selectStartFrame.setMaximum(20000)
        self.selectStartFrame.setValue(fp.load_para('processPara.xml', 'startFrame'))

        self.selectEndFrameLabel = QLabel('结束帧')
        self.selectEndFrame = QSpinBox()
        self.selectEndFrame.setMaximum(20000)
        self.selectEndFrame.setValue(fp.load_para('processPara.xml', 'endFrame'))
        self.selectLeftPixelLabel = QLabel('左')
        self.selectLeftPixel = QSpinBox()
        self.selectLeftPixel.setMaximum(31)
        self.selectLeftPixel.setValue(fp.load_para('processPara.xml', 'leftPixel'))
        self.selectRightPixelLabel = QLabel('右')
        self.selectRightPixel = QSpinBox()
        self.selectRightPixel.setMaximum(31)
        self.selectRightPixel.setValue(fp.load_para('processPara.xml', 'rightPixel'))
        self.selectUpPixelLabel = QLabel('上')
        self.selectUpPixel = QSpinBox()
        self.selectUpPixel.setMaximum(23)
        self.selectUpPixel.setValue(fp.load_para('processPara.xml', 'upPixel'))
        self.selectDownPixelLabel = QLabel('下')
        self.selectDownPixel = QSpinBox()
        self.selectDownPixel.setMaximum(23)
        self.selectDownPixel.setValue(fp.load_para('processPara.xml', 'downPixel'))

        # 在线模式按钮
        self.openport = QToolButton()  # 打开端口接收数据
        self.button_init(self.openport, "打开端口", "./images/Icons/play-circle.png")
        self.closeport = QToolButton()  # 停止接收数据
        self.button_init(self.closeport, "关闭端口", "./images/Icons/timeout.png")
        self.save = QToolButton()  # 接收并保存指定帧数的数据
        self.button_init(self.save, "数据采集", "./images/Icons/save.png")

        self.selectFramesLabel = QLabel('采集帧数')
        self.selectFrames = QSpinBox()
        self.selectFrames.setMaximum(20000)
        self.selectFrames.setValue(fp.load_para('processPara.xml', 'frames'))
        # self.loadBackgroundBt = QToolButton()  # 载入背景按钮
        # self.button_init(self.loadBackgroundBt, "载入背景", "/images/Icons/screenshot.png")

        self.offlineBts = []  # 离线按钮列表
        self.offlineBts.append(self.openFileBt)
        self.offlineBts.append(self.startPauseBt)
        self.offlineBts.append(self.stopBt)
        self.offlineBts.append(self.backwardBt)
        self.offlineBts.append(self.forwardBt)
        self.offlineBts.append(self.cutBt)
        self.offlineBts.append(self.combineDataBt)
        self.offlineBts.append(self.savePriorBt)
        self.offlineBts.append(self.saveGroundTruthBt)
        self.offlineBts.append(self.plotHistBt)
        self.offlineBts.append(self.plotCurveBt)
        self.offlineBts.append(self.dataBaseBt)
        self.offlineBts.append(self.selectStartFrame)
        self.offlineBts.append(self.selectStartFrameLabel)
        self.offlineBts.append(self.selectEndFrame)
        self.offlineBts.append(self.selectEndFrameLabel)
        self.offlineBts.append(self.selectLeftPixel)
        self.offlineBts.append(self.selectLeftPixelLabel)
        self.offlineBts.append(self.selectRightPixel)
        self.offlineBts.append(self.selectRightPixelLabel)
        self.offlineBts.append(self.selectUpPixel)
        self.offlineBts.append(self.selectUpPixelLabel)
        self.offlineBts.append(self.selectDownPixel)
        self.offlineBts.append(self.selectDownPixelLabel)

        self.fileBts = []  # 文件操作按钮列表
        self.fileBts.append(self.startPauseBt)
        self.fileBts.append(self.stopBt)
        self.fileBts.append(self.backwardBt)
        self.fileBts.append(self.forwardBt)
        self.fileBts.append(self.cutBt)
        self.fileBts.append(self.savePriorBt)
        self.fileBts.append(self.saveGroundTruthBt)
        self.fileBts.append(self.plotHistBt)
        self.fileBts.append(self.plotCurveBt)
        self.fileBts.append(self.selectStartFrame)
        self.fileBts.append(self.selectStartFrameLabel)
        self.fileBts.append(self.selectEndFrame)
        self.fileBts.append(self.selectEndFrameLabel)
        self.fileBts.append(self.selectLeftPixel)
        self.fileBts.append(self.selectLeftPixelLabel)
        self.fileBts.append(self.selectRightPixel)
        self.fileBts.append(self.selectRightPixelLabel)
        self.fileBts.append(self.selectUpPixel)
        self.fileBts.append(self.selectUpPixelLabel)
        self.fileBts.append(self.selectDownPixel)
        self.fileBts.append(self.selectDownPixelLabel)

        self.onlineBts = []  # 在线按钮列表
        self.onlineBts.append(self.openport)
        self.onlineBts.append(self.closeport)
        self.onlineBts.append(self.save)
        self.onlineBts.append(self.selectFrames)
        self.onlineBts.append(self.selectFramesLabel)
        # self.onlineBts.append(self.loadBackgroundBt)

        # 按钮布局
        self.startFrameLayout = QHBoxLayout()
        self.startFrameLayout.addWidget(self.selectStartFrameLabel)
        self.startFrameLayout.addWidget(self.selectStartFrame)
        self.endFrameLayout = QHBoxLayout()
        self.endFrameLayout.addWidget(self.selectEndFrameLabel)
        self.endFrameLayout.addWidget(self.selectEndFrame)
        self.frameLayout = QVBoxLayout()
        self.frameLayout.addLayout(self.startFrameLayout)
        self.frameLayout.addLayout(self.endFrameLayout)
        self.leftPixelLayout = QHBoxLayout()
        self.leftPixelLayout.addWidget(self.selectLeftPixelLabel)
        self.leftPixelLayout.addWidget(self.selectLeftPixel)
        self.rightPixelLayout = QHBoxLayout()
        self.rightPixelLayout.addWidget(self.selectRightPixelLabel)
        self.rightPixelLayout.addWidget(self.selectRightPixel)
        self.upPixelLayout = QHBoxLayout()
        self.upPixelLayout.addWidget(self.selectUpPixelLabel)
        self.upPixelLayout.addWidget(self.selectUpPixel)
        self.downPixelLayout = QHBoxLayout()
        self.downPixelLayout.addWidget(self.selectDownPixelLabel)
        self.downPixelLayout.addWidget(self.selectDownPixel)
        self.pixelLayout = QVBoxLayout()
        self.pixelLayout.addLayout(self.leftPixelLayout)
        self.pixelLayout.addLayout(self.rightPixelLayout)
        self.pixel2Layout = QVBoxLayout()
        self.pixel2Layout.addLayout(self.upPixelLayout)
        self.pixel2Layout.addLayout(self.downPixelLayout)

        self.buttonLayout = QHBoxLayout()
        self.buttonLayout.addWidget(self.changeModeBt)
        self.buttonLayout.addWidget(self.modeLabel)
        self.buttonLayout.addWidget(self.openFileBt)
        self.buttonLayout.addWidget(self.startPauseBt)
        self.buttonLayout.addWidget(self.stopBt)
        self.buttonLayout.addWidget(self.backwardBt)
        self.buttonLayout.addWidget(self.forwardBt)
        self.buttonLayout.addLayout(self.frameLayout)
        self.buttonLayout.addLayout(self.pixelLayout)
        self.buttonLayout.addLayout(self.pixel2Layout)
        self.buttonLayout.addWidget(self.cutBt)
        self.buttonLayout.addWidget(self.combineDataBt)
        self.buttonLayout.addWidget(self.savePriorBt)
        self.buttonLayout.addWidget(self.saveGroundTruthBt)
        self.buttonLayout.addWidget(self.plotHistBt)
        self.buttonLayout.addWidget(self.plotCurveBt)
        self.buttonLayout.addWidget(self.dataBaseBt)
        self.buttonLayout.addWidget(self.openport)
        self.buttonLayout.addWidget(self.closeport)
        self.buttonLayout.addWidget(self.save)
        self.buttonLayout.addWidget(self.selectFramesLabel)
        self.buttonLayout.addWidget(self.selectFrames)
        self.buttonLayout.addWidget(self.machineLearningBt)
        # self.buttonLayout.addWidget(self.loadBackgroundBt)
        self.buttonLayout.addWidget(self.exitBt)
        self.buttonLayout.addStretch()

        self.setLayout(self.buttonLayout)
        self.mode = work_mode  # 初始模式
        self.change_mode()
        self.change_mode()
        self.startPause = 1  # 初始时开始/暂停按键为开始

        # 按钮事件连接
        self.changeModeBt.clicked.connect(self.change_mode)
        self.openport.clicked.connect(self.open_port_clicked)
        self.closeport.clicked.connect(self.close_port_clicked)
        self.save.clicked.connect(self.save_clicked)

        self.selectStartFrame.valueChanged.connect(self.startFrameParaSave)
        self.selectEndFrame.valueChanged.connect(self.endFrameParaSave)
        self.selectLeftPixel.valueChanged.connect(self.leftPixelParaSave)
        self.selectRightPixel.valueChanged.connect(self.rightPixelParaSave)
        self.selectUpPixel.valueChanged.connect(self.upPixelParaSave)
        self.selectDownPixel.valueChanged.connect(self.downPixelParaSave)
        self.selectFrames.valueChanged.connect(self.framesParaSave)

    @staticmethod
    def button_init(button, button_name, image_path):
        # 按钮初始化，设置文字、字体、Icon和按钮样式
        button.setText(button_name)
        button.setFont(QFont("Microsoft Yahei", 8, QFont.Normal))
        button.setIcon(QIcon(os.getcwd() + image_path))
        button.setIconSize(QSize(25, 25))
        button.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        button.setStyleSheet("QToolButton{border-radius:5px;\
                                     border:1px ;}"
                             "QToolButton:hover{background-color:#80FFFF;}")

    """按钮互锁"""
    def change_mode(self):
        # 离线模式和实时检测模式的切换
        if self.mode == 'offline':
            self.mode = 'online'
            self.modeLabel.setText("通信模式")
            self.hide_offline_bt()
            self.show_online_bt()
        elif self.mode == 'online':
            self.mode = 'offline'
            self.modeLabel.setText("离线模式")
            self.show_offline_bt()
            self.hide_online_bt()

    def change_start_pause(self, state):
        # 开始暂停按钮图标切换，state等于1时变为开始，state等于0时变为暂停
        if state:
            self.button_init(self.startPauseBt, "开始", "./images/Icons/play-circle.png")
        else:
            self.button_init(self.startPauseBt, "暂停", "./images/Icons/timeout.png")
        self.startPause = state

    def hide_offline_bt(self):
        # 隐藏离线模式按钮
        for bt in self.offlineBts:
            bt.hide()
            bt.setEnabled(False)

    def show_offline_bt(self):
        # 显示离线模式按钮（不包括play按钮）
        for bt in self.offlineBts:
            bt.show()
            bt.setEnabled(True)
        for bt in self.fileBts:
            bt.hide()
            bt.setEnabled(False)

    def hide_online_bt(self):
        # 隐藏在线模式按钮
        for bt in self.onlineBts:
            bt.hide()
            bt.setEnabled(False)

    def show_online_bt(self):
        # 显示在线模式按钮
        for bt in self.onlineBts:
            bt.show()
            bt.setEnabled(True)

    def hide_play_bt(self):
        # 隐藏播放相关按钮
        for bt in self.fileBts:
            bt.hide()
            bt.setEnabled(False)

    def show_play_bt(self):
        # 显示播放相关按钮
        for bt in self.fileBts:
            bt.show()
            bt.setEnabled(True)

    def open_port_clicked(self):
        # 打开端口按钮按下，打开串口按钮、转换工作模式按钮、截取按钮关闭
        self.openport.setEnabled(False)
        self.changeModeBt.setEnabled(False)
        self.save.setEnabled(False)

    def close_port_clicked(self):
        # 关闭串口按钮按下，打开串口按钮、转换工作模式按钮、截取按钮激活
        self.openport.setEnabled(True)
        self.changeModeBt.setEnabled(True)
        self.save.setEnabled(True)

    def save_clicked(self):
        # 截取按钮按下，转换工作模式按钮、打开串口按钮关闭
        self.changeModeBt.setEnabled(False)
        self.openport.setEnabled(False)
        self.save.setEnabled(False)

    def startFrameParaSave(self):
        fp.save_para('processPara.xml', 'startFrame', self.selectStartFrame.value())
    def endFrameParaSave(self):
        fp.save_para('processPara.xml', 'endFrame', self.selectEndFrame.value())
    def leftPixelParaSave(self):
        fp.save_para('processPara.xml', 'leftPixel', self.selectLeftPixel.value())
    def rightPixelParaSave(self):
        fp.save_para('processPara.xml', 'rightPixel', self.selectRightPixel.value())
    def upPixelParaSave(self):
        fp.save_para('processPara.xml', 'upPixel', self.selectUpPixel.value())
    def downPixelParaSave(self):
        fp.save_para('processPara.xml', 'downPixel', self.selectDownPixel.value())
    def framesParaSave(self):
        fp.save_para('processPara.xml', 'frames', self.selectFrames.value())