import os
import time

from PyQt5.QtWidgets import QWidget, QToolButton, QHBoxLayout, \
    QPushButton, QLabel, QSpinBox, QGridLayout, QProgressBar, QMessageBox, QProgressDialog
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal
from button import Button
import file_process as fp


class Result(QWidget):
    process_bar = pyqtSignal(int, int)
    init_processedBar = pyqtSignal()

    def __init__(self):
        super(Result, self).__init__()

        self.resultLayout = QHBoxLayout()
        self.resultLayout_1 = QHBoxLayout()
        # self.resultLayout = QHBoxLayout()
        self.result_analyse = QToolButton()
        Button.button_init(self.result_analyse, "开始识别", "./images/Icons/AI.png")
        self.file_select = QToolButton()
        Button.button_init(self.file_select, "数据预处理", "./images/Icons/file-copy.png")
        self.result_label_1 = QLabel()
        self.result_label_1.setText('是否为同一行人:')
        self.result_label_1.setFont(QFont("Microsoft Yahei", 18, QFont.Normal))
        self.result_label_2 = QLabel()
        self.result_label_2.setText('行人1为：')
        self.result_label_2.setFont(QFont("Microsoft Yahei", 18, QFont.Normal))
        self.result_label_3 = QLabel()
        self.result_label_3.setText('行人2为：')
        self.result_label_3.setFont(QFont("Microsoft Yahei", 18, QFont.Normal))

        self.resultLayout_1.addWidget(self.file_select)
        self.resultLayout_1.addWidget(self.result_analyse)
        # self.resultLayout.addWidget(self.file_select,0,0, Qt.AlignLeft)
        # self.resultLayout.addWidget(self.result_analyse,0,1, Qt.AlignLeft)
        self.resultLayout.addLayout(self.resultLayout_1, Qt.AlignLeft)
        self.resultLayout.addStretch(100)
        self.resultLayout.addWidget(self.result_label_1, Qt.AlignCenter)
        self.resultLayout.addWidget(self.result_label_2, Qt.AlignCenter)
        self.resultLayout.addWidget(self.result_label_3, Qt.AlignCenter)
        self.setLayout(self.resultLayout)
        self.process_bar.connect(self.processbar_show)
        self.init_processedBar.connect(self.init_processedbar)
        # self.processBar = QProgressBar()
        # self.processBar.setWindowTitle('数据分析进度')
        # self.processBar.setFixedSize(250, 30)
        # self.processBar.setRange(0, 10)
        # self.button = QPushButton()
        # self.process_bar.connect(self.processbar_show)
    def init_processedbar(self):
        self.num = 100
        self.processBar = QProgressDialog()
        self.processBar.setWindowTitle('数据分析进度')
        self.processBar.setLabelText('正在分析请稍等...')
        self.processBar.setMinimumDuration(2)
        # self.processBar.set
        self.processBar.setWindowModality(Qt.WindowModal)
        self.processBar.setRange(0, self.num)

    def processbar_show(self, count1, count2):
        """"#######################################
        这里希望设置一个进度条，但是其实是安慰提示，并不代表
        实际进度，time.sleep(0.01)是暂停10ms
        #######################################"""
        for i in range(count1, count2):
            time.sleep(0.01)
            self.processBar.setValue(i)
            if i == self.num:
                self.processBar.hide()
                QMessageBox.information(self, "提示", "识别完成")
