import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import file_process as fp
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QFileDialog, QToolButton, QVBoxLayout, QCheckBox, QLabel, QLineEdit, \
    QGridLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

matplotlib.use("Qt5Agg")
from PyQt5.QtCore import Qt, QSize
from matplotlib import patches
from scipy.stats import norm
from button import Button
from ports import Ports


class FigureCon(QWidget):
    # 图像显示控制类FigureCon，指挥图像显示类Figure工作
    def __init__(self, num):
        super().__init__()  # QWidget初始化
        self.start_flag = True
        self.num = num
        figure_para = fp.load_para('figurePara.xml')
        figureHeight = figure_para['figureHeight']  # 图像高度
        figureWidth = figure_para['figureWidth']  # 图像宽度
        row = figure_para['row']
        col = figure_para['col']
        color = figure_para['colorStyle']

        self.temperatureShow = QCheckBox('显示温度')
        self.temperatureShow.setChecked(False)
        self.adaptiveColor = QCheckBox('固定色调')
        self.adaptiveColor.setChecked(False)

        self.start_side = QToolButton()
        Button.button_init(self.start_side, "开始", "./images/Icons/play-circle.png")
        self.pause_side = QToolButton()
        Button.button_init(self.pause_side, "停止", "./images/Icons/stop.png")

        self.settingBt = QToolButton()
        Button.button_init(self.settingBt, "设置参数", "./images/Icons/setting.png")

        self.screenshotBt = QToolButton()
        Button.button_init(self.screenshotBt, "保存截图", "./images/Icons/camera.png")

        self.information = QLabel('处理流程')
        self.information.setFont(QFont("Microsoft Yahei", 10, QFont.Normal))

        self.set_information()

        # 布局
        self.HLayout = QGridLayout()
        self.HLayout.addWidget(self.start_side, 0, 0)
        self.HLayout.addWidget(self.pause_side, 0, 1, Qt.AlignLeft)
        self.HLayout.addWidget(self.settingBt, 0, 2, Qt.AlignRight)
        self.HLayout.addWidget(self.screenshotBt, 0, 3)

        self.figure = Figure(row, col, color)
        self.figure.setFixedSize(figureWidth, figureHeight)
        self.VLayout = QVBoxLayout()
        self.VLayout.addWidget(self.figure)
        self.VLayout.addLayout(self.HLayout)

        self.mainLayout = QHBoxLayout()
        self.mainLayout.addLayout(self.VLayout)
        self.mainLayout.addWidget(self.information)
        self.setLayout(self.mainLayout)

        # 创建信号连接
        self.temperatureShow.stateChanged.connect(self.temperature)
        self.adaptiveColor.stateChanged.connect(self.adaptive_color)
        self.screenshotBt.clicked.connect(self.save_screenshot)

    def temperature(self):
        if self.temperatureShow.isChecked():
            self.figure.temperatureBool = True
        else:
            self.figure.temperatureBool = False

    def adaptive_color(self):
        if self.adaptiveColor.isChecked():
            self.figure.adaptiveColor = True
        else:
            self.figure.adaptiveColor = False

    def save_screenshot(self):
        # 保存图像
        path = QFileDialog.getSaveFileName(self, "Save Png File", os.getcwd() + "/images/save/", "Png Files(*.png)")
        self.figure.save(path[0])

    def set_information(self):
        process_para = fp.load_para('processPara.xml')
        flow = (process_para['flow' + str(self.num)])
        information = 'input\n'
        for value in flow:
            if value == 'None':
                pass
            elif value == 'backgroundRemove':
                background_remove_method = process_para['backgroundRemove']
                information += background_remove_method[self.num] + '\n'
            elif value == 'segment':
                segment_method = process_para['segment']
                information += segment_method[self.num] + '\n'
            elif value == 'temporalFilter':
                temporal_filter_method = process_para['temporalFilter']
                information += temporal_filter_method[self.num] + '\n'
            elif value == 'spatialFilter':
                spatial_filter_method = process_para['filter']
                information += spatial_filter_method[self.num] + '\n'
            else:
                information += value + '\n'
        information += 'output'
        self.information.setText(str(information))


class Figure(FigureCanvas):
    # 图像显示类
    def __init__(self, row=24, col=32, color='jet'):
        self.row = row
        self.col = col
        self.colorStyle = color
        self.temperatureBool = False  # 温度数值显示标志位
        self.adaptiveColor = False  # 自适应色调标志位
        self.fig = plt.figure()  # 新建图片
        self.axes = self.fig.add_subplot(1, 1, 1)  # 在图片中新建subplot
        FigureCanvas.__init__(self, self.fig)  # 激活Figure窗口
        X = 10 * np.random.rand(self.row, self.col)  # 初始化时给出一副随机的图像
        # self.fig.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        self.fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0, wspace=0)
        self.axes.imshow(X, interpolation='nearest', cmap=self.colorStyle)

    def update_figure(self, figure_data):
        # 得到图像的最大值和非零最小值，用于固定色调
        minimum = 0
        maximum = figure_data.max()
        if figure_data.max() != 0:
            minimum = figure_data[figure_data.nonzero()].min()

        # 更新图像，得到动画效果
        if self.adaptiveColor:
            im = self.axes.imshow(figure_data, interpolation='nearest', cmap=self.colorStyle, vmin=minimum,
                                  vmax=maximum)
        else:
            im = self.axes.imshow(figure_data, interpolation='nearest', cmap=self.colorStyle)

        self.axes.draw_artist(im)
        self.blit(self.axes.bbox)

        if self.temperatureBool:
            row = figure_data.shape[0]
            col = figure_data.shape[1]
            for i in range(row):
                for j in range(col):  # 保留整数位
                    value = str(int(figure_data[i, j]))
                    self.axes.text(j, i, value, color='white', va='center', ha='center', size=7)
            self.draw()

    def save(self, path):
        # 保存图像
        self.fig.savefig(fname=path, dpi=600)


class AnalysisCon(QWidget):
    def __init__(self):
        super().__init__()  # QWidget初始化
        self.title = QLineEdit()
        self.x_axis = QLineEdit()
        self.y_axis = QLineEdit()
        self.title_label = QLabel('标题')
        self.x_axis_label = QLabel('X轴')
        self.y_axis_label = QLabel('y轴')

        self.check_bt = QToolButton()
        self.check_bt.setText('确定')
        self.save_bt = QToolButton()
        self.save_bt.setText('保存')

        self.analysis = Analysis()

        self.controlLayout = QHBoxLayout()
        self.controlLayout.addWidget(self.title_label)
        self.controlLayout.addWidget(self.title)
        self.controlLayout.addWidget(self.x_axis_label)
        self.controlLayout.addWidget(self.x_axis)
        self.controlLayout.addWidget(self.y_axis_label)
        self.controlLayout.addWidget(self.y_axis)
        self.controlLayout.addWidget(self.check_bt)
        self.controlLayout.addWidget(self.save_bt)

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addLayout(self.controlLayout)
        self.mainLayout.addWidget(self.analysis)
        self.setLayout(self.mainLayout)

        self.save_bt.clicked.connect(self.analysis.save)
        self.check_bt.clicked.connect(
            lambda: self.analysis.set_information(self.title.text(), self.x_axis.text(), self.y_axis.text()))


class Analysis(FigureCanvas):
    # 数据分析图类
    def __init__(self):
        self.fig = plt.figure()  # 新建图片
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
        self.axes = self.fig.add_subplot(1, 1, 1)  # 在图片中新建subplot
        FigureCanvas.__init__(self, self.fig)  # 激活Figure窗口

    def show_histogram(self, pixel_list):
        try:
            self.axes.cla()
            weights = np.ones_like(pixel_list) / len(pixel_list)
            params = self.axes.hist(pixel_list, bins=20, weights=weights, facecolor='blue', edgecolor='black')
            self.axes.set_aspect('auto')  # x轴和y轴的比例设为自适应
            bins = params[1]
            y = norm.pdf(bins, np.mean(pixel_list), np.std(pixel_list))
            scale = bins[1] - bins[0]
            self.axes.plot(bins, y * scale, 'r--', linewidth=5)
            self.draw()
        except:
            pass

    def show_curve(self, average_list):
        try:
            self.axes.cla()
            for i in range(len(average_list)):
                self.axes.plot(average_list[i])
            self.axes.legend(['0', '1', '2', '3'])
            self.draw()
        except:
            pass

    def save(self):
        path = QFileDialog.getSaveFileName(self, "Save Png File", os.getcwd() + "/images/save/", "Png Files(*.png)")
        self.fig.savefig(fname=path[0], dpi=600)

    def set_information(self, title, x_axis, y_axis):
        self.axes.set_title(title)
        self.axes.set_xlabel(x_axis)
        self.axes.set_ylabel(y_axis)
        self.draw()
