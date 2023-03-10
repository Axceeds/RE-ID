"""
@File    : main.py
@Contact : 18373786_buaa@buaa.edu.cn

@Created Time        @Author         @Version    @Description
------------        ------------    --------    ------------
2022/02/28          shihaoxiang      1.0           None
"""
'''系统库'''
import os
import sys
from functools import partial
import numpy as np

'''PyQt库'''
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QApplication, QLabel, QDesktopWidget, \
    QFileDialog
from PyQt5.QtGui import QFont, QIcon

"""自己的库"""
from ports import Ports
from button import Button
from video_start_pause import Video
from dataenv_information import EnvironmentInformation
from figureCon import FigureCon, AnalysisCon
from process_setting import ProcessSetting
from database import Database
from information import Information
from result import Result
from data_process import DataProcess
from mymodel import Mymodel
import file_process as fp


class Main(QWidget):
    pause_then_clear = pyqtSignal()
    open_port_signal = pyqtSignal()
    line_edit = pyqtSignal(dict)

    def __init__(self):
        super(Main, self).__init__()
        self.start_flag = 1
        self.new_start_flag = 0
        self.open_port_flag = False
        self.class1 = []
        self.class2 = []
        self.myfigure = {}
        self.buttonList = []
        self.ports_flag = []
        self.address = []
        self.dict = {'0': '师', '1': '闵', '2': '王', '3': '高', '4': '杨', '5': 'wy', '999': 'unknown'}

        # 实例化对象
        self.button = Button('offline')  # 操作按钮
        self.information = Information()  # 帧数进度图
        self.data_process = DataProcess()  # 数据处理对象
        self.ports = Ports()  # 端口对象
        self.dataBase = Database()  # 数据库界面对象
        self.environment = EnvironmentInformation()
        self.result = Result()
        self.video = Video()
        self.model = Mymodel()

        # 载入xml的初始化系统参数
        self.process_Para = fp.load_para('processPara.xml')
        self.port_name = self.process_Para['portName']
        for key in self.port_name:
            self.myfigure[key] = []

        self.publicPara = fp.load_para('publicPara.xml')  # 载入public参数
        self.timeInterval = self.publicPara['timeInterval']  # 载入计时器间隔
        self.workMode = self.publicPara['workMode']  # 载入工作模式
        self.figureNum = self.publicPara['figureNum']  # 载入图像数量

        self.figures = []
        self.processSetting = []
        for i in range(self.figureNum):  # 创建多个红外热图显示和处理流程控制对象
            self.figures.append(FigureCon(i))
            self.processSetting.append(ProcessSetting(i))

        self.analysisCon = AnalysisCon()  # 数据分析二级界面
        self.timer = QTimer()  # 计时器

        # 相关对象和相关参数的设定
        self.set_para()  # 数据处理对象参数初始化
        self.timer.setInterval(self.timeInterval)  # 设定计时器间隔
        self.length = 0  # 设定离线数据长度
        self.step = -1  # 图像目前显示的帧，-1表示还没有开始

        # 主界面整体布局
        self.setWindowIcon(QIcon('./images/Icons/main-window.png'))
        self.setWindowTitle('基于TPAS的跨域行人重识别系统')

        self.mainLayout = QVBoxLayout()
        self.imageLayout = QHBoxLayout()
        self.buttonLayout = QHBoxLayout()

        self.imageLayout.addWidget(self.figures[0])
        self.imageLayout.addWidget(self.figures[1])
        self.imageLayout.addWidget(self.figures[2])
        self.figures[2].pause_side.hide()
        self.figures[2].start_side.hide()
        self.imageLayout.addWidget(self.figures[3])
        self.figures[3].hide()
        self.figures[2].hide()
        # self.result.file_select.hide()

        self.mainLayout.addLayout(self.imageLayout)
        self.mainLayout.addWidget(self.result)
        self.mainLayout.addWidget(self.button)
        self.mainLayout.addWidget(self.information)
        self.mainLayout.addWidget(self.environment)
        self.setLayout(self.mainLayout)

        self.screen = QDesktopWidget().screenGeometry()
        # self.setFixedSize(self.screen.width() * 4 / 5, self.screen.height() * 7 / 10)
        # self.move((self.screen.width() - self.width()) // 2,
        #           (self.screen.height() - self.height()) // 2)
        self.move(300, 130)
        self.show()

        # 建立按钮、计时器等相关信号连接
        self.build_button_signal_connection()

    def build_button_signal_connection(self):
        # 建立控制按钮等的信号连接
        self.button.changeModeBt.clicked.connect(self.change_mode)
        self.button.exitBt.clicked.connect(self.exit)
        self.button.openFileBt.clicked.connect(self.open_file_bt_clicked)

        self.button.startPauseBt.clicked.connect(self.start_pause_bt_clicked)
        self.button.stopBt.clicked.connect(self.stop_bt_clicked)

        self.button.backwardBt.clicked.connect(self.backward_bt_clicked)
        self.button.forwardBt.clicked.connect(self.forward_bt_clicked)
        self.button.cutBt.clicked.connect(self.cut_bt_clicked)

        self.button.combineDataBt.clicked.connect(self.combine_data_bt_clicked)

        self.button.savePriorBt.clicked.connect(self.save_prior_bt_clicked)
        self.button.saveGroundTruthBt.clicked.connect(self.save_ground_truth_bt_clicked)
        self.button.plotHistBt.clicked.connect(self.plot_hist_bt_clicked)
        self.button.plotCurveBt.clicked.connect(self.plot_curve_bt_clicked)

        self.button.dataBaseBt.clicked.connect(self.database_bt_clicked)
        # 这里开始连接port
        self.button.openport.clicked.connect(self.open_port)
        self.button.openport.clicked.connect(self.open_port_flagset)
        self.button.closeport.clicked.connect(self.close_port)
        self.button.closeport.clicked.connect(self.close_port_flagset)

        self.button.save.clicked.connect(self.save)
        self.figures[0].start_side.clicked.connect(self.save)
        self.figures[1].start_side.clicked.connect(self.save)
        self.figures[0].start_side.clicked.connect(self.start_pause)
        self.figures[1].start_side.clicked.connect(self.start_pause)
        self.figures[0].pause_side.clicked.connect(self.close_port)
        self.figures[1].pause_side.clicked.connect(self.close_port)
        self.figures[0].pause_side.clicked.connect(self.start_pause_1)
        self.figures[1].pause_side.clicked.connect(self.start_pause_1)
        self.result.result_analyse.clicked.connect(self.start_analyse)
        self.result.file_select.clicked.connect(self.pre_process)

        self.pause_then_clear.connect(self.clear_myfigure)
        self.line_edit.connect(self.environment.set_text)

        self.ports.dataReady.connect(self.update_figure)  # 串口数据准备好的信号，将更新图像
        self.timer.timeout.connect(self.forward_bt_clicked)  # 计时器事件，相当于按下了下一帧按钮
        self.ports.portClosed.connect(self.button.close_port_clicked)  # 端口停止数据传输信号，改变按钮
        # self.ports.portClosed.connect(self.start_pause)  # 端口停止数据传输信号，改变按钮
        self.ports.buttonClosed.connect(self.start_pause_1)
        self.ports.dataOver.connect(self.data_over)

        self.buttonList.append(self.figures[0].start_side)
        self.buttonList.append(self.figures[0].pause_side)
        self.buttonList.append(self.figures[1].start_side)
        self.buttonList.append(self.figures[1].pause_side)
        for bt in self.buttonList:
            bt.hide()

        for i in range(self.figureNum):
            self.processSetting[i].settingChangeSignal.connect(self.set_para)  # self.processSetting的参数修改了
            self.figures[i].settingBt.clicked.connect(partial(self.set_process, i))
        self.dataBase.selectFile.connect(self.select_database_file)  # 从数据库中载入了某个文件

    def pre_process(self):
        """#######################################
        此函数主要进行原始数据的处理、得到处理后的数据
        并将其存储在all_dataprocessed中
        #######################################"""
        file_path_type = QFileDialog.getOpenFileName(self, "Open File", "./data/",
                                                     "Xml Files(*.xml)")
        file_path = file_path_type[0]
        try:
            data, data_len, info = fp.xml_process(file_path)  # 打开文件，获取数据
            self.line_edit.emit(info)
            fp.save_para('processPara.xml', 'filePath', file_path)
            fp.save_para('processPara.xml', 'length', data_len)
            self.data_process.paras[0]['filePath'] = file_path  # 在这里必须更新文件路径，否则将载入上一次的背景模型
            self.button.show_play_bt()
            if data_len > 0:
                self.information.set_status('载入红外数据文件：' + file_path.split('/')[-1])
                self.length = data_len  # 设定self.length
                self.data_process.set_data(data, data_len)  # dataProcess载入数据
                self.information.init_slider(self.length)  # 初始化进度条
                self.step = -1  # 播放位置复位
            process_data = np.array(self.data_process.get_processed_data())
            d1 = {}
            i = 0
            for keys in self.port_name:
                d1[keys] = []
            for i in range(47):
                d1['side0'].append(process_data[i][0])
            for i in range(47):
                d1['side1'].append(process_data[i][1])
            for i in range(47):
                d1['top0'].append(process_data[i][2])
            for i in range(47):
                d1['top1'].append(process_data[i][3])
            file_path = QFileDialog.getSaveFileName(self, "Save xml File",  "./data/all_dataprocessed/",
                                                    "Xml Files(*.xml)")
            save_path = file_path[0]
            fp.save_data(d1, save_path)
            print('successfully saved processed_data:{0}'.format(save_path))
        except:
            pass

    def close_port_flagset(self):
        """"#######################################
        这个函数主要是控制”开始“的使能，按下去之后希望能
        把self.open_port_flag这个信号False，用于在update
        中判断是否留存这段数据，在线模式实时传输的时候，我们不希望
        让他把数据全都存下来
        #######################################"""
        self.open_port_flag = False

    def open_port_flagset(self):
        """#######################################
        作用同上，按下停止之后，再把使能True
        #######################################"""
        self.open_port_flag = True

    def clear_myfigure(self):
        """#######################################
        用于清空存储经过各项预处理之后的数据列表和flag，
        简化作用，不必每次再写这两句
        #######################################"""
        for key in self.port_name:
            self.myfigure[key] = []

    def start_analyse(self):
        """#######################################
        更新QLabel显示的信息，class1，class2分别是识别的
        两个类，判断是不是同一类输出即可
        #######################################"""

        class1, class2 = self.model.data_perception()
        if class1[0] == 999 or class2[0] == 999:
            self.result.result_label_1.setText('是否为同一行人：unknown')
            self.result.result_label_2.setText('行人1为：' + self.dict[str(class1[0])])
            self.result.result_label_3.setText('行人2为：' + self.dict[str(class2[0])])
        else:
            if class1 == class2:
                self.result.result_label_1.setText('是否为同一行人：是')
                self.result.result_label_2.setText('行人1为：' + self.dict[str(class1[0])])
                self.result.result_label_3.setText('行人2为：' + self.dict[str(class2[0])])
            else:
                self.result.result_label_1.setText('是否为同一行人：否')
                self.result.result_label_2.setText('行人1为：' + self.dict[str(class1[0])])
                self.result.result_label_3.setText('行人2为：' + self.dict[str(class2[0])])

    def start_pause(self):
        """#######################################
        这里是按键互锁，防止连按“开始”按钮出现存储故障
        #######################################"""
        self.figures[0].start_side.setDisabled(True)
        self.figures[1].start_side.setDisabled(True)

    def start_pause_1(self):
        """#######################################
        按下”停止“之后，让”开始“重新使能，同时发出一个信号，
        执行clear_myfigure，清空缓存
        #######################################"""
        self.figures[0].start_side.setDisabled(False)
        self.figures[1].start_side.setDisabled(False)
        self.pause_then_clear.emit()

    def get_processed_data_auto(self):
        """"#######################################
        这里是定义自动保存处理后的数据
        #######################################"""
        pass

    def data_over(self, str):
        """#######################################
        保存_origin.xml的函数
        #######################################"""
        if str != '':
            temp = os.path.basename(str)
            temp = temp.split('.')[0]
            filename = os.getcwd() + r'\data\all_dataprocessed'
            self.address = filename + '/' + temp + '_processed' + '.xml'
            fp.save_data(self.myfigure, self.address)
            print('成功保存预处理过的数据：{0}'.format(self.address))
            self.clear_myfigure()

    def update_figure(self, step=0):
        """#######################################
        更新图像，更新进度条
        输入：
            1、步数step（整数）
        输出：
            1、处理后的数据processedData（tData字典）
        备注：
            1、根据工作模式，选择方式
            2、在线模式下，使用的step是由tPort返回的
            3、在线模式输入的step=1是为了不进行缓存清空
            4、用myfigue存下来处理后的数据，把他变成一个连续帧
        #######################################"""
        #  dsadsa

        if self.workMode == 'offline':
            processed_data = self.data_process.get_processed_figure(step, self.workMode)  # 从数据处理对象获取处理后的数据
            for i in range(self.figureNum - 1):
                self.figures[i].figure.update_figure(processed_data[i])  # 更新图像
                if not self.open_port_flag:
                    self.myfigure[self.port_name[i]].append(processed_data[i])
            self.information.set_slider(step + 1)  # 更新进度条
            self.information.set_step('第' + str(step) + '帧')  # 更新信息显示
        else:
            port_step, original_data = self.ports.get()  # 从串口获取原始数据
            processed_data = self.data_process.get_processed_figure(1, self.workMode, original_data)  # 获取处理后的数据
            for i in range(self.figureNum - 1):
                self.figures[i].figure.update_figure(processed_data[i])  # 更新图像
                # print(len(processed_data[i]))
                if not self.open_port_flag:
                    self.myfigure[self.port_name[i]].append(processed_data[i])
            self.information.set_slider(port_step + 1)  # 更新进度条
            self.information.set_step('第' + str(port_step) + '帧')  # 更新信息显示
        return processed_data

    '''切换模式的信号槽2 '''

    def change_mode(self):
        # 按下转换工作模式按钮，转换工作模式
        if self.workMode == 'offline':
            self.workMode = 'online'
            for bt in self.buttonList:
                bt.show()
                # bt.setEnabled(True)
            self.stop_bt_clicked()
        elif self.workMode == 'online':
            self.workMode = 'offline'
            for bt in self.buttonList:
                bt.hide()
                # bt.setEnabled(False)
            self.close_port()

    def close_port(self):
        # self.scene.clear()
        # 新加的模块，意思是让开始按钮可以不随着ports_closed改变，因为这个信号在切换模式时也会停止

        self.ports.close_ports()  # 关闭串口
        self.data_process.clear()  # dataProcess缓存清空
        # self.perception.clear()

    '''设置para'''  '''在process_init 里的paras[0]['filePath']用到'''

    def set_para(self, info='参数初始化完成'):
        # 获取参数，传递给数据处理对象和串口处理对象
        # 并且在信息显示对象上显示
        process_para = []
        for i in range(self.figureNum):
            process_para.append(self.processSetting[i].processPara)  # 获取处理流程参数
            self.figures[i].set_information()
        self.data_process.set_para(process_para)  # 设定数据处理对象参数
        self.information.set_status(info)  # 显示相关信息

    """‘退出’ 按键"""

    def exit(self):
        app = QApplication.instance()
        app.quit()

    """‘打开文件’ 按键"""

    def open_file_bt_clicked(self):
        # 打开文件，并进行相应操作
        if not self.button.startPause:  # 如果正在运行，那么就暂停
            self.start_pause_bt_clicked()
        file_path_type = QFileDialog.getOpenFileName(self, "Open File", os.getcwd() + "/data/",
                                                     "Xml Files(*.xml)")

        self.process_init(file_path_type[0])

    def process_init(self, file_path):
        # 数据的载入及初始化
        try:
            data, data_len, info = fp.xml_process(file_path)  # 打开文件，获取数据
            # print(info)
            self.line_edit.emit(info)
            fp.save_para('processPara.xml', 'filePath', file_path)
            fp.save_para('processPara.xml', 'length', data_len)
            self.data_process.paras[0]['filePath'] = file_path  # 在这里必须更新文件路径，否则将载入上一次的背景模型
            self.button.show_play_bt()
            if data_len > 0:
                self.information.set_status('载入红外数据文件：' + file_path.split('/')[-1])
                self.length = data_len  # 设定self.length
                self.data_process.set_data(data, data_len)  # dataProcess载入数据
                self.information.init_slider(self.length)  # 初始化进度条
                self.step = -1  # 播放位置复位
                # self.perception.set_offline_data(data)  # 行为感知对象载入数据
        except:
            pass

    '''开始停止 按键'''

    def start_pause_bt_clicked(self):
        # 根据开始暂停键目前的状态，进行相应操作
        if self.button.startPause:
            self.start_bt_clicked()
        else:
            self.pause_bt_clicked()

    def start_bt_clicked(self):
        # 开始按钮按下，计时器开始计时，并且开始键变为暂停键
        self.timer.start()
        self.button.change_start_pause(0)  # 转换按键为暂停

    def pause_bt_clicked(self):
        # 暂停键按下，计时器暂停，并且暂停键变为开始键
        self.timer.stop()
        self.button.change_start_pause(1)  # 转换按键为开始

    def select_database_file(self, path):
        # 从数据库中选取了某个文件
        if not self.button.startPause:  # 如果正在运行，那么就暂停
            self.start_pause_bt_clicked()
        self.process_init(path)

    '''停止  按键'''
    def stop_bt_clicked(self):
        # self.scene.clear()
        self.timer.stop()  # 计时器停止
        self.step = -1  # 播放位置复位
        self.information.set_slider(1)  # 进度条复位
        if not self.button.startPause:  # 如果按键是暂停，则将其转换为开始
            self.pause_bt_clicked()

    '''前一帧、后一帧 按键'''

    def backward_bt_clicked(self):
        # 步数减一，判断其范围是否合法，合法则更新图像
        # 如果涉及时序处理，此功能需慎用
        self.step -= 1
        if self.step < 0:
            self.step = 0
        else:
            self.update_figure(self.step)

    def forward_bt_clicked(self):
        # 步数加一，判断其范围是否合法，合法则更新图像
        self.step += 1
        if self.step >= self.length:
            self.step = self.length - 1
            self.stop_bt_clicked()  # 播放到尾部，自动按下停止键
        else:
            self.update_figure(self.step)

    '''合并数据 按键'''

    def combine_data_bt_clicked(self):
        # 合并数据
        self.data_process.combine_data()

    '''截取数据 按键'''

    def cut_bt_clicked(self):
        # 截取数据
        startFrame = fp.load_para('processPara.xml', 'startFrame')
        endFrame = fp.load_para('processPara.xml', 'endFrame')
        self.data_process.cut_data(startFrame, endFrame)

    '''绘图，四个图'''

    def save_prior_bt_clicked(self):
        # 设置先验知识图
        if self.workMode == "offline":
            startFrame = fp.load_para('processPara.xml', 'startFrame')
            endFrame = fp.load_para('processPara.xml', 'endFrame')
            leftPixel = fp.load_para('processPara.xml', 'leftPixel')
            rightPixel = fp.load_para('processPara.xml', 'rightPixel')
            upPixel = fp.load_para('processPara.xml', 'upPixel')
            downPixel = fp.load_para('processPara.xml', 'downPixel')
            self.data_process.save_prior_map(startFrame, endFrame, leftPixel, rightPixel, upPixel, downPixel)
        else:
            print("工作模式错误")

    def save_ground_truth_bt_clicked(self):
        # 设置真值
        if self.workMode == "offline":
            startFrame = fp.load_para('processPara.xml', 'startFrame')
            endFrame = fp.load_para('processPara.xml', 'endFrame')
            leftPixel = fp.load_para('processPara.xml', 'leftPixel')
            rightPixel = fp.load_para('processPara.xml', 'rightPixel')
            upPixel = fp.load_para('processPara.xml', 'upPixel')
            downPixel = fp.load_para('processPara.xml', 'downPixel')
            self.data_process.save_ground_truth(startFrame, endFrame, leftPixel, rightPixel, upPixel, downPixel)
        else:
            print("工作模式错误")

    def plot_hist_bt_clicked(self):
        # 画直方图
        self.analysisCon.show()
        startFrame = fp.load_para('processPara.xml', 'startFrame')
        endFrame = fp.load_para('processPara.xml', 'endFrame')
        leftPixel = fp.load_para('processPara.xml', 'leftPixel')
        rightPixel = fp.load_para('processPara.xml', 'rightPixel')
        upPixel = fp.load_para('processPara.xml', 'upPixel')
        downPixel = fp.load_para('processPara.xml', 'downPixel')
        pixel_list = self.data_process.get_histogram(startFrame, endFrame, leftPixel, rightPixel, upPixel, downPixel)
        self.analysisCon.analysis.show_histogram(pixel_list)

    def plot_curve_bt_clicked(self):
        # 画温度平均曲线
        self.analysisCon.show()
        startFrame = fp.load_para('processPara.xml', 'startFrame')
        endFrame = fp.load_para('processPara.xml', 'endFrame')
        leftPixel = fp.load_para('processPara.xml', 'leftPixel')
        rightPixel = fp.load_para('processPara.xml', 'rightPixel')
        upPixel = fp.load_para('processPara.xml', 'upPixel')
        downPixel = fp.load_para('processPara.xml', 'downPixel')
        temerature_curve = self.data_process.get_temperature_curve(startFrame, endFrame, leftPixel, rightPixel, upPixel,
                                                                   downPixel)
        self.analysisCon.analysis.show_curve(temerature_curve)

    '''显示数据库 按钮'''

    def database_bt_clicked(self):
        # 显示数据库界面
        self.dataBase.show()

    '''port相关'''

    def open_port(self):
        # 打开串口
        self.ports.open()

    def save(self):
        # 打开串口，保存数据
        max_frame = self.ports.cut()
        self.information.init_slider(max_frame)  # 初始化进度条

    ''''''

    def set_process(self, num):
        # 按下设置按钮，打开设置界面
        self.processSetting[num].show()
        if self.workMode == 'online':
            self.processSetting[num].selectDataName.hide()
            self.processSetting[num].selectPortName.show()
            self.processSetting[num].portLabel.show()
        elif self.workMode == 'offline':
            self.processSetting[num].selectDataName.show()
            self.processSetting[num].selectPortName.hide()
            self.processSetting[num].portLabel.hide()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mywindow = Main()
    app.exit(app.exec_())
