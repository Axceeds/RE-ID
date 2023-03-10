import os
import numpy as np
import socket
import file_process as fp
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QFile, QIODevice
from PyQt5.QtWidgets import QMessageBox, QWidget, QFileDialog


class Ports(QWidget):
    """########################################
    多个端口多个传感器同时采集数据
    功能：
        1、创建、打开、关闭多个端口
        2、获取数据，发送数据
    #######################################"""
    dataReady = pyqtSignal()  # 数据准备好信号
    portClosed = pyqtSignal()  # 串口关闭信号
    buttonClosed = pyqtSignal() #
    dataOver = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.myflag = 0
        self.save_flag = 0
        # 设定计时器

        self.timer = QTimer()  # 计时器
        self.timer.setInterval(fp.load_para('publicPara.xml', 'timeInterval'))  # 设定计时器采样时间
        self.timer.timeout.connect(self.get_port_data)  # 计时器事件的信号连接

        self.ports = []  # 多个端口
        self.port_num_list = fp.load_para('processPara.xml')['port']
        self.port_count = fp.load_para('processPara.xml')['processNum']
        self.port_name_list = fp.load_para('processPara.xml')['portName']

        for i in range(self.port_count):  # 创建多个端口并输入参数
            self.ports.append(Port(self.port_num_list[i]))

        self.maxFrame = 0
        self.step = 0  # 目前的帧数
        self.data = {}  # 多帧数据
        self.oneFrame = {}

    def clear(self):
        self.maxFrame = 0
        self.step = 0
        self.data.clear()
        self.oneFrame.clear()
        self.timer.stop()

    def open(self):
        # 打开多个端口
        for port in self.ports:  # 打开多个串口
            result = port.open()
            if result == 0:
                QMessageBox.information(self, "警告", port.port_num + "打开失败", QMessageBox.Ok)
        self.timer.start()  # 计时器开始计时

    def close_ports(self):
        # 关闭多个串口
        for port in self.ports:
            port.close()  # 关闭每一个串口
        self.clear()
        self.portClosed.emit()  # 发出串口关闭信号

    def get_port_data(self):
        # 获取串口数据，根据self.maxFrame决定是否保存
        if self.maxFrame <= 0:
            # 如果maxFrame小于等于0，则一直采集数据，不保存数据
            self.get_one_frame()
        else:
            # 如果maxFrame大于0，则采集到一定长度后保存数据
            if self.step == 0:
                # 如果step等于0，则创建self.data
                for key in self.get_one_frame().keys():
                    self.data[key] = []
                self.step += 1
            elif self.step < self.maxFrame:
                # 如果step小于self.maxFrame，将采集到的数据append到self.data中
                one_frame = self.get_one_frame()
                for name in one_frame.keys():
                    self.data[name].append(one_frame[name])
                self.step += 1
            elif self.step == self.maxFrame:
                # 到达最大帧数，保存数据，关闭端口
                self.timer.stop()  # 计时器停止
                file_path = QFileDialog.getSaveFileName(self, "Save xml File", os.getcwd() + "/data/all_data/",
                                                        "Xml Files(*.xml)")
                save_path = file_path[0]
                # if self.save_flag < 2:
                #     save_path = os.getcwd() + '/data/' + str(self.save_flag) + '.xml'
                #     print('成功保存' + str(self.save_flag) + '.xml')
                #     self.save_flag += 1
                # else:
                #     self.save_flag = 0
                #     save_path = os.getcwd() + '/data/' + str(self.save_flag) + '.xml'
                #     print('成功保存' + str(self.save_flag) + '.xml')
                #     self.save_flag += 1
                fp.save_data(self.data, save_path)  # 保存数据
                if save_path != '':
                    print('成功保存原始数据：{0}'.format(save_path))
                self.dataOver.emit(save_path)
                self.buttonClosed.emit()
                self.close_ports()  # 关闭端口
        # return save_path

    def get_one_frame(self):
        # 获取单帧数据
        self.oneFrame.clear()
        for i in range(self.port_count):
            data = self.ports[i].get()
            self.oneFrame[self.port_name_list[i]] = data
        copy_data = {}  # 复制
        for name, data in self.oneFrame.items():
            copy_data[name] = data.copy()
        self.dataReady.emit()  # 发出数据准备好的信号
        self.myflag += 1 # 这里是为了记录一共多少帧数，因为一次dataReady就代表处理一帧图像
        return copy_data

    def cut(self):
        # 打开串口，采集并保存一定长度的数据
        self.maxFrame = fp.load_para('processPara.xml')['frames']
        self.open()
        return self.maxFrame

    def get(self):
        # 返回一帧串口数据
        return self.step, self.oneFrame


class Port(QThread):
    # 单端口
    def __init__(self, portNum):
        super().__init__()

        self.port_num = portNum  # 端口号
        self.usoc = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.usoc.bind(('', self.port_num))  # 监听端口号

        self.port_is_running = False  # 端口操作标志位，该标志位为假则线程会停止
        self.startFlag = False  # 找到数据的标志位
        self.endFlag = False  # 完成标志位，表示读取一帧数据完成
        self.headLow = 0  # 标志位低八位
        self.headHigh = 0  # 标志位高八位
        self.dataWord = []  # 需要计算的数据，包含低八位和高八位
        self.tMatrixNow = []  # 温度矩阵，用于外部程序的读取
        self.tMatrix = []
        self.dataPosition = 0  # 数据位置
        self.sum = 0  # 一帧数据的和，用于校准

    def open(self):
        # 打开端口
        try:
            self.start()  # 启动线程
            print('端口' + str(self.port_num) + '打开成功')
            return 1
        except:
            print('端口' + str(self.port_num) + '打开失败，请重试')
            return 0

    def get(self):
        # 返回数据
        if len(self.tMatrixNow) == 768:
            self.tMatrixNow = np.array(self.tMatrixNow)
            return self.tMatrixNow.reshape((24, 32))
        else:
            return np.zeros((24, 32))

    def run(self):
        # 线程启动后执行的程序
        print('启动串口采集线程, 端口号: ', str(self.port_num))
        self.port_is_running = True
        while True:
            buffer, _ = self.usoc.recvfrom(16000)  # 最多一次接收10000字节
            for i in range(len(buffer)):
                self.transform(buffer[i])
            if not self.port_is_running:
                # 每次都检测self.port_is_running，一旦其为假，就停止串口运行
                print('关闭串口采集线程, 串口号: ', str(self.port_num))
                break

    def transform(self, data):
        # 数据解算程序
        # 使用self.headLow和self.headHigh检测数据帧头5A5A
        self.headLow = self.headHigh
        self.headHigh = data
        if not self.startFlag:
            # 之前没有找到帧头5A5A，检测当前是否为帧头
            if self.headHigh == 90 and self.headLow == 90:
                # 当前为帧头5A5A，开始转换数据
                self.startFlag = True
                self.endFlag = False
                self.dataPosition = 2
                self.sum = 23130
                self.tMatrix = []
                self.dataWord.clear()
        else:
            # 已经找到帧头5A5A，转换当前数据
            self.dataWord.append(data)
            if len(self.dataWord) == 2:
                # self.dataWord中有两字节时进行一次转换
                result = self.dataWord[0] + self.dataWord[1] * 256  # 数据转换，高位+低位*256
                self.sum += result  # self.sum用于校验

                if 4 <= self.dataPosition < 1540:
                    # 从第4字节开始是数据位
                    if result & 0x8000:  # 如果有负的温度，需要取反
                        tmp = 0xFFFF - (0xFFFF & result)
                        tmp = -tmp / 100.0
                        self.tMatrix.append(tmp)
                    else:
                        tmp = result / 100.0
                        self.tMatrix.append(tmp)
                elif 1540 <= self.dataPosition < 1542:
                    # 1540至1542字节为热敏电阻温度
                    if result & 0x8000:  # 如果有负的温度，需要取反
                        tmp = 0xFFFF - (0xFFFF & result)
                        tmp = -tmp / 100.0
                        self.thermistor = tmp
                    else:
                        tmp = result / 100.0
                        self.thermistor = tmp
                elif self.dataPosition >= 1542:
                    # 1542至1544字节为校验位
                    check_word = result
                    self.sum -= result
                    self.sum &= 0xFFFF
                    if check_word == self.sum:
                        self.tMatrixNow = self.tMatrix
                    else:
                        # print('this frame is wrong')
                        pass

                    # 完成一帧数据的转换，清空各个变量和标志位，准备下一帧
                    self.tMatrix = []
                    self.dataPosition = 0
                    self.startFlag = False
                    self.endFlag = True

                # 完成2字节数据的转换，位置+2，self.dataWord清空
                self.dataPosition += 2
                self.dataWord.clear()

    def close(self):
        # 改变标志位，停止串口采集线程
        self.port_is_running = False
