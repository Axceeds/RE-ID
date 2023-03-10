from sequence_process import SequenceProcess as sp
import file_process as fp
import numpy as np
import matrix_process as mp
from PyQt5.QtWidgets import QFileDialog, QWidget, QMessageBox
import os
import matplotlib.pyplot as plt
from scipy.stats import norm
from figureCon import AnalysisCon


class DataProcess(QWidget):
    """########################################
    数据处理
    功能：
        1、依照不同方法和参数处理数据
    备注：
        1、注意，一旦涉及到时序处理，不要使用后退键，否则前一幅图像将会输入到
        时序处理对象中，出现不可描述的现象。如果非要使用后退键，请设计相关的程序。
    ########################################"""

    def __init__(self):
        super().__init__()
        self.processNum = fp.load_para('processPara.xml', 'processNum')
        self.sp = []  # 序列处理对象（根据processNum设置多个）
        for i in range(self.processNum):
            self.sp.append(sp())
        self.data = {}  # 原始数据
        self.length = 0  # 数据长度
        self.paras = []  # 参数
        self.ground_truth = {}  # 真值
        self.myflag = 0
        self.myfigure = []

    def clear(self):
        self.data = {}
        self.ground_truth = {}
        self.length = 0
        for i in range(self.processNum):
            self.sp[i].clear()

    def set_data(self, data, length):
        # 载入数据
        self.clear()
        self.data = data
        self.length = length

    def set_para(self, paras):
        # 设置参数
        self.paras = paras

    def match_files(self):
        # 匹配背景数据、先验图和真值
        file_path = self.paras[0]['filePath']
        dir_path = file_path.split('/')
        prefix_file_name = dir_path[-1].split('.')[0]
        dir_path = dir_path[:-1]
        dir_path = '/'.join(dir_path)
        files = os.listdir(dir_path)
        background = None
        if prefix_file_name + '_background.xml' in files:
            print('背景模型已设置')
            background, _ = fp.xml_process(dir_path + '/' + prefix_file_name + '_background.xml')
            for key in background.keys():
                for i in range(self.processNum):
                    if key == self.paras[i]['dataName']:
                        self.sp[i].set_background(background[key])
        if prefix_file_name + '_prior.xml' in files:
            print('先验热图已设置')
            prior, _ = fp.xml_process(dir_path + '/' + prefix_file_name + '_prior.xml')
            for key in prior.keys():
                for i in range(self.processNum):
                    if key == self.paras[i]['dataName']:
                        self.sp[i].set_prior(prior[key])
        if prefix_file_name + '_ground_truth.xml' in files:
            ground_truth, _ = fp.xml_process(dir_path + '/' + prefix_file_name + '_ground_truth.xml')
            for key in ground_truth.keys():
                for i in range(self.processNum):
                    if key == self.paras[i]['dataName']:
                        self.ground_truth[key] = ground_truth[key]
        return background

    def get_processed_figure(self, step, workMode, data={}):
        # 处理数据
        if len(data):  # 获取原始数据
            original_data = data
        else:
            original_data = self.data

        if step == 0:  # 如果step为0，说明重新开始了，需要清除缓存
            for process in self.sp:
                process.clear()
            self.match_files()  # 重新匹配背景、先验热图和真值
        processed_data = []  # 处理后的数据，包含四幅红外热图
        for i in range(self.processNum):  # 有多幅图像，多套参数
            para = self.paras[i]  # 获取参数
            if workMode == "offline":
                name = para['dataName']  # 数据名
            elif workMode == 'online':
                name = para['portName']  # 端口视角名
            if name not in original_data.keys():  # 如果没有这个数据，则跳过
                figure = np.zeros((24, 32))
            else:
                if not len(data):  # 如果是离线数据，则需要截取此帧数据
                    figure = original_data[name][step]  # 取出第step个数据
                else:  # 如果是在线数据，则不需要截取
                    figure = original_data[name]

                for process in para['flow']:  # 处理图像
                    if process == 'None':
                        pass
                    elif process == 'quantify':
                        figure = mp.quantify(figure, _min=0, resolution=0.2)
                        if self.sp[i].background_is_set:
                            if self.sp[i].back.max() < 100:  # 说明背景还未量化
                                self.sp[i].back = mp.quantify(self.sp[i].back, _min=0, resolution=0.2)
                    elif process == 'temporalFilter':
                        figure = self.sp[i].temporal_filter(figure, para['temporalFilter'],
                                                            para['temporalFilterBuffer'])
                    elif process == 'backgroundRemove':
                        figure = self.sp[i].remove_background(figure, para['backgroundRemove'], para['RBbuffer'])
                    elif process == 'interpolate':
                        figure = mp.interpolate(figure, para['interPoints'])
                    elif process == 'spatialFilter':
                        figure = mp.spatial_filter(figure, para['filter'], para['filterPara'])
                    elif process == 'segment':
                        figure = mp.segment(figure, para['segment'], 3)
                    elif process == 'opticalFlow':
                        figure = self.sp[i].optical_flow(figure)
                    elif process == 'diff':
                        figure = self.sp[i].diff(figure)
                    elif process == 'resize':
                        figure = mp.interpolate(figure, new_size=(32, 24))
                    elif process == 'connectedComponent':
                        figure = mp.max_connected_component(figure)
                        # self.myfigure.append(figure)
                        # if len(self.myfigure)==47*20:
                        #     np.save(file="D:/2.PycharmProjects/红外数据记录/2022.03.09第一次数据采集与实验/20组_全转化为正向_已经背景移除过2.npy",
                        #             arr=self.myfigure)
                        # print(np.array(self.myfigure).shape)
                    else:
                        print('不存在这种数据处理方式，不进行任何处理')
            processed_data.append(figure)
        processed_data = np.array(processed_data, dtype=object)
        return processed_data

    def get_processed_data(self):
        # 获取处理后的数据
        # 仅限离线数据
        process_data = []
        for step in range(self.length):
            process_data.append(self.get_processed_figure(step, 'offline'))
        process_data = np.array(process_data)
        return process_data

    def save_prior_map(self, startFrame=-1, endFrame=-1, leftPixel=0, rightPixel=31, upPixel=0, downPixel=23):
        # 生成并保存先验热图,仅限离线模式
        original_data = self.data
        final_data = {}  # 用于存放处理后的结果
        if endFrame > self.length:
            QMessageBox.warning(self, "警告", "结束帧必须小于数据长度"+str(self.length))
        elif startFrame >= endFrame:
            QMessageBox.warning(self, "警告", "结束帧必须大于起始帧")
        elif upPixel > downPixel:
            QMessageBox.warning(self, "警告", "上约束必须小于下约束")
        elif leftPixel > rightPixel:
            QMessageBox.warning(self, "警告", "左约束必须小于右约束")
        else:
            for name in original_data.keys():
                matrix = original_data[name][startFrame:endFrame]  # 截取数据
                prior = np.zeros_like(matrix[0])
                for i in range(matrix.shape[0]):
                    figure = matrix[i]
                    figure = mp.quantify(figure, _min=0, resolution=0.2)
                    figure = mp.segment(figure, "triangle2", 0)
                    for k in range(upPixel, downPixel + 1):
                        for l in range(leftPixel, rightPixel + 1):
                            if figure[k][l] > 0:
                                prior[k][l] += 1
                            else:
                                pass
                final_data[name] = prior
            path = QFileDialog.getSaveFileName(self, "Save Prior Map", os.getcwd() + '/data/', "xml Files(*.xml)")
            fp.save_data(final_data, path[0])

    def save_ground_truth(self, startFrame=-1, endFrame=-1, leftPixel=0, rightPixel=31, upPixel=0, downPixel=23):
        # 此函数用于生成背景移除真值
        if endFrame > self.length:
            QMessageBox.warning(self, "警告", "结束帧必须小于数据长度" + str(self.length))
        elif startFrame != endFrame:
            QMessageBox.warning(self, "警告", "起始帧必须与结束帧相同")
        elif upPixel > downPixel:
            QMessageBox.warning(self, "警告", "上约束必须小于下约束")
        elif leftPixel > rightPixel:
            QMessageBox.warning(self, "警告", "左约束必须小于右约束")
        else:
            original_data = self.data
            final_data = {}  # 用于存放处理后的结果
            for name in original_data.keys():  # 针对两个视角
                matrix = original_data[name][startFrame]  # 截取数据
                mask = np.zeros_like(matrix)
                figure = matrix
                figure = mp.quantify(figure, _min=0, resolution=0.2)
                figure = mp.segment(figure, "triangle", 0)
                for k in range(upPixel, downPixel + 1):
                    for l in range(leftPixel, rightPixel + 1):
                        mask[k][l] += 1
                figure = figure * mask
                final_data[name] = figure
            path = QFileDialog.getSaveFileName(self, "Save Ground Truth", os.getcwd() + '/data/', "xml Files(*.xml)")
            fp.save_data(final_data, path[0])

    def get_histogram(self, startFrame=-1, endFrame=-1, leftPixel=0, rightPixel=0, upPixel=0, downPixel=0):
        pixel_list = []
        if endFrame > self.length:
                QMessageBox.warning(self, "警告", "结束帧必须小于数据长度"+str(self.length))
        elif leftPixel != rightPixel:
            QMessageBox.warning(self, "警告", "左约束与右约束必须相同")
        elif upPixel != downPixel:
            QMessageBox.warning(self, "警告", "上约束与下约束必须相同")
        elif upPixel > downPixel:
            QMessageBox.warning(self, "警告", "上约束必须小于下约束")
        elif leftPixel > rightPixel:
            QMessageBox.warning(self, "警告", "左约束必须小于右约束")
        elif startFrame >= endFrame:
            QMessageBox.warning(self, "警告", "结束帧必须大于起始帧")
        else:
            original_data = self.data
            for name in original_data.keys():
                if name in ['side','side0']:  # 只处理侧视数据的
                    for i in range(startFrame, endFrame):
                        # 把指定位置的像素点的值按帧的顺序放入列表中
                        temp = original_data[name][i][upPixel][leftPixel]
                        pixel_list.append(temp)
                    pixel_list.sort()
        return pixel_list

    def get_temperature_curve(self, startFrame=-1, endFrame=-1, leftPixel=0, rightPixel=31, upPixel=0, downPixel=23):
        average_list = []
        if endFrame > self.length:
                QMessageBox.warning(self, "警告", "结束帧必须小于数据长度"+str(self.length))
        elif startFrame >= endFrame:
            QMessageBox.warning(self, "警告", "结束帧必须大于起始帧")
        elif upPixel > downPixel:
            QMessageBox.warning(self, "警告", "上约束必须小于下约束")
        elif leftPixel > rightPixel:
            QMessageBox.warning(self, "警告", "左约束必须小于右约束")
        else:
            for i in range(self.processNum):
                temp_list = []
                for step in range(startFrame, endFrame):
                    temp = 0
                    data_list = self.get_processed_figure(step, 'offline')
                    for j in range(leftPixel, rightPixel + 1):
                        for k in range(upPixel, downPixel + 1):
                            temp += data_list[i][j][k]
                    temp = temp / ((rightPixel - leftPixel + 1) * (downPixel - upPixel + 1))
                    temp_list.append(temp)
                average_list.append(temp_list)
        return average_list

    def cut_data(self, startFrame, endFrame):
        if endFrame > self.length:
            QMessageBox.warning(self, "警告", "结束帧必须小于数据长度"+str(self.length))
        elif startFrame >= endFrame:
            QMessageBox.warning(self, "警告", "结束帧必须大于起始帧")
        else:
            original_data = self.data
            temp = {}
            for key in original_data.keys():
                temp[key] = original_data[key][startFrame:endFrame]
            path = QFileDialog.getSaveFileName(self, "Save Cut Data", os.getcwd() + '/data/', "xml Files(*.xml)")
            fp.save_data(temp, path[0])

    def combine_data(self):
        try:
            file_path_list = QFileDialog.getOpenFileNames(self, "Open File", os.getcwd() + "/data/",
                                                          "Xml Files(*.xml)")
            combined_data = {}
            data, datalen = fp.xml_process(file_path_list[0][0])
            for name in data.keys():
                combined_data[name] =[]
            for path in file_path_list[0]:
                data, datalen = fp.xml_process(path)
                for name in data.keys():
                    for i in range(datalen):
                        combined_data[name].append(data[name][i])
            path = QFileDialog.getSaveFileName(self, "Save Combined Data", os.getcwd() + '/data/', "xml Files(*.xml)")
            fp.save_data(combined_data, path[0])
        except:
            QMessageBox.warning(self, "警告", "请检查两个文件的格式，格式相同才可以合并")

    def get_indictor(self, figure):
        if self.ground_truth == {}:
            print('未载入真值')
        else:
            if not np.all(figure == 0):
                if np.all(self.ground_truth[0] == 0):
                    figure = ~(figure > 0)
                    ground_truth = ~(self.ground_truth[0] > 0)
                else:
                    figure = figure > 0
                    ground_truth = self.ground_truth[0] > 0
                TP = 0
                TN = 0
                FP = 0
                FN = 0
                for i in range(figure.shape[0]):
                    for j in range(figure.shape[1]):
                        if ground_truth[i][j] == True:
                            if figure[i][j] == True:
                                TP += 1
                            else:
                                FN += 1
                        elif ground_truth[i][j] == False:
                            if figure[i][j] == True:
                                FP += 1
                            else:
                                TN += 1
                accuracy = float((TP + TN) / (TP + TN + FP + FN))
                precision = float(TP / (TP + FP))
                recall = float(TP / (TP + FN))
                F1 = float((2 * TP) / (2 * TP + FP + FN))
                # indicator = {'TP':TP,'TN':TN,'FP':FP,'FN':FN,'accuracy':accuracy,'Precision':precision,'Recall':recall,'F-Measure':F1}
                indicator = {'accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F-Measure': F1}
                print(indicator)

