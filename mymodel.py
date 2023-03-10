import os
import numpy as np
import matplotlib.pyplot as plt
import keras
import cv2 as cv
from tensorflow.keras.models import Sequential, Model, load_model
from sklearn.ensemble import IsolationForest

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
import file_process as fp
from result import Result
from PyQt5.QtWidgets import QFileDialog, QWidget, QMessageBox
from PyQt5.QtCore import pyqtSignal
import time
from goto import with_goto
from retrying import retry  # => 引入retry装饰器



# 绘制出图
class Mymodel(QWidget):
    def __init__(self):
        super(Mymodel, self).__init__()
        self.IM_ROW = 24
        self.IM_COL = 32
        self.ALL_FRAME = 47  # 单个视频采集总帧数
        self.COLLECT_FRAME = 8  # 背景采集的帧数
        self.VALID_FRAME = self.ALL_FRAME - self.COLLECT_FRAME  # 有效帧
        self.SAMPLE_NUM = 40  # 总的样本数
        self.CHANEL = 1  # 通道数
        self.all_data1 = []
        self.all_data2 = []
        self.result = Result()
        self.dict = {'0': '师', '1': '闵', '2': '王', '3': '高'}

    def closest(self, mylist, Number):
        answer = []
        for i in mylist:
            answer.append(abs(Number - i))
        return answer.index(min(answer))

    def centroid(self, matrix):
        frame = len(matrix)
        col = len(matrix[0])
        row = len(matrix[0][0])
        list = []
        for i in range(frame):
            count = 0
            cnt = 0
            for j in range(col):
                for k in range(row):
                    if matrix[i][j][k] > 0:
                        count = count + k
                        cnt += 1
            if cnt == 0:
                return np.array([[0]])
            else:
                list.append(int(count / cnt))
        if self.closest(list, 16) + 12 > 39 or self.closest(list, 16) - 12 < 0:
            print('此矩阵的数值分布太差，直接剔除，不进行识别')
            return np.array([[0]])
        else:
            newMatrix = matrix[self.closest(list, 16) - 12: self.closest(list, 16) + 12]
            return np.array(newMatrix)

    def plot_im_dynastic(self, data):
        minium = 0
        maxium = data.max()
        for i in range(0, 24):
            plt.clf()
            plt.imshow(data[i], interpolation='nearest', cmap='jet', vmin=minium, vmax=maxium)
            # 根据像素绘制图片 origin表示渐变程度
            # 显示像素与数据对比
            plt.xticks(())
            plt.yticks(())
            plt.pause(0.2)
            # 不显示坐标轴刻度
            plt.ioff()

    def cos_sim(self, a, b):
        """"#######################################
        数据读取、数据随机打乱、训练模型
        #######################################"""
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
        cos = np.dot(a, b) / (a_norm * b_norm)
        return cos

    def build_model(self):
        """#######################################
        搭建模型或者载入模型
        #######################################"""
        model = keras.models.load_model('model/centroid_model_24frame_4people.h5')
        return model

    @retry()
    def load_file(self):
        file_path_1 = QFileDialog.getOpenFileName(None, "Open File", os.getcwd() + "/data/all_dataprocessed/",
                                                  "Xml Files(*.xml)")
        file_path_2 = QFileDialog.getOpenFileName(None, "Open File", os.getcwd() + "/data/all_dataprocessed/",
                                                  "Xml Files(*.xml)")
        temp_data1, length, label_1_temp = fp.xml_process(file_path_1[0])
        temp_data2, length, label_2_temp = fp.xml_process(file_path_2[0])
        label_1 = np.array(label_1_temp['label'])
        label_2 = np.array(label_2_temp['label'])

        temp1 = np.array(temp_data1['side0'][8:47])
        temp2 = np.array(temp_data2['side1'][8:47])

        temp1 = self.centroid(temp1)
        temp1 = np.array(temp1)
        temp2 = self.centroid(temp2)
        temp2 = np.array(temp2)
        if np.all(temp1 == 0) or np.all(temp2 == 0):
            QMessageBox.information(self, "提示", "数据格式选择错误或数据本身不符合标准，请重新选择")
            raise
        else:
            return temp1, temp2, label_1, label_2

    def iso(self, tempdata):
        iso = IsolationForest(random_state=0, n_estimators=100)
        all_data = np.load('./data/featureVec_data.npy')
        all_data = np.reshape(all_data, newshape=(len(all_data), len(all_data[0][0])))
        iso.fit(all_data[0:105])
        return iso.predict(tempdata)


    def data_perception(self):
        # (48*3) * 24 * 32
        """"#######################################
        数据读取、数据随机打乱、训练模型
        #######################################"""
        self.result.init_processedBar.emit()
        threshold = 0.8
        temp1, temp2, label_1, label_2 = self.load_file()
        self.result.process_bar.emit(1, 30)
        model = self.build_model()
        self.result.process_bar.emit(30, 50)
        """此处为单个、全部遍历的预测"""
        # count = 0
        # true = 0
        # list_true = []
        # list_pred = []
        # matrix = np.zeros((4, 4))
        # path = os.getcwd() + '/data/all_dataprocessed/'
        # file_path = os.listdir(path)
        # for file in file_path:
        #     dir_path = file.split('_')
        #     if str(dir_path[0]) == 'Exp2':
        #         if str(dir_path[3]) == '1':
        #             count += 1
        #             new_filepath = path + file
        #             temp1, length, templabel = fp.xml_process(new_filepath)
        #             temp1 = temp1['side0']
        #             label = templabel['label']
        #             class1 = model.predict_classes(np.reshape(temp1[8:47], newshape=(1, 39, 24, 32, 1)))
        #             list_true.append(label)
        #             list_pred.append(class1[0])
        #             if str(label) == str(class1[0]):
        #                 true += 1
        #         else:
        #             count += 1
        #             new_filepath = path + file
        #             temp1, length, templabel = fp.xml_process(new_filepath)
        #             temp1 = temp1['side1']
        #             label = templabel['label']
        #             class1 = model.predict_classes(np.reshape(temp1[8:47], newshape=(1, 39, 24, 32, 1)))
        #             list_true.append(label)
        #             list_pred.append(class1[0])
        #             if str(label) == str(class1[0]):
        #                 true += 1
        #
        #         print('real_label={0}\tpred_label={1}'.format(label, str(class1[0])))
        # for i in range(len(list_pred)):
        #     matrix[int(list_true[i])][int(list_pred[i])]+=1
        # print(true)
        # print(count)
        # print(class1)
        # print(matrix)
        # print('the acc={0}'.format(true / count))

        # 打乱顺序
        '''模型train'''
        # 暂时用不到了，模型训练好就保存.h5了
        # history = model.fit(data_train, label_train, batch_size=4, epochs=20)
        # model.save('./mymodel.h5')
        # print(history.history)
        # evaluate model
        # val_loss, val_acc = model.evaluate(data_test, label_test, verbose=0)
        # print('summary: val_loss: %f, val_acc: %f' % (val_loss, val_acc))
        '''预测结果'''
        # 抽取单个层结果输出，这里输出LSTM之后的256维度特征向量
        LSTM_output = Model(inputs=model.input, outputs=model.get_layer('LSTM_1').output)
        vector_pre_1 = LSTM_output.predict(np.reshape(temp1, newshape=(1, 24, 24, 32, 1)))
        vector_pre_2 = LSTM_output.predict(np.reshape(temp2, newshape=(1, 24, 24, 32, 1)))
        vector_pre_1 = np.reshape(vector_pre_1, newshape=(1, 384))
        vector_pre_2 = np.reshape(vector_pre_2, newshape=(1, 384))
        flag_1 = self.iso(vector_pre_1)
        flag_2 = self.iso(vector_pre_2)
        if flag_1 == -1 or flag_2 == -1:
            if flag_1 == -1:
                class1 = [999]
            elif flag_1 == 1:
                class1 = model.predict_classes(np.reshape(temp1, newshape=(1, 24, 24, 32, 1)))
            if flag_2 == -1:
                class2 = [999]
            elif flag_2 == 1:
                class2 = model.predict_classes(np.reshape(temp2, newshape=(1, 24, 24, 32, 1)))
        else:
            class1 = model.predict_classes(np.reshape(temp1, newshape=(1, 24, 24, 32, 1)))
            class2 = model.predict_classes(np.reshape(temp2, newshape=(1, 24, 24, 32, 1)))
        self.result.process_bar.emit(50, 101)
        return class1, class2




        # vec_mean = np.load('./data/vec_mean.npy')
        # vec_mean_0 = vec_mean[0][0]
        # vec_mean_1 = vec_mean[1][0]
        # vec_mean_2 = vec_mean[2][0]
        # vec_mean_3 = vec_mean[3][0]
        # # vec_mean_4 = vec_mean[4][0]
        # # vec_mean_5 = vec_mean[5][0]
        # print(self.cos_sim(vector_pre_1, vec_mean_0))
        # print(self.cos_sim(vector_pre_1, vec_mean_1))
        # print(self.cos_sim(vector_pre_1, vec_mean_2))
        # print(self.cos_sim(vector_pre_1, vec_mean_3))
        # print(self.cos_sim(vector_pre_2, vec_mean_0))
        # print(self.cos_sim(vector_pre_2, vec_mean_1))
        # print(self.cos_sim(vector_pre_2, vec_mean_2))
        # print(self.cos_sim(vector_pre_2, vec_mean_3))
        # if self.cos_sim(vector_pre_1, vec_mean_0) < threshold and self.cos_sim(vector_pre_1, vec_mean_1) < threshold and \
        #     self.cos_sim(vector_pre_1, vec_mean_2) < threshold and self.cos_sim(vector_pre_1, vec_mean_3) < threshold :
        #     # and self.cos_sim(vector_pre_1, vec_mean_4) < threshold and self.cos_sim(vector_pre_1, vec_mean_5) < threshold:
        #     class1 = [999]
        # else:
        #     class1 = model.predict_classes(np.reshape(temp1, newshape=(1, 24, 24, 32, 1)))
        # if self.cos_sim(vector_pre_2, vec_mean_0) < threshold and self.cos_sim(vector_pre_2, vec_mean_1) < threshold and \
        #     self.cos_sim(vector_pre_2, vec_mean_2) < threshold and self.cos_sim(vector_pre_2, vec_mean_3) < threshold:
        #     # and self.cos_sim(vector_pre_2, vec_mean_4) < threshold and self.cos_sim(vector_pre_2, vec_mean_5) < threshold:
        #     class2 = [999]
        # else:
        #     class2 = model.predict_classes(np.reshape(temp2, newshape=(1, 24, 24, 32, 1)))
        # print(np.array(vector_pre[0][:10]))
        # print(self.cos_sim(vector_pre[0], mean_1))
        # print(class1)
        # print(class2)
        # class1 = [999]
        # class2 = [1]
        # print('行人一为：{0}\t行人二为：{1}'.format(self.dict[str(label_1)], self.dict[str(label_2)]))



