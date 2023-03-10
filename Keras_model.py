import numpy as np
import matplotlib.pyplot as plt
import keras
import cv2 as cv
import os
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers import BatchNormalization
from keras.layers import Dropout
import file_process as fp
import math
import time


# 作出红外热图
def plot_im(data):
    # data (frame,24,32)
    minium = 0
    maxium = data.max()
    for i in range(0, 20):
        plt.subplot(1, 20, i + 1)
        plt.imshow(data[i], interpolation='nearest', cmap='jet', vmin=minium, vmax=maxium)
        # 根据像素绘制图片 origin表示渐变程度
        plt.colorbar()
        # 显示像素与数据对比
        plt.xticks(())
        plt.yticks(())
        # 不显示坐标轴刻度
    plt.show()


# 计算余弦相似度
def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a, b) / (a_norm * b_norm)
    return cos


# 作图，动态刷新的红外热图
def plot_im_dynastic(data):
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


# 搭建模型
def build_model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(None, 24, 32, 1)))  # (None,50,50,1)
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Dropout(0.20)))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))  # (None,50,50,1)
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Dropout(0.20)))
    model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu')))  # (None,50,50,1)
    model.add(TimeDistributed(BatchNormalization()))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
    model.add(TimeDistributed(Dropout(0.20)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(384, name='LSTM_1'))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    # model = load_model(r'D:\2.PycharmProjects\Keras/model/newmodel2.h5')
    model.summary()
    return model


# 载入数据，将xml数据转化为npy
def load_data():
    """数据载入"""
    path = 'D:/2.PycharmProjects/仿真平台构建/data/all_dataprocessed/'
    file_path = os.listdir(path)
    all_data = []
    all_label = []
    i = 0
    for file in file_path:
        dir_path = file.split('_')
        if str(dir_path[3]) == '1':
            file_name = path + file
            data_temp, length_temp, label_temp = fp.xml_process(file_name)
            all_data.append(data_temp['side0'][8:47])
            all_label.append(int(label_temp['label']))
        elif str(dir_path[3]) == '2':
            file_name = path + file
            data_temp, length_temp, label_temp = fp.xml_process(file_name)
            all_data.append(data_temp['side1'][8:47])
            all_label.append(int(label_temp['label']))
    all_data = np.array(all_data)
    all_label = np.array(all_label)
    np.save(file=r'D:\2.PycharmProjects\Keras\data/newdata_39frame.npy', arr=all_data)
    np.save(file=r'D:\2.PycharmProjects\Keras\data/newlabel_39frame.npy', arr=all_label)


# 关键帧提取，平均提取、质心中点提取
def cut_data():
    """数据的筛选、阈值设定"""
    # new_data = []
    # new_label = []
    # sample = [i for i in range(24)]
    # c = 0
    # # data (118 47 24 32)
    # for j in range(len(all_data)):
    #     temp = all_data[j]
    #     list = []
    #     for i in range(len(temp)):
    #         count = np.sum(temp[i] > 0)
    #         if count < 25:
    #             list.append(i)
    #     a = np.delete(temp, list, 0)
    #     print('数据为 arr={0}\t标签为= {1}'.format(a.shape,all_label[j]))
    #     if len(a) >= 25:
    #         old_arr = a
    #         sample_rate = len(old_arr) / len(sample)
    #         new_arr = np.array([
    #             num for i, num in enumerate(old_arr)
    #             if math.floor(i % sample_rate) == 0
    #         ])
    #         new_data.append(new_arr)
    #         new_label.append(all_label[j])
    #
    # all_data = np.array(new_data)
    # all_label = np.array(new_label)
    # np.save(file=r'D:\2.PycharmProjects\Keras\data/newdata.npy', arr=all_data)
    # np.save(file=r'D:\2.PycharmProjects\Keras\data/newlabel.npy', arr=all_label)


if __name__ == '__main__':
    '''搭模型'''
    IM_ROW = 24
    IM_COL = 32
    ALL_FRAME = 47  # 单个视频采集总帧数
    COLLECT_FRAME = 8  # 背景采集的帧数
    VALID_FRAME = ALL_FRAME - COLLECT_FRAME  # 有效帧
    SAMPLE_NUM = 40  # 总的样本数
    CHANEL = 1  # 通道数
    data_train = []  # 训练数据
    data_test = []  # 测试数据
    label_train = []  # 训练标签
    label_test = []  # 测试标签
    matrix_list = []
    acc_list = []

    all_data = np.load('./data/npy/centroid_data_24frame.npy')
    all_label = np.load('./data/npy/centroid_label_24frame.npy')
    matrix = np.zeros((4, 4))
    T1 = time.time()
    for p in range(100):
        model = build_model()
        '''参数设置'''
        num_example = len(all_data)
        arr = np.arange(num_example)
        np.random.shuffle(arr)
        data = all_data[arr]
        data_label = all_label[arr]
        print(data.shape)
        ratio = 0.8
        num_train = np.int(num_example * ratio)
        data_train = data[:num_train]
        label_train = data_label[:num_train]
        data_test = data[num_train:]
        label_test = data_label[num_train:]
        data_train = np.reshape(data_train, newshape=(num_train, 24, IM_ROW, IM_COL, CHANEL))
        data_test = np.reshape(data_test, newshape=(SAMPLE_NUM - num_train, 24, IM_ROW, IM_COL, CHANEL))
        label_train = keras.utils.to_categorical(np.array(data_label[:num_train]))
        label_test = keras.utils.to_categorical(np.array(data_label[num_train:]))
        history = model.fit(data_train, label_train, batch_size=16, epochs=20)
        # model.save(r'D:\2.PycharmProjects\Keras/data/exp_10_39frame/newmodel10.h5')

        list_true = []
        list_pred = []
        count = 0
        for i in range(len(label_test)):
            pre = model.predict_classes(np.reshape(data_test[i], newshape=(1, 24, 24, 32, 1)))
            list_true.append(np.where(label_test[i] > 0)[0])
            list_pred.append(pre[0])
            #     print(str(i + 1) + '.true_label=' + str(np.where(label_test[i] > 0)[0]), end='\t')
            #     if i % 2 == 1:
            #         print('pred_label=[{0}]'.format(pre[0]))
            #     else:
            #         print('pred_label=[{0}]'.format(pre[0]), end='\t\t')
            if pre[0] == np.where(label_test[i] > 0):
                count = count + 1
        # print("new_acc={0}".format(count / len(label_test)))
        for i in range(len(list_pred)):
            matrix[int(list_true[i])][int(list_pred[i])] += 1
        matrix_list.append(matrix)
        acc_list.append(count / len(label_test))
        print('第{0}次实验'.format(p + 1))

        for k in range(len(matrix_list)):
            print(matrix_list[k])
        #     print(acc_list[k])
    # acc_list = np.array(acc_list)
    # print(acc_list)
    T2 = time.time()
    print('程序运行时间:%s秒' % (T2 - T1))
    print('平均准确率为：{0}'.format(np.mean(acc_list)))

    '''模型train'''
    # 暂时用不到了，模型训练好就保存.h5了
    # print(history.history)
    # evaluate model
    # val_loss, val_acc = model.evaluate(data_test, label_test, verbose=0)
    # print('summary: val_loss: %f, val_acc: %f' % (val_loss, val_acc))
    # count = 0
    # for i in range(len(label_test)):
    #     pre = model.predict_classes(np.reshape(data_test[i], newshape=(1, 39, 24, 32, 1)))
    #     print(str(i + 1) + '.true_label=' + str(np.where(label_test[i] > 0)[0]), end='\t')
    #     if i % 2 == 1:
    #         print('pred_label=[{0}]'.format(pre[0]))
    #     else:
    #         print('pred_label=[{0}]'.format(pre[0]), end='\t\t')
    #     if pre[0] == np.where(label_test[i] > 0):
    #         count = count + 1
    # print("new_acc={0}".format(count / len(label_test)))
    '''预测结果'''
    # 抽取单个层结果输出，这里输出LSTM之后的256维度特征向量
    # LSTM_output = Model(inputs=model.input, outputs=model.get_layer('LSTM_1').output)
    # vector_pre = LSTM_output.predict(np.reshape(data_test[0], newshape=(1, 39, 24, 32, 1)))

    '''计算裕度'''
    # data_class_1 = all_data[:10, COLLECT_FRAME:]
    # data_class_1 = np.reshape(data_class_1, newshape=(10, VALID_FRAME, IM_ROW, IM_COL, CHANEL))
    # data_class_2 = all_data[10:20, COLLECT_FRAME:]
    # data_class_2 = np.reshape(data_class_2, newshape=(10, VALID_FRAME, IM_ROW, IM_COL, CHANEL))
    # data_class_3 = all_data[20:30, COLLECT_FRAME:]
    # data_class_3 = np.reshape(data_class_3, newshape=(10, VALID_FRAME, IM_ROW, IM_COL, CHANEL))
    # data_class_4 = all_data[30:40, COLLECT_FRAME:]
    # data_class_4 = np.reshape(data_class_4, newshape=(10, VALID_FRAME, IM_ROW, IM_COL, CHANEL))
    # fea_vec_1 = []
    # fea_vec_2 = []
    # fea_vec_3 = []
    # fea_vec_4 = []
    # for i in range(10):
    #     vector_pre = LSTM_output.predict(np.reshape(data_class_1[i], newshape=(1, 39, 24, 32, 1)))
    #     fea_vec_1.append(vector_pre[0])
    # for i in range(10):
    #     vector_pre = LSTM_output.predict(np.reshape(data_class_2[i], newshape=(1, 39, 24, 32, 1)))
    #     fea_vec_2.append(vector_pre[0])
    # for i in range(10):
    #     vector_pre = LSTM_output.predict(np.reshape(data_class_3[i], newshape=(1, 39, 24, 32, 1)))
    #     fea_vec_3.append(vector_pre[0])
    # for i in range(10):
    #     vector_pre = LSTM_output.predict(np.reshape(data_class_4[i], newshape=(1, 39, 24, 32, 1)))
    #     fea_vec_4.append(vector_pre[0])
    # fea_vec_1 = np.array(fea_vec_1)
    # fea_vec_2 = np.array(fea_vec_2)
    # fea_vec_3 = np.array(fea_vec_3)
    # fea_vec_4 = np.array(fea_vec_4)
    # mean_class_1 = np.mean(fea_vec_1, axis=0)
    # mean_class_2 = np.mean(fea_vec_2, axis=0)
    # mean_class_3 = np.mean(fea_vec_3, axis=0)
    # mean_class_4 = np.mean(fea_vec_4, axis=0)
    # print(fea_vec_1[0][:10])
    # dist1 = []
    # dist2 = []
    # dist3 = []
    # dist4 = []
    # sample = []
    # for i in range(10):
    #     dist1.append(np.linalg.norm(fea_vec_1[i] - mean_class_1))
    #     dist2.append(np.linalg.norm(fea_vec_2[i] - mean_class_2))
    #     dist3.append(np.linalg.norm(fea_vec_3[i] - mean_class_3))
    #     dist4.append(np.linalg.norm(fea_vec_4[i] - mean_class_4))
    # for i in range(10):
    #     dist1.append(cos_sim(fea_vec_1[i], mean_class_3))
    #     dist2.append(cos_sim(fea_vec_2[i], mean_class_3))
    #     dist3.append(cos_sim(fea_vec_3[i], mean_class_3))
    #     dist4.append(cos_sim(fea_vec_4[i], mean_class_3))
    # print(dist1)
    # print(dist2)
    # print(dist3)
    # print(dist4)

    # x = np.arange(1, 11)
    # plt.plot(x, dist1, label='1')
    # plt.plot(x, dist2, label='2')
    # plt.plot(x, dist3, label='3')
    # plt.plot(x, dist4, label='4')
    # np.save(r'D:\2.PycharmProjects\仿真平台构建\data\mean_1.npy', arr=mean_class_1)
    # np.save(r'D:\2.PycharmProjects\仿真平台构建\data\mean_2.npy', arr=mean_class_2)
    # np.save(r'D:\2.PycharmProjects\仿真平台构建\data\mean_3.npy', arr=mean_class_3)
    # np.save(r'D:\2.PycharmProjects\仿真平台构建\data\mean_4.npy', arr=mean_class_4)

    '''
    for i in range(10):
        sample.append(keras.losses.cosine_proximity(fea_vec_1[i], mean_class_1))
        sample.append(keras.losses.cosine_proximity(fea_vec_2[i], mean_class_1))
    print(np.array(sample))
    '''

    # plt.legend()
    # plt.show()
    # count = 0
    # temp_data1 = np.array(fp.xml_process(r'D:\2.PycharmProjects\红外数据记录\2022.03.09第一次数据采集与实验\3/' + '1.xml'))
    # all_data1 = np.array(temp_data1[0]['side0'])
    # all_data1 = all_data1[COLLECT_FRAME:]
    # pre = model.predict_classes(np.reshape(all_data1, newshape=(1, 39, 24, 32, 1)))
    # print(pre[0])
