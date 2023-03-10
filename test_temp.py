import file_process as fp
import numpy as np
import cv2 as cv
import os
import glob
from keras.models import Sequential, Model, load_model
import matplotlib.pyplot as plt

# a = str("2022.10.31-123:12")
# print(a)
#
# str = 'D:/2.PycharmProjects/仿真平台构建/data/all_data/abcd.xml'
# temp = os.path.basename(str)
# temp = temp.split('.')[0]
# filename = os.getcwd() + r'\data\all_dataprocessed'
# address = filename + '/' + temp + '_processed' + '.xml'
# print(address)
# 作图，动态刷新的红外热图
# file = 'Exp3_shx_1_2_processed.xml'
# dir_path = file.split('_')
# print(dir_path[3])
"""批量改名"""
# file_path = 'D:/2.PycharmProjects/仿真平台构建/data/all_dataprocessed/shx_1_1_origin.xml'
# path = 'D:/2.PycharmProjects/仿真平台构建/data/all_data/'
# fileList = os.listdir(path)
# for file in fileList:
#     file_name = file
#     old_name = path + file
#     new_name = path + 'Exp2_' + file
#     # print(old_name)
#     # print(new_name)
#     # break
#     os.rename(old_name, new_name)
# #
# path = 'D:/2.PycharmProjects/仿真平台构建/data/all_dataprocessed/'
# file_path = os.listdir(path)
# all_data = []
# all_label = []
# i=0
all_data = np.load('./data/npy/16centroid_data_24frame_6people.npy')
all_label = np.load('./data/npy/16centroid_label_24frame_6people.npy')
model = load_model('./model/centroid_model_24frame_4people.h5')
LSTM_output = Model(inputs=model.input, outputs=model.get_layer('LSTM_1').output)
vec_data = []
for i in range(len(all_data)):
    vec_data.append(LSTM_output.predict(np.reshape(all_data[i], newshape=(1, 24, 24, 32, 1))))
vec_data = np.array(vec_data)
vec_data = np.reshape(vec_data, newshape=(len(vec_data), 384))
arrIndex = all_label.argsort()
vec_data = vec_data[arrIndex]
all_label = all_label[arrIndex]

np.save('./data/order_featureVec_data.npy', arr=vec_data)
np.save('./data/order_featureVec_label.npy', arr=all_label)
# for file in file_path:
#     dir_path = file.split('_')
#     if str(dir_path[3]) == '1':
#         file_name = path + file
#         data_temp, length_temp, label_temp = fp.xml_process(file_name)
#         all_data.append(data_temp['side0'][8:47])
#         all_label.append(int(label_temp['label']))
#         print(i)
#     else:
#         if str(dir_path[3]) == '2':
#             file_name = path + file
#             data_temp, length_temp, label_temp = fp.xml_process(file_name)
#             all_data.append(data_temp['side1'][8:47])
#             all_label.append(int(label_temp['label']))
#             print(i)
#     i+=1
# path = os.getcwd() + '/data/all_dataprocessed/'
# file_path = os.listdir(path)
# for file in file_path:
#     dir_path = file.split('_')
#     if str(dir_path[0]) == 'Exp':
#         if str(dir_path[3]) == '1':
#             new_filepath = path + file
#             temp1, length, templabel = fp.xml_process(new_filepath)
#             print(temp1['side0'].shape)
# dict1 = {'side0': [], 'side1': [], 'top0': [], 'top1': []}
# for keys in dict1:
#     print(keys)
# print(file_path)
# print(fileList)
# dir_path = file_path.split('_')
# new_name = str(dir_path[0])+'_'+str(dir_path[1])+'_'+str(dir_path[2])+'_'+str(dir_path[3])+'.xml'
# os.rename(file_path, new_name)
# prefix_file_name = dir_path[-1].split('.')[0]
# dir_path = dir_path[:-1]
# dir_path = '/'.join(dir_path)
# files = os.listdir(dir_path)
# print(dir_path)
# print(dir_path[-1])
# print(new_name)
# print(files)
# a = {'1':[]}
# da = np.zeros((24,32))
# a['1']=da
# fp.save_data(a, os.getcwd()+'/temp123456.xml')
# b=fp.xml_process(os.getcwd()+'/data/all_data'+'/Exp2_gwp_1_1.xml')
# print(b)
# import math
# array_small = [i for i in range(30)]
# array_large = [i for i in range(100)]
# sample_rate = len(array_large) / len(array_small)
#
# array_large_sample = [
#     num for i, num in enumerate(array_large)
#     if math.floor(i % sample_rate) == 0
# ]
# print(array_large_sample)
# print(len(array_large_sample))

# OUTPUT:
# [0, 4, 7, 11, 14, 17, 21, 24, 27, 31, 34, 37, 41, 44, 47, 51, 54, 57, 61, 64, 67, 71, 74, 77, 81, 84, 87, 91, 94, 97]
# 30
"""删除指定目录下指定后缀的文件"""
# path = os.getcwd()+'/data/all_data/'
# path = os.getcwd()+'/data/all_dataprocessed/'
# for infile in glob.glob(os.path.join(path, '*.xml')):
#     os.remove(infile)
"""更改文件路径的basename"""
# temp = os.path.basename(str)
# filename = os.path.splitext(str)[0]
# address = filename+'_origin'+'.xml'
# print(temp)
"""读取xml"""
# temp = []
# temp = 'seda'
# # temp, length, label = fp.xml_process(r'D:\2.PycharmProjects\仿真平台构建\data\all_data\shx_1.xml')
#
# print(temp)

# str = r'D:\2.PycharmProjects\仿真平台构建\data\all_data\shx.xml'
# temp = os.path.basename(str)
# temp = temp.split('.')[0]
# filename = os.getcwd()+r'\data\all_dataprocessed'
# address = filename+'/'+temp+'_origin'+'.xml'
# print(address)
# data, length = fp.xml_process(os.getcwd()+'/data/'+'1.xml')
# print(data['side0'].shape)
# fp.save_para('processPara.xml','name','123',0)
# os.remove(path=os.getcwd()+'/mymodel.h5')

# all_data1 = process_data('D:/2.PycharmProjects/红外数据记录/2022.03.09第一次数据采集与实验/20组_全转化为正向_已经背景移除过1.npy',
#                              20, 47, 24, 32)
# all_data2 = process_data('D:/2.PycharmProjects/红外数据记录/2022.03.09第一次数据采集与实验/20组_全转化为正向_已经背景移除过2.npy',
#                              20, 47, 24, 32)
# all_data = np.vstack((all_data1, all_data2))
# temp1 = all_data[0]
# temp2 = all_data[1]
# np.save(file=os.getcwd()+'/data/'+'shi1.npy', arr=temp1)
# np.save(file=os.getcwd()+'/data/'+'shi2.npy', arr=temp2)
# model=build_model()
# a = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
#      1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
# b = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
#      0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1]
# c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
#      0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
# arr= []
# for i in range(len(a)):
#     if a[i] == 1 or b[i] == 1 or c[i] == 1:
#         arr.append(1)
#     else:
#         arr.append(0)
# arr = np.array(arr)
# print(np.sum(arr == 0))
# print(arr)
