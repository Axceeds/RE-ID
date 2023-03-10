import math
import os
import numpy as np
import matplotlib.pyplot as plt
import keras
import cv2 as cv
from tensorflow.keras.models import Sequential, Model, load_model
from sklearn.ensemble import IsolationForest
from pyod.models.abod import ABOD
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.combination import aom
from pyod.models.cd import CD
from pyod.models.copod import COPOD
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.ecod import ECOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.kde import KDE
from pyod.models.knn import KNN
from pyod.models.lmdd import LMDD
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.loci import LOCI
from pyod.models.lscp import LSCP
from pyod.models.mad import MAD
from pyod.models.mcd import MCD
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.rod import ROD
from pyod.models.sampling import Sampling
from pyod.models.sod import SOD
from pyod.models.so_gaal import SO_GAAL
from pyod.models.sos import SOS
from pyod.models.suod import SUOD
from pyod.models.vae import VAE
from pyod.models.xgbod import XGBOD

from sklearn.metrics import classification_report
import warnings
from pyod.utils.data import generate_data, get_outliers_inliers


def hight(matrix):
    frame = len(matrix)
    temp1 = []
    temp2 = []
    for i in range(frame):
        for j in range(11, 14):
            if len(max(np.where(matrix[i][j] > 0))) != 0:
                temp1.append(max(max(np.where(matrix[i][j] > 0))))
        if len(temp1) == 0:
            temp2.append(0)
        else:
            temp2.append(sum(temp1) / len(temp1))
    return sum(temp2) / len(temp2)


def area(matrix):
    frame = len(matrix)
    count = 0
    temp = []
    for i in range(frame):
        temp.append(len((np.where(matrix[i] > 0))[0]))

    return sum(temp) / len(temp)


if __name__ == '__main__':
    # 0~24 25~56 57~80 81~113 114~147 148~177
    warnings.filterwarnings("ignore")
    outlier_fraction = 0.05
    # knn hobs cblof abod
    classifiers = {
        'ABOD': ABOD(contamination=outlier_fraction),
        # 'AutoEncoder': AutoEncoder(contamination=outlier_fraction),
        # 'CBLOF': CBLOF(contamination=outlier_fraction),
        # # 'COF': COF(contamination=outlier_fraction),
        # # 'aom': aom(),
        # # 'CD': CD(contamination=outlier_fraction),
        # # 'COPOD': COPOD(contamination=outlier_fraction),
        # 'DeepSVDD': DeepSVDD(contamination=outlier_fraction),
        # # 'ECOD': ECOD(contamination=outlier_fraction),
        # # 'FeatureBagging': FeatureBagging(contamination=outlier_fraction),
        # 'HBOS': HBOS(contamination=outlier_fraction),
        # 'IForest': IForest(contamination=outlier_fraction),
        # 'KDE': KDE(contamination=outlier_fraction),
        # 'KNN': KNN(contamination=outlier_fraction),
        # 'LMDD': LMDD(contamination=outlier_fraction),
        # 'LODA': LODA(contamination=outlier_fraction),
        # 'LOF': LOF(contamination=outlier_fraction),
        # 'LOCI': LOCI(contamination=outlier_fraction),
        # # 'MAD': MAD(),
        # 'MO_GAAL': MO_GAAL(contamination=outlier_fraction),
        # 'OCSVM': OCSVM(contamination=outlier_fraction),
        # 'PCA': PCA(contamination=outlier_fraction),
        # # 'ROD': ROD(contamination=outlier_fraction),
        # 'Sampling': Sampling(contamination=outlier_fraction),
        # 'SOD': SOD(contamination=outlier_fraction),
        # 'SO_GAAL': SO_GAAL(contamination=outlier_fraction),
        # 'SOS': SOS(contamination=outlier_fraction),
        # 'SUOD': SUOD(contamination=outlier_fraction),
        # 'VAE': VAE(contamination=outlier_fraction),
        # 'XGBOD': XGBOD(contamination=outlier_fraction)
    }
    all_data = np.load('./data/order_featureVec_data.npy')
    all_label = np.load('./data/order_featureVec_label.npy')
    # label_train = np.zeros(114, )
    #

    label = []
    origin_data = np.load('./data/npy/16centroid_data_24frame_6people.npy')

    height_mean = [10.653698517185349, 11.416868355557547, 10.32569009264441, 11.822536981136164,
                   10.671057894961628, 9.81077143981074]
    area_mean = [84.17333333333335, 75.12499999999999, 66.73437500000001, 82.11363636363636, 66.74264705882352,
                 69.85138888888888]
    # label1 = []
    # label2 = []
    # label3 = []
    shoreld = 7.5
    acc1 = []
    acc2 = []
    acc3 = []
    for s in range(1):
        # for i, (clf_name, clf) in enumerate(classifiers.items()):
        new_data = all_data[0:114]
        num_example = len(new_data[0:114])
        arr = np.arange(num_example)
        np.random.shuffle(arr)
        new_data = new_data[arr]
        ratio = 0.7
        num_train = np.int(num_example * ratio)
        train_data = new_data[:num_train]
        test_data = new_data[num_train:]
        test_data_new = np.vstack((test_data, all_data[114:]))
        clf_name = 'ABOD'
        clf = ABOD(contamination=outlier_fraction)
        print('i=', s)
        # print(i + 1, 'fitting', clf_name)
        clf.fit(train_data)
        label0 = clf.predict(test_data_new)
        #     label.append(label0)
        # label = np.array(label)
        # processed_label = []
        # for i in range(len(label[0])):
        #     if np.sum(label[:, i] == 0) >= np.sum(label[:, i] == 1):
        #         processed_label.append(0)
        #     else:
        #         processed_label.append(1)
        #
        # processed_label = np.array(processed_label)
        # print(processed_label[:len(test_data)])
        # print(np.sum(processed_label[:len(test_data)] == 0)/len(test_data))
        # processed_label = np.array(processed_label)
        # print(processed_label[len(test_data):])
        # a = processed_label[len(test_data):]
        # print(np.sum(a == 1)/64)
        #     print(test_data_new.shape)
        label1 = clf.predict(train_data)
        label2 = clf.predict(test_data)
        label3 = clf.predict(all_data[114:])
        scores1 = clf.decision_function(test_data)
        scores2 = clf.decision_function(all_data[114:])

        label1 = np.array(label1)
        label2 = np.array(label2)
        label3 = np.array(label3)
        print(label1)
        print('已知类的测试结果', label2)
        print('未知类的测试结果', label3)
        list1 = []
        list2 = []
        list3 = []
        list4 = []
        for i in range(len(label3)):
            if label3[i] == 1:
                print(scores2[i] * 1000)
                list1.append(-scores2[i] * 1000)
        print('-------------------------------------------')
        for i in range(len(label3)):
            if label3[i] == 0:
                print(scores2[i] * 1000)
                list2.append(-scores2[i] * 1000)
        print('-------------------------------------------')
        for i in range(len(label2)):
            if label2[i] == 1:
                print(scores1[i] * 1000)
                list3.append(-scores1[i] * 1000)
        print('-------------------------------------------')
        for i in range(len(label2)):
            if label2[i] == 0:
                print(scores1[i] * 1000)
                list4.append(-scores1[i] * 1000)
        x1 = np.arange(1, len(list1) + 1, 1)
        x2 = np.arange(1, len(list2) + 1, 1)
        # x3 = np.arange(1, len(list3) + 1, 1)
        x4 = np.arange(1, len(list4) + 1, 1)
        plt.plot(x1, list1, label='unknown=1')
        plt.plot(x2, list2, label='unknown=0')
        # plt.plot(x3, list3)
        plt.plot(x4, list4, label='known=0')
        plt.legend()
        plt.show()
        # for i in range(len(label3)):
        #     if label3[i] == 0:
        #         h1 = hight(origin_data[i + 114])
        #         a1 = area(origin_data[i + 114])
        #         t1 = []
        #         t2 = []
        #         for k in range(4):
        #             t1.append(math.sqrt((height_mean[k]-h1)**2)+math.sqrt((area_mean[k]-a1)**2))
        #         odistance = min(t1)
        #         # print(odistance)
        #         if odistance >= shoreld:
        #             label3[i] = 1
        #         else:
        #             pass
        # print('-----------------------------------')
        # for i in range(len(label1)):
        #     if label1[i] == 0:
        #         h1 = hight(origin_data[i])
        #         a1 = area(origin_data[i])
        #         t1 = []
        #         t2 = []
        #         for k in range(4):
        #             t1.append(math.sqrt((height_mean[k]-h1)**2)+math.sqrt((area_mean[k]-a1)**2))
        #         odistance = min(t1)
        #         # print(odistance)
        #         if odistance >= shoreld:
        #             label1[i] = 1
        #         else:
        #             pass
        # print('-----------------------------------')
        # for i in range(len(label2)):
        #     if label2[i] == 0:
        #         h1 = hight(origin_data[i+79])
        #         a1 = area(origin_data[i+79])
        #         t1 = []
        #         t2 = []
        #         for k in range(4):
        #             t1.append(math.sqrt((height_mean[k]-h1)**2)+math.sqrt((area_mean[k]-a1)**2))
        #         odistance = min(t1)
        #         # print(odistance)
        #         if odistance >= shoreld:
        #             label2[i] = 1
        #         else:
        #             pass
        # print('已知类，抽取作为拟合数据的正确率：', end='')
        # print(np.sum(label1 == 0) / (num_train))
        # print('已知类，抽取作为测试数据的正确率：', end='')
        # print(np.sum(label2 == 0) / ((num_example - num_train)))
        # print('未知类，作为测试数据的正确率：', end='')
        # print(np.sum(label3 == 1) / (64))
        acc1.append(np.sum(label1 == 0) / (num_train))
        acc2.append(np.sum(label2 == 0) / ((num_example - num_train)))
        acc3.append(np.sum(label3 == 1) / (64))
    print('已知类，抽取作为拟合数据的正确率：', sum(acc1) / len(acc1))
    print('已知类，抽取作为测试数据的正确率：', sum(acc2) / len(acc2))
    print('未知类，作为测试数据的正确率：', sum(acc3) / len(acc3))
    # print(len(label1))
    # print(len(label2))
    # print(len(label3))
    # print(label1)
    # print(label2)
    # print(label3)

    # for j in range(len(all_data[114:])):
    #     print(-scores2[j] * 10 ** 6, end='\t')
    #     print(label3[j])

    # arrIndex = all_label.argsort()
    # all_data = all_data[arrIndex]
    # all_label = all_label[arrIndex]
    # np.save('./data/order_featureVec_data.npy', arr=all_data)
    # np.save('./data/order_featureVec_label.npy', arr=all_label)
