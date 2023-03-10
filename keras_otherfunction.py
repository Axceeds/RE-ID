import matplotlib.pyplot as plt
import numpy as np

# def plot_im(data):
#     # data (frame,24,32)
#     minium = 0
#     maxium = data.max()
#     for i in range(0, 20):
#         plt.subplot(1, 20, i + 1)
#         plt.imshow(data[i], interpolation='nearest', cmap='jet', vmin=minium, vmax=maxium)
#         # 根据像素绘制图片 origin表示渐变程度
#         plt.colorbar()
#         # 显示像素与数据对比
#         plt.xticks(())
#         plt.yticks(())
#         # 不显示坐标轴刻度
#     plt.show()

def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a, b) / (a_norm * b_norm)
    return cos
# '''计算裕度'''

# data_class_1 = self.all_data[:10, COLLECT_FRAME:]
# data_class_1 = np.reshape(data_class_1, newshape=(10, VALID_FRAME, IM_ROW, IM_COL, CHANEL))
# data_class_2 = self.all_data[10:20, COLLECT_FRAME:]
# data_class_2 = np.reshape(data_class_2, newshape=(10, VALID_FRAME, IM_ROW, IM_COL, CHANEL))
# data_class_3 = self.all_data[20:30, COLLECT_FRAME:]
# data_class_3 = np.reshape(data_class_3, newshape=(10, VALID_FRAME, IM_ROW, IM_COL, CHANEL))
# data_class_4 = self.all_data[30:40, COLLECT_FRAME:]
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
#
# dist1 = []
# dist2 = []
# dist3 = []
# dist4 = []
# sample = []
# # for i in range(10):
# #     dist1.append(np.linalg.norm(fea_vec_1[i] - mean_class_1))
# #     dist2.append(np.linalg.norm(fea_vec_2[i] - mean_class_2))
# #     dist3.append(np.linalg.norm(fea_vec_3[i] - mean_class_3))
# #     dist4.append(np.linalg.norm(fea_vec_4[i] - mean_class_4))
# for i in range(10):
#     dist1.append(cos_sim(fea_vec_1[i], mean_class_1))
#     dist2.append(cos_sim(fea_vec_2[i], mean_class_1))
#     dist3.append(cos_sim(fea_vec_3[i], mean_class_1))
#     dist4.append(cos_sim(fea_vec_4[i], mean_class_1))
#
# x = np.arange(1, 11)
# plt.plot(x, dist1, label='1')
# plt.plot(x, dist2, label='2')
# plt.plot(x, dist3, label='3')
# plt.plot(x, dist4, label='4')
# plt.legend()
# plt.show()

 # plt.legend()
        # plt.show()
        # count = 0
        # pre = model.predict_classes(np.reshape(data_test[0], newshape=(1, 39, 24, 32, 1)))
        # print(pre[0])
        # count = 0
        # for i in range(16):
        #     pre = model.predict_classes(np.reshape(data_test[i], newshape=(1, 39, 24, 32, 1)))
        #     print(str(i + 1) + '.true_label=' + str(np.where(label_test[i] > 0)[0]), end='\t')
        #     if i % 2 == 1:
        #         print('pred_label=[{0}]'.format(pre[0]))
        #     else:
        #         print('pred_label=[{0}]'.format(pre[0]), end='\t\t')
        #     if pre[0] == np.where(label_test[i] > 0):
        #         count = count + 1
        # print("new_acc={0}".format(count / 16))