# coding=UTF-8
import numpy as np
import matplotlib.pyplot as plt



def cos_sim(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a, b) / (a_norm * b_norm)
    return cos


if __name__ == '__main__':
    # plt.rcParams['font.sans-serif'] = ['KaiTi']
    # 0~24 25~56 57~80 81~113 114~147 148~177
    all_data = np.load('./data/order_featureVec_data.npy')
    all_label = np.load('./data/order_featureVec_label.npy')
    data = []
    data2 = []
    for j in range(148, 178):
        result = []
        result2 = []
        for i in range(148, 178):
            if i != j:
                result.append(cos_sim(all_data[j], all_data[i]))
        for i in range(114, 148):
            if i != j:
                result2.append(cos_sim(all_data[j], all_data[i]))
        # print(max(result), end=',')
        data.append(sum(result)/len(result))
        data2.append(sum(result2)/len(result2))
        # data.append(max(result))
        # data2.append(max(result2))
    x = np.arange(1, 31, 1)
    plt.xlabel('data')
    plt.ylabel('cos_similarity')
    plt.plot(x, data, label='inclass')
    plt.plot(x, data2, label='outclass')
    plt.legend()
    plt.ylim((-1, 1))
    plt.show()
    pass
