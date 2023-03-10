import cv2 as cv
import numpy as np
import matrix_process as mp
from scipy import signal

class SequenceProcess:
    # 序列处理类
    # 一定要注意的是，这里缓存的所有数据并没有深度复制，而是指向原始数据的指针
    # 一旦修改，将会造成不可预料的后果。

    def __init__(self):
        self.queue_of_temporal_filter = []  # 时域滤波器缓冲区
        self.queue_of_rb = []  # 背景移除缓冲区
        self.queue_of_diff = []  # 差分缓冲区

        self.back = None  # 背景
        self.threshold = None  # 阈值
        self.prior_list = None  # 先验图列表
        self.densityMatrix = None  # 支持度矩阵

        self.miu_n = None  # 模糊背景移除使用的参数
        self.mog = None  # mog对象

        self.background_is_set = False  # 背景是否创建标志位
        self.prior_is_set = False  # 先验图是否创建标志位
        self.mog_is_created = False  # MOG对象创建标志位

        self.pre_figure = None  # 光流使用的前一幅图
        self.flow_figure = None  # 光流图
        self.pre_figure_of_optical_flow_is_set = False  # 光流标志位

        self.GaussKernel = {1: [1],
                            3: [0.25, 0.5, 0.25],
                            5: [0.0625, 0.25, 0.375, 0.25, 0.0625],
                            7: [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]}  # 高斯核
        for i in 9, 11:  # 计算高斯核
            st = 0.3 * ((i - 1) * 0.5 - 1) + 0.8
            kernel = cv.getGaussianKernel(i, st)
            self.GaussKernel[i] = kernel

    def clear(self):
        """########################################
        标志位置0，在后续使用中，将会重新创建缓存区
        输入：
            None
        输出：
            None
        备注：
        ########################################"""
        self.queue_of_temporal_filter.clear()
        self.queue_of_rb.clear()
        self.queue_of_diff.clear()
        self.back = None  # 背景
        self.threshold = None  # 阈值
        self.prior_list = None  # 先验图列表
        self.densityMatrix = None  # 支持度矩阵
        self.miu_n = None  # 模糊背景移除使用的参数
        self.mog = None  # mog对象
        self.background_is_set = False  # 背景是否创建标志位
        self.prior_is_set = False  # 先验图是否创建标志位
        self.mog_is_created = False  # MOG对象创建标志位
        self.pre_figure = None  # 光流使用的前一幅图
        self.flow_figure = None  # 光流图
        self.pre_figure_of_optical_flow_is_set = False  # 光流标志位

    @staticmethod
    def put_figure_queue(queue, figure, max_len):
        # 图像队列输入 队列queue（列表）新图像figure（uint8 ndarray）队列最大长度max_len（整数）
        if len(queue) == 0:
            queue.append(figure)
        elif queue[-1].shape[0] != figure.shape[0] or queue[-1].shape[1] != figure.shape[1]:
            queue.clear()  # 新加入的矩阵和队列最后一个矩阵维度不一致，重置队列
            queue.append(figure)
        elif len(queue) < max_len:  # 队列长度小于max_len，输入figure
            queue.append(figure)
        elif len(queue) == max_len:  # 队列长度等于max_len，输入figure，队首出队
            del queue[0]
            queue.append(figure)
        else:  # 队列长度大于max_len，队首出队
            del queue[0]

    def temporal_filter(self, figure, mode, win_size):
        """########################################
        时间维度滤波
        输入：
            1、图像figure（uint8 ndarray）
            2、滤波模式mode（字符串）
            3、窗口长度win_size（整数）
        输出：
            1、滤波后的图像（uint8 ndarray）
        备注：
        ########################################"""
        self.put_figure_queue(self.queue_of_temporal_filter, figure, win_size)
        if len(self.queue_of_temporal_filter) < win_size:  # 如果缓冲区长度不足，则直接使用当前图像填满
            return self.temporal_filter(figure, mode, win_size)
        elif mode == 'None':
            return figure
        elif mode == 'average':
            return self.average_filter()
        elif mode == 'Gauss':
            return self.gauss_filter()
        elif mode == 'median':
            return self.median_filter()
        else:
            print('不存在这种时间滤波方法，返回原数据')
            return figure

    def average_filter(self):
        # 时间维度均值滤波
        new_figure = np.zeros_like(self.queue_of_temporal_filter[-1], dtype='float64')
        for figure in self.queue_of_temporal_filter:
            new_figure += figure
        if self.queue_of_temporal_filter[-1].dtype == 'uint8':
            return (new_figure / len(self.queue_of_temporal_filter)).astype('uint8')
        else:
            return new_figure / len(self.queue_of_temporal_filter)

    def gauss_filter(self):
        # 时间维度高斯滤波
        n = len(self.queue_of_temporal_filter)
        if (n - 1) % 2 != 0:
            print('时域高斯滤波缓存长度必须是奇数，返回原数据')
            return self.queue_of_temporal_filter[-1]
        elif n not in [1, 3, 5, 7, 9, 11]:
            print('大于11的高斯核还未设置，返回原数据')
            return self.queue_of_temporal_filter[-1]
        else:
            new_figure = np.zeros_like(self.queue_of_temporal_filter[-1], dtype='float64')
            for i in range(n):
                new_figure += self.queue_of_temporal_filter[i] * self.GaussKernel[n][i]
            return new_figure.astype('uint8')

    def median_filter(self):
        # 时间维度中值滤波
        n = len(self.queue_of_temporal_filter)
        if (n - 1) % 2 != 0:
            print('时域滤波缓存长度必须是奇数，返回原数据')
            return self.queue_of_temporal_filter[-1]
        else:
            new_figure = np.zeros_like(self.queue_of_temporal_filter[-1], dtype='float64')
            large_matrix = np.array(self.queue_of_temporal_filter)
            for i in range(new_figure.shape[0]):
                for j in range(new_figure.shape[1]):
                    large_matrix[:, i, j].sort()
                    new_figure[i, j] = large_matrix[(n - 1) // 2, i, j]

            if self.queue_of_temporal_filter[-1].dtype == 'uint8':
                return new_figure.astype('uint8')
            else:
                return new_figure

    def set_background(self, background):
        # 设置背景模型
        self.background_is_set = True
        self.back = np.zeros_like(background[0], dtype='float64')
        for frame in background:
            self.back = self.back + frame
        self.back = self.back / background.shape[0]  # 算了个均值
        # threshold
        queue = background[2:, ...]
        back_list = np.array(queue)
        self.threshold = np.zeros_like(queue[-1], dtype='float64')
        for i in range(self.threshold.shape[0]):
            for j in range(self.threshold.shape[1]):
                v = back_list[:, i, j].var()
                self.threshold[i, j] = v ** 0.5
        # miuN
        self.miu_n = np.zeros_like(self.back)

    def set_prior(self, prior):
        # 设置先验图
        self.prior_is_set = True
        self.prior_list = prior
        self.densityMatrix = np.zeros_like(self.prior_list)
        for i in range(self.prior_list.shape[0]):
            for j in range(self.prior_list.shape[1]):
                for k in range(self.prior_list.shape[2]):
                    if self.prior_list[i][j][k] != 0:
                        self.densityMatrix[i][j][k] = 1 / self.prior_list[i][j][k]

    def remove_background(self, figure, mode, buffer_len):
        """########################################
        背景移除
        输入：
            1、原始图像figure（uint8 ndarray）
            2、移除算法mode（字符串）
            3、缓存长度bufferLen（整数）
        输出：
            1、背景移除后的图像
        ########################################"""
        if not self.background_is_set:  # 如果没有创建背景和阈值，则开始采集
            print('没有载入有效背景，开始采集背景')
            self.put_figure_queue(self.queue_of_rb, figure, buffer_len)
            if len(self.queue_of_rb) < buffer_len:  # 如果缓冲区长度不足，则返回原数据
                return figure
            else:  # 如果缓冲区达到预定长度，计算背景和阈值
                print('缓冲区达到预定长度')
                self.set_background(np.array(self.queue_of_rb))
                return figure
        else:  # 如果已经创建了背景和阈值，则开始背景移除
            self.put_figure_queue(self.queue_of_rb, figure, 2)
            if len(self.queue_of_rb) < 2:  # 如果缓冲区长度不足，则返回原数据
                return figure
            if mode == 'None':
                return figure
            elif mode == 'BS':
                return self.BS(figure)
            elif mode == 'BS-SEG':
                return self.BS_SEG(figure)
            elif mode in ('DFBR', 'DFBR_back', 'DFBR_threshold'):
                return self.DFBR(1e-2, mode)
            elif mode in ('AGBR-PM', 'AGBR-PM_back', 'AGBR-PM_threshold', 'AGBR-PM_prior', 'AGBR-PM_density'):
                return self.AGBR_PM(1e-2, mode)
            elif mode == 'MOG':
                return self.mog(figure)  # OpenCV算法
            elif mode == 'FBR':
                return self.FBR(figure)
            elif mode == 'GBR':
                return self.GBR(figure)
            else:
                print('不存在的背景移除方法，返回原数据')
                return figure

    def BS(self, figure):
        # 背景减除法
        newFigure = figure.astype('int32') - self.back
        newFigure = newFigure * (newFigure > 0)
        return newFigure.astype('uint8')

    def BS_SEG(self, figure):
        # 背景移除和阈值分割结合
        subtraction = figure.astype('int32') - self.back
        subtraction = (subtraction - subtraction.min()).astype('uint8')
        mask = mp.segment(subtraction, 'triangle2', 0)
        mask = mask > 0
        new_figure = figure * mask
        return new_figure.astype('uint8')

    def DFBR(self, lr, mode):
        figure = self.queue_of_rb[-1].astype('int32')
        pre_figure = self.queue_of_rb[-2].astype('int32')

        diff_figure = figure - self.back  # 获取当前图像和背景的差值
        unfuzzy_foreground = diff_figure > self.threshold * 2  # 非模糊前景计算

        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype='uint8')  # 腐蚀核心
        unfuzzy_foreground = cv.erode(unfuzzy_foreground.astype('uint8'), kernel, iterations=1)  # 腐蚀
        unfuzzy_foreground = mp.connected_component(unfuzzy_foreground,5)  # 连通域分析

        fuzzy_foreground = 1 / (1 + np.exp(-2.2 * (diff_figure - 2 * self.threshold) / (self.threshold + 1)))  # 模糊前景计算
        kernel = mp.vector_to_matrix(self.GaussKernel[3], self.GaussKernel[3])  # 高斯核
        fuzzy_foreground = signal.convolve2d(fuzzy_foreground, kernel, boundary='symm', mode='same')  # 空间模糊化，借用邻近像素点信息
        # 更新背景
        self.back = fuzzy_foreground * self.back + (1 - fuzzy_foreground) * ((1 - lr) * self.back + figure * lr)
        # 更新阈值
        deviation = figure - pre_figure
        new_lr = (1 - fuzzy_foreground) * lr * 0.1
        self.threshold = (1 - new_lr) * self.threshold + new_lr * 2.5 * np.abs(deviation)
        # 前景重建
        unfuzzy_foreground = self.complement(unfuzzy_foreground, figure, 1.5)  # 前景补足
        # 背景移除结果
        new_figure = unfuzzy_foreground * figure

        if mode == 'DFBR_back':
            return self.back
        elif mode == 'DFBR_threshold':
            return self.threshold
        elif mode == 'DFBR':
            return new_figure.astype('uint8')
        else:
            return new_figure.astype('uint8')

    def AGBR_PM(self, lr, mode):
        # 结合先验知识图的自适应背景移除
        figure = self.queue_of_rb[-1].astype('int32')
        pre_figure = self.queue_of_rb[-2].astype('int32')

        integrated_dm = np.zeros_like(figure)  # 聚合后的支持度矩阵
        lr_matrix = np.ones_like(figure)  # 学习率矩阵
        mask1 = np.zeros_like(figure)  # 对应情况：有先验热图
        mask2 = np.zeros_like(figure)  # 对应情况：没有先验热图的背景
        lr_matrix = lr * lr_matrix  # 对应情况：没有先验热图的前景，之后会替换其中的值

        diff_figure = figure - self.back  # 获取当前图像和背景的差值
        unfuzzy_foreground = diff_figure > self.threshold * 2  # 非模糊前景计算

        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype='uint8')  # 腐蚀核心
        test_frame = cv.erode(unfuzzy_foreground.astype('uint8'), kernel, iterations=1)  # 腐蚀
        test_frame = mp.connected_component(test_frame, 5)  # 连通域分析
        test_frame = self.complement(test_frame, figure, 1.5)  # 前景补足

        if self.prior_is_set:
            for order in range(len(self.prior_list)):
                switch = self.check_switch(figure,order)  # 检查开关状态
                if switch:  # 如果背景中先验热源此时与人体发生交叠，那么所在区域不进行背景移除和更新
                    is_cross = self.check_cross(test_frame, order)
                    if is_cross:
                        unfuzzy_foreground = unfuzzy_foreground+(unfuzzy_foreground * self.prior_list[order])
                    else:
                        unfuzzy_foreground = unfuzzy_foreground-(unfuzzy_foreground * self.prior_list[order])
                    unfuzzy_foreground = unfuzzy_foreground > 0

            kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype='uint8')  # 腐蚀核心
            unfuzzy_foreground = cv.erode(unfuzzy_foreground.astype('uint8'), kernel, iterations=1)  # 腐蚀
            unfuzzy_foreground = mp.connected_component(unfuzzy_foreground, 5)  # 连通域分析
            # 前景重建，此处是消除腐蚀带来的核型（十字型）空洞
            unfuzzy_foreground = self.complement(unfuzzy_foreground, figure, 1.5)  # 前景补足
            for order in range(len(self.prior_list)):
                mask1 = mask1 + self.prior_list[order]
                integrated_dm = integrated_dm + self.densityMatrix[order]
            # 得到掩膜
            mask1 = (mask1 > 0)
            mask2 = mask1 + unfuzzy_foreground
            mask2 = (mask2 == 0)
            lr_matrix = lr_matrix * (~mask1) + (mask1*lr*integrated_dm)
            lr_matrix = lr_matrix * (~mask2) + (mask2*1.5*lr)

        else:
            print('WARNING:未载入先验热图')
            unfuzzy_foreground = test_frame
            mask2 = test_frame
            mask2 = (mask2 == 0)
            lr_matrix = lr_matrix * (~mask2) + (mask2 * 1.5 * lr)


        fuzzy_foreground = 1 / (1 + np.exp(-2.2 * (diff_figure - 2 * self.threshold) / (self.threshold + 1)))  # 模糊前景计算
        kernel = mp.vector_to_matrix(self.GaussKernel[3], self.GaussKernel[3])  # 高斯核
        fuzzy_foreground = signal.convolve2d(fuzzy_foreground, kernel, boundary='symm', mode='same')  # 空间模糊化，借用邻近像素点信息

        # 更新背景
        self.back = fuzzy_foreground * self.back + (1 - fuzzy_foreground) * ((1 - lr_matrix) * self.back + figure * lr_matrix)
        # 更新阈值
        deviation = figure - pre_figure
        new_lr_matrix = (1 - fuzzy_foreground) * lr_matrix * 0.1
        self.threshold = (1 - new_lr_matrix) * self.threshold + new_lr_matrix * 2.5 * np.abs(deviation)

        # 背景移除结果
        new_figure = unfuzzy_foreground * figure

        if mode == 'AGBR-PM_back':
            return self.back
        elif mode == 'AGBR-PM_threshold':
            return self.threshold
        elif mode == 'AGBR-PM':
            return new_figure.astype('uint8')
        elif mode == 'AGBR-PM_prior':
            temp = np.zeros_like(self.prior_list[0])
            for prior in self.prior_list:
                temp += prior
            return temp
        elif mode == 'AGBR-PM_density':
            return integrated_dm
        else:
            return new_figure.astype('uint8')


    def check_switch(self,figure,order):
        # 检验开关状态
        prior = self.prior_list[order] > 0
        threshold, _ = cv.threshold(figure.astype('uint8'), 0, 255, cv.THRESH_TOZERO + cv.THRESH_TRIANGLE)  # 计算阈值
        figure = figure > threshold
        # 热源开关状态检测，如果热图在先验位置上有20%面积的区域大于阈值，那么认为此处的先验热源的状态为“开”
        prior_area = prior.sum()
        hot_area = figure.sum()
        if hot_area / prior_area >= 0.2:
            switch = True
        else :
            switch = False
        return switch

    def check_cross(self, figure, order):
        # 交叠检测
        current_pic = figure > 0
        temp = self.densityMatrix[order] * current_pic
        if temp.sum() > self.densityMatrix[order].sum() * 0.8:
            is_cross = True
        else:
            is_cross = False
        return is_cross

    @staticmethod
    def complement(foreground, figure, gain=1):
        # 前景补足
        row = figure.shape[0]
        col = figure.shape[1]
        segment = mp.segment(figure.astype('uint8'),'triangle')
        x_mean, y_mean, x_var, y_var, _ = mp.feature(foreground * figure, gain)
        for i in range(row):
            for j in range(col):
                if abs(i - y_mean) <= y_var and abs(j - x_mean) <= x_var:
                    if segment[i, j] and not foreground[i, j]:
                        foreground[i, j] = 1
        return foreground

    def FBR(self, old_figure):
        # 模糊背景移除
        figure = old_figure - old_figure.min()

        max_sigma = 1.5 / 0.2
        min_sigma = 0.5 / 0.2
        sigma = self.cal_threshold(self.queue_of_rb[-5:])
        sigma = sigma * sigma
        miu_sigma = (sigma < min_sigma) * 1
        miu_sigma = miu_sigma + (sigma > max_sigma) * 0
        miu_sigma = miu_sigma + (min_sigma < sigma) * (sigma < max_sigma) * (max_sigma - sigma) / (
                    max_sigma - min_sigma)

        max_t = 3 / 0.2
        min_t = 0.5 / 0.2
        miu_t = (figure < min_t) * 0
        miu_t = miu_t + (figure > max_t) * 1
        miu_t = miu_t + (min_sigma < sigma) * (sigma < max_sigma) * (sigma - min_t) / (max_t - min_t)

        self.miu_n = miu_t * (1 - miu_sigma) + self.miu_n * miu_sigma
        foreground = self.miu_n > 0.6
        new_figure = (figure * foreground).astype('uint8')
        for i in range(13, 18):
            for j in range(20, 25):
                new_figure[i, j], new_figure[i, j + 6] = new_figure[i, j + 6], new_figure[i, j]  # 这一部分是干啥的？
        return new_figure

    def GBR(self, old_figure):
        # 模糊背景移除
        figure = old_figure.astype('int32')

        diff_figure = np.abs(figure - self.back)  # 获取当前图像和背景的差值
        foreground = diff_figure > 2.5 * self.threshold

        lr = 0.001
        self.back = foreground * self.back + (1 - foreground) * ((1 - lr) * self.back + lr * figure)
        new_threshold = (1 - foreground) * ((1 - lr) * self.threshold ** 2 + lr * diff_figure ** 2)
        self.threshold = foreground * self.threshold + new_threshold ** 0.5

        new_figure = figure * foreground
        return new_figure.astype('uint8')

    def diff(self, figure, diff_type='3'):
        # 计算差分
        self.put_figure_queue(self.queue_of_diff, figure, 3)
        if len(self.queue_of_diff) < 3:  # 如果差分缓冲区长度小于3，则使用当前数据填满
            return self.diff(figure, diff_type=diff_type)
        else:
            diff1 = self.queue_of_diff[-1].astype('int32') - self.queue_of_diff[-2].astype('int32')
            diff2 = self.queue_of_diff[-2].astype('int32') - self.queue_of_diff[-3].astype('int32')
            diff1 = np.abs(diff1)
            diff2 = np.abs(diff2)
            diff = diff1 * diff2
            diff = (diff <= 255) * diff + (diff > 255) * 255
        if diff_type == '3':
            return diff.astype('uint8')
        if diff_type == '2':
            return diff1.astype('uint8')
        return diff.astype('uint8')

    def MOG(self, figure):
        # 利用OpenCV的背景移除技术
        # 有两种方法，分别是MOG和KNN，KNN更加平滑
        # 设置history参数，该参数越大，背景移除的速度就越慢
        if not self.mog_is_created:
            self.mog=cv.createBackgroundSubtractorMOG2(history=500)
            # self.mog = cv.createBackgroundSubtractorKNN(history=500)
            self.mog_is_created = True
            return figure
        else:
            return (self.mog.apply(figure) * figure).astype('uint8')



