import numpy as np
import matrix_process as mp


class KalmanFilter:
    """########################################
    卡尔曼滤波器KalmanFilter
    功能：
        1、输入一系列坐标，进行卡尔曼滤波
    备注：
        1、Q代表对测量值的信任程度，R代表噪声的强度
        Q越大，滤波的效果就越弱
    ########################################"""

    def __init__(self):
        self.target_has_entered = False
        self.init_velocity = False
        self.sampling_time = 0.125

        self.F = np.array([[1, 0, self.sampling_time, 0],
                           [0, 1, 0, self.sampling_time],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        self.R = np.array([[0.1, 0, 0, 0],
                           [0, 0.1, 0, 0],
                           [0, 0, 0.1, 0],
                           [0, 0, 0, 0.1]])

        self.Q = np.array([[self.sampling_time**4/4, 0, self.sampling_time**3/2, 0],
                           [0, self.sampling_time**4/4, 0, self.sampling_time**3/2],
                           [self.sampling_time**3/2, 0, self.sampling_time**2, 0],
                           [0, self.sampling_time**3/2, 0, self.sampling_time**2]])

        self.x = np.zeros(4)
        self.P = np.zeros((4, 4))

    def clear(self):
        # 重置
        self.target_has_entered = False
        self.init_velocity = False
        self.x = np.zeros(4)
        self.P = np.zeros((4, 4))

    def velocity(self, p):
        """########################################
        测速
        输入：
            1、坐标p（通常为ndarray）
        输出：
            1、速度（数值）
        备注：
        ########################################"""
        if not self.target_has_entered:
            self.x[0] = p[0]
            self.x[1] = p[1]
            self.target_has_entered = True
            return 0, 0
        else:
            if not self.init_velocity:
                self.x[2] = (p[0] - self.x[0]) / self.sampling_time
                self.x[3] = (p[1] - self.x[1]) / self.sampling_time
                self.init_velocity = True
                return self.x[2], self.x[3]
            else:
                x_k = np.array([p[0], p[1],
                                (p[0] - self.x[0]) / self.sampling_time,
                                (p[1] - self.x[1]) / self.sampling_time]).T
                x_e_k_k1 = self.F.dot(self.x)
                z_k = self.H.dot(x_k)
                P_k_k1 = self.F.dot(self.P).dot(self.F.T) + self.Q
                G_k = P_k_k1.dot(self.H.T).dot(self.H.dot(P_k_k1).dot(self.H.T) + self.R)
                self.x = x_e_k_k1 + G_k.dot(z_k - self.H.dot(x_e_k_k1))
                self.P = (np.eye(4) - G_k.dot(self.H)).dot(P_k_k1)
                return self.x[2], self.x[3]


class Velocity:
    """########################################
    速度估算
    功能：
        1、根据输入数据，计算位置及速度
    备注：
    ########################################"""

    def __init__(self):
        self.target_has_entered = False
        self._velocity = 0
        self.x = 0
        self.y = 0
        self.sampling_time = 0.125
        self.filter_gain = 0.5  # 数值越大，低通效果越强，不能超过1
        self.Kalman_filter = KalmanFilter()

    def clear(self):
        self.target_has_entered = False
        self.Kalman_filter.clear()
        self._velocity = 0
        self.x = 0
        self.y = 0

    def velocity(self, data, v_type='1D'):
        """########################################
        测速
        输入：
            1、输入数据x（通常为ndarray）
        输出：
            1、速度self._velocity（数值）
        备注：
        ########################################"""
        features = mp.feature(data)
        x = features[0]
        y = features[1]

        # 卡尔曼滤波
        if x and y:
            if self.target_has_entered:
                self.x = x
                self.y = y
                v_x, v_y = self.Kalman_filter.velocity([self.x, self.y])
            else:
                self.target_has_entered = True
                self.x = x
                self.y = y
                v_x, v_y = self.Kalman_filter.velocity([self.x, self.y])
        else:
            if self.target_has_entered:
                v_x, v_y = self.Kalman_filter.velocity([self.x, self.y])
            else:
                v_x, v_y = 0, 0

        if v_type == '1D':
            return (v_x ** 2 + v_y ** 2) ** 0.5
        if v_type == '2D':
            return v_x, v_y

        # 低通滤波
        # if x and y:
        #     if self.target_has_entered:
        #         _velocity = ((x - self.x) ** 2 + (y - self.y) ** 2) ** 0.5 / self.sampling_time
        #         self.x = x
        #         self.y = y
        #     else:
        #         _velocity = 0
        #         self.x = x
        #         self.y = y
        #         self.target_has_entered = True
        # else:
        #     _velocity = 0
        # self._velocity = self._velocity * self.filter_gain + _velocity * (1 - self.filter_gain)
        # return self._velocity

    def max_velocity(self, data):
        """########################################
        输入一套数据，检测最大速度
        输入：
            1、数据x（通常为ndarray）
        输出：
            1、最大速度max_velocity（数值）
        备注：
        ########################################"""
        time_length = data.shape[0]
        _velocity = np.zeros(time_length)
        for i in range(time_length):
            _velocity[i] = self.velocity(data[i])
        return _velocity.max()

    def mean_velocity(self, data):
        """########################################
        输入一套数据，检测平均速度
        输入：
            1、数据x（通常为ndarray）
        输出：
            1、平均速度mean_velocity（数值）
        备注：
        ########################################"""
        time_length = data.shape[0]
        _velocity = np.zeros(time_length)
        for i in range(time_length):
            _velocity[i] = self.velocity(data[i])
        return _velocity.sum() / time_length

    def mean_2d_velocity(self, data):
        """########################################
        输入一套数据，检测平均速度
        输入：
            1、数据x（通常为ndarray）
        输出：
            1、平均速度mean_velocity（数值）
        备注：
            1、分为x方向和y方向
        ########################################"""
        time_length = data.shape[0]
        _velocity = np.zeros((time_length, 2))
        for i in range(time_length):
            v_x, v_y = self.velocity(data[i], v_type='2D')
            _velocity[i, 0] = v_x
            _velocity[i, 1] = v_y
        return _velocity[:, 0].sum() / time_length, _velocity[:, 1].sum() / time_length
