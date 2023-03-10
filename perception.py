import numpy as np
from sequence_process import SequenceProcess as sp
import matrix_process as mp
from velocity import Velocity
import matplotlib.pyplot as plt

class Perception:
    def __init__(self):
        self.static_actions = ['坐', '站', '蹲', '躺', '弯腰']
        self.high_level_actions = ['工作', '检查设备', '接水', '外出', '归来']
        # 办公室内陈列的物品的几何中心坐标
        self.objects = {'computer': (250, 490), 'equipment': (295, 365), 'bucket': (250, 625)}
        #
        # self.axes.add_patch(patches.Rectangle((230, 460), 40, 60, edgecolor='k', facecolor='red'))
        # self.axes.add_patch(patches.Rectangle((280, 350), 30, 30, edgecolor='k', facecolor='yellow'))
        # self.axes.add_patch(patches.Circle((250, 625), 30, edgecolor='k', facecolor='blue'))

        # 敏感半径，在此范围内则认为在与其交互
        self.sensitive_radius = 100

        self.TFD_buffer_len = 5
        self.buffer_len = 10
        self.top0_buffer = []
        self.top1_buffer = []
        self.side0_buffer = []
        self.side1_buffer = []

        self.side0_TFDS_list = []
        self.side1_TFDS_list = []

        self.is_static_model_loaded = False
        self.static_model = None

        self.lc = Velocity()

        #  离线数据
        self.data = {}
        self.is_data_loaded = False
        #  数据处理对象
        self.sp_side0 = sp()
        self.sp_side1 = sp()
        self.sp_top0 = sp()
        self.sp_top1 = sp()

        self.MTFDS_threshold = 3000
        self.human_detection_threshold = 10
        self.velocity_threshold = 3
        self.S = 0
        self.timer = 0
        self.last_action = 'None'
        self.current_action = 'None'
        self.high_level_action = 'None'
        self.location_x = -1
        self.location_y = -1
        self.last_location_x = -1
        self.last_location_y = -1

    def clear(self):
        self.top0_buffer = []
        self.top1_buffer = []
        self.side0_buffer = []
        self.side1_buffer = []
        self.side0_TFDS_list = []
        self.side1_TFDS_list = []
        self.sp_side0.clear()
        self.sp_side1.clear()
        self.sp_top0.clear()
        self.sp_top1.clear()
        self.S = 0
        self.timer = 0
        self.last_action = 'None'
        self.current_action = 'None'
        self.high_level_action = 'None'
        self.location_x = -1
        self.location_y = -1
        self.last_location_x = -1
        self.last_location_y = -1

    def load_model(self, model_path):
        from mymodel.models import load_model
        self.static_model = load_model(model_path)
        self.is_static_model_loaded = True

    def percept(self, side0, side1, top0, top1):
        side0 = mp.quantify(side0, _min=0, resolution=0.2)
        side0 = self.sp_side0.temporal_filter(side0, 'Gauss', win_size=3)
        side0 = self.sp_side0.remove_background(side0, 'BS-SEG', buffer_len=5)
        side0 = mp.max_connected_component(side0)

        side1 = mp.quantify(side1, _min=0, resolution=0.2)
        side1 = self.sp_side1.temporal_filter(side1, 'Gauss', win_size=3)
        side1 = self.sp_side1.remove_background(side1, 'BS-SEG', buffer_len=5)
        side1 = mp.max_connected_component(side1)

        top0 = mp.quantify(top0, _min=0, resolution=0.2)
        top0 = self.sp_top0.temporal_filter(top0, 'Gauss', win_size=3)
        top0 = self.sp_top0.remove_background(top0, 'BS-SEG', buffer_len=5)
        top0 = mp.max_connected_component(top0)

        top1 = mp.quantify(top1, _min=0, resolution=0.2)
        top1 = self.sp_top1.temporal_filter(top1, 'Gauss', win_size=3)
        top1 = self.sp_top1.remove_background(top1, 'BS-SEG', buffer_len=5)
        top1 = mp.max_connected_component(top1)

        exist = self.human_detection(top0, top1)  # 探测人体目标是否位于检测范围内
        # print(exist)

        if exist == 'None':
            # 如果不在检测范围内，则行为定义为None
            self.side0_buffer.clear()
            self.side1_buffer.clear()
            self.top0_buffer.clear()
            self.top1_buffer.clear()
            self.current_action = 'None'

        elif exist == 'top0':
            self.location_x, self.location_y = self.human_location(top0, Height=300, deploy_x=230, deploy_y=420, horizon=True)
            self.put_buffer(side0, side1, top0, top1)
            if len(self.side0_buffer) < self.buffer_len:
                self.current_action = 'waiting'
            else:
                velocity = self.lc.velocity(top0)
                if velocity >= self.velocity_threshold:
                    self.current_action = '移动'
                else:
                    if (side0 > 0).sum() < self.human_detection_threshold*2 and (side1 > 0).sum() < self.human_detection_threshold*2:
                        self.current_action = '移动'
                    else:
                        if (side0 > 0).sum() > (side1 > 0).sum():
                            # print('side0')
                            judge_result = self.static_dynamic_judge('side0')
                        else:
                            # print('side1')
                            judge_result = self.static_dynamic_judge('side1')
                        if judge_result == 'static':
                            self.S += 1
                            if self.S == 4:
                                result = self.static_action_percept(self.side0_buffer[-1], self.side1_buffer[-1], top0, top1 )
                                self.current_action = result
                                self.S = 0
                        elif judge_result == 'dynamic':
                            self.S = 0
                        elif judge_result == 'waiting':
                            self.S = 0
                            self.current_action = 'waiting'

        elif exist == 'top1':
            self.location_x, self.location_y = self.human_location(top1, Height=300, deploy_x=350, deploy_y=610, horizon=False)
            self.put_buffer(side0, side1, top0, top1)
            if len(self.side0_buffer) < self.buffer_len:
                self.current_action = 'waiting'
            else:
                velocity = self.lc.velocity(top1)
                if velocity >= self.velocity_threshold:
                    self.current_action = '移动'
                else:
                    if (side0 > 0).sum() < self.human_detection_threshold*2 and (side1 > 0).sum() < self.human_detection_threshold*2:
                        self.current_action = '移动'
                    else:
                        if (side0 > 0).sum() > (side1 > 0).sum():
                            # print('side0')
                            judge_result = self.static_dynamic_judge('side0')
                        else:
                            # print('side1')
                            judge_result = self.static_dynamic_judge('side1')
                        if judge_result == 'static':
                            self.S += 1
                            if self.S == 4:
                                result = self.static_action_percept(self.side0_buffer[-1], self.side1_buffer[-1], top0,
                                                                    top1)
                                self.current_action = result
                                self.S = 0
                        elif judge_result == 'dynamic':
                            self.S = 0
                        elif judge_result == 'waiting':
                            self.S = 0
                            self.current_action = 'waiting'
        if self.timer == 0:
            self.high_level_action = self.action_understanding(self.last_action,self.current_action,self.location_x,self.location_y)
            if self.high_level_action == '外出归来' or self.high_level_action == '离开':
                self.timer = 1
        elif self.timer != 0:
            self.timer += 1
            if self.timer == 16:
                self.timer = 0
        self.last_action = self.current_action
        return self.current_action, self.high_level_action, self.location_x, self.location_y

    def static_action_percept(self, side0, side1, top0, top1):
        side0 = side0 / 255.
        side0 = np.expand_dims(side0, axis=-1)
        side0 = np.expand_dims(side0, axis=0)

        prediction0 = self.static_model.predict(side0)
        result0 = self.static_actions[np.argmax(prediction0[0])]

        side1 = side1/255.
        side1 = np.expand_dims(side1, axis=-1)
        side1 = np.expand_dims(side1, axis=0)
        prediction1 = self.static_model.predict(side1)
        result1 = self.static_actions[np.argmax(prediction1[0])]

        # 如果两个视角预测的不同
        if result0 == result1:
            return result0
        else:
            if (side0 > 0).sum() > (side1 > 0).sum():
                # print('result0：'+result0)
                return result0
            else:
                # print('result1：'+result1)
                return result1

    def human_detection(self,top0, top1):
        #  用来检测区域内是否存在人体目标，返回测到的传感器代号
        if (top0 > 0).sum() > self.human_detection_threshold:
            # print('top0')
            return 'top0'
        elif (top1 > 0).sum() > self.human_detection_threshold:
            # print('top1')
            return 'top1'
        else:
            return 'None'

    def human_location(self, top_data, Height, deploy_x, deploy_y, horizon):
        #  用来进行人体目标定位，返回其在实际场景中的坐标
        # 输入：顶视数据、传感器相对于地面的布置高度及相对于场景原点的横纵坐标
        if horizon:
            # 如果朝南（窗户）布置
            # 探测范围
            L = 2*(Height-170)/0.7
            H = 2*(Height-170)/1.3
            # 注意这种布置方向传感器的坐标方向和场景的坐标方向正交
            sensor_x, sensor_y = mp.center(top_data)  # 传感器坐标下的人体目标重心，单位为格
            # 转换为相对于传感器视场的实际距离, 坐标原点为传感器的布置位置，单位为厘米
            real_x = (sensor_x/32)*L
            real_y = (sensor_y/24)*H
            #传感器方向改变
            temp_x = real_x
            temp_y = real_y
            real_x = temp_y
            real_y = L-temp_x
            # 传感器原点坐标，根据布置位置计算
            original_x = deploy_x-(H/2)
            original_y = deploy_y-(L/2)
            # 在场景中的实际坐标
            location_x = real_x + original_x
            location_y = real_y + original_y
            return location_x, location_y
        else:
            # 如果朝东（桌子）布置
            L = 2*(Height-170)/0.7
            H = 2*(Height-170)/1.3
            sensor_x, sensor_y = mp.center(top_data)  # 传感器坐标下的人体目标重心，单位为格
            # 转换为相对于传感器视场的实际距离, 坐标原点为传感器的布置位置，单位为厘米
            # sensor_y = 24-sensor_y
            real_x = (sensor_x/32)*L
            real_y = (sensor_y/24)*H
            # 传感器原点坐标，根据布置位置计算
            original_x = deploy_x-(L/2)
            original_y = deploy_y-(H/2)
            # 在场景中的实际坐标
            location_x = real_x + original_x
            location_y = real_y + original_y
            return location_x, location_y

    def action_understanding(self,last_action, current_action, location_x, location_y):
        # 行为理解，输入为上一个行为，当前行为，坐标x，坐标y
        distance_dict = self.calculate_distance(location_x, location_y)
        if current_action == '坐':
            if distance_dict['computer'] <= self.sensitive_radius:
                return '坐着工作'
            else:
                return current_action
        elif current_action == '弯腰':
            if distance_dict['equipment'] <= self.sensitive_radius:
                return '检查设备'
            elif distance_dict['bucket'] <= self.sensitive_radius:
                return '接水'
            else:
                return current_action
        elif current_action == 'None':
            if last_action == '移动':
                return '离开'
            else:
                return current_action
        elif current_action == '移动':
            if last_action == 'waiting':
                return '外出归来'
            else:
                return current_action
        else:
            return current_action

    def calculate_distance(self, location_x, location_y):
        distance_dict ={}
        for key, value in self.objects.items():
            delta_x = (location_x - value[0])
            delta_y = (location_y - value[1])
            distance = pow((delta_x*delta_x)+(delta_y*delta_y), 0.5)
            distance_dict[key] = distance
        return distance_dict

    def put_buffer(self, side0, side1, top0, top1):
        self.side0_buffer.append(side0)
        self.side1_buffer.append(side1)
        self.top0_buffer.append(top0)
        self.top1_buffer.append(top1)
        if len(self.side0_buffer) > self.buffer_len:
            del self.side0_buffer[0]
            del self.side1_buffer[0]
            del self.top0_buffer[0]
            del self.top1_buffer[0]
            side0_TFDS = (abs(self.side0_buffer[-1] - self.side0_buffer[-2]) * abs(
                self.side0_buffer[-2] - self.side0_buffer[-3])).sum()
            side1_TFDS = (abs(self.side1_buffer[-1] - self.side1_buffer[-2]) * abs(
                self.side1_buffer[-2] - self.side1_buffer[-3])).sum()
            self.side0_TFDS_list.append(side0_TFDS)
            self.side1_TFDS_list.append(side1_TFDS)
            if len(self.side0_TFDS_list) > self.TFD_buffer_len:
                del self.side0_TFDS_list[0]
                del self.side1_TFDS_list[0]

    def static_dynamic_judge(self, string):
        #  用来进行动态/静态行为判定
        if len(self.side0_TFDS_list) < self.TFD_buffer_len:
            # 如果缓存区没满，则继续等待，不返回判定结果
            return 'waiting'
        else:
            if string == 'side0':
                MTFDS0 = np.array(self.side0_TFDS_list).mean()
                # print('MTFDS0:'+str(MTFDS0))
                if MTFDS0 < self.MTFDS_threshold :
                    return 'static'
                else:
                    return 'dynamic'
            elif string == 'side1':
                MTFDS1 = np.array(self.side1_TFDS_list).mean()
                # print('MTFDS1:' + str(MTFDS1))
            if MTFDS1 < self.MTFDS_threshold:
                return 'static'
            else:
                return 'dynamic'

    def set_offline_data(self, data):
        # data为字典
        self.data = data.copy()
        self.is_data_loaded = True

    def forward(self, step):
        # 在载入了离线数据的情况下进行感知
        if not self.is_static_model_loaded:
            return 'The model is not loaded', 'The model is not loaded', self.location_x, self.location_y
        if step == 0:
            self.clear()
        if self.is_data_loaded:
            return self.percept(side0=self.data['side0'][step], side1=self.data['side1'][step], top0=self.data['top0'][step], top1=self.data['top1'][step])
        else:
            return 'The data is not loaded.', 'The model is not loaded', self.location_x, self.location_y

    def online_forward(self, data):
        # 在线模式的行为感知
        if not self.is_static_model_loaded:
            return 'The model is not loaded', 'The model is not loaded',self.location_x, self.location_y
        else:
            return self.percept(side0=data['side0'], side1=data['side1'], top0=data['top0'], top1=data['top1'])
