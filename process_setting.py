from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QComboBox
from PyQt5.QtWidgets import QLabel, QSpinBox, QDoubleSpinBox, QCheckBox, QToolButton, QMessageBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import pyqtSignal
import file_process as fp
from functools import partial


class ProcessSetting(QWidget):
    """########################################
    功能：
        1、生成不同的数据处理方案。数据处理方案包括处理流程，以及流程中使用的参数
        2、创建各个框体并设置框体的值
        3、框体值改变时将其保存到xml文件中，并且发出参数改变信号
        4、返回参数
    ########################################"""
    settingChangeSignal = pyqtSignal(str)  # 参数改变信号

    def __init__(self, num):
        super().__init__()
        self.num = num  # 设置每张图对应的参数
        self.setWindowTitle('参数设置')
        self.processOriginalPara = fp.load_para('processPara.xml')  # 载入初始processPara
        self.processPara = self.divide_para(self.processOriginalPara)  # 改变格式

        self.menu = fp.load_para('menu.xml')  # 载入menu
        self.create_boxes(self.num)  # 创建框体
        self.boxes_init()  # 框体初始化
        self.hide_none()
        self.build_connection(self.num)  # 框体信号连接

    def divide_para(self, processOriginalPara):
        # 划分参数,每一幅图一套处理参数
        process_para = {}
        for key, value in processOriginalPara.items():
            if key[0:4] == 'flow':
                continue
            else:
                if type(value) == list:
                    process_para[key] = value[self.num]
                else:
                    process_para[key] = value
        process_para['flow'] = processOriginalPara['flow' + str(self.num)]
        return process_para

    def create_boxes(self, num):
        """########################################
        创建框体
            1、这里创建了全局变量：
                self.processComboBoxes，self.processSpinBoxes
                self.flowList，self.mainLayout
                self.layout1~4
            2、在这个框体中，包含了4个QHBoxLayout
        #######################################"""
        self.processComboBoxes = {'temporalFilter': [], 'backgroundRemove': [], 'filter': [],
                                  'segment': []}  # ComboBox字典
        self.processSpinBoxes = {'temporalFilterBuffer': [], 'RBbuffer': [], 'interPoints': [], 'segmentPara': [],
                                 'filterPara': []}  # SpinBoxes字典
        self.flowList = []  # 流程框，多个ComboBox
        self.delete_button_list = []
        self.add_button_list = []
        self.arrow_label_list = []
        self.function_layout_list = []

        # 数据源
        self.layout1 = QHBoxLayout()
        self.layout1.addWidget(QLabel('数据处理方案' + str(num) + '  视角'))

        self.selectDataName = QComboBox()
        self.selectDataName.addItems(self.menu['dataNameList'])
        self.selectDataName.setFixedSize(300, 40)
        self.processComboBoxes['dataName'] = [self.selectDataName]

        self.layout1.addWidget(self.selectDataName)
        self.layout1.addStretch()

        self.selectPortName = QComboBox()
        self.selectPortName.addItems(self.menu['portNameList'])
        self.selectPortName.setFixedSize(300, 40)
        self.processComboBoxes['portName'] = [self.selectPortName]
        self.layout1.addWidget(self.selectPortName)

        self.portLabel = QLabel('端口号' + str(self.menu['portList'][num]))
        self.layout1.addWidget(self.portLabel)

        # 处理流程
        self.layout2 = QVBoxLayout()
        self.layout2.addWidget(QLabel('流程：'))
        self.layout2.addWidget(QLabel('input'))
        self.layout2.addWidget(QLabel('   ↓'))
        # 多个流程选择框以及中间的>>符号以及选值框的初始化

        for i in range(10):
            self.function_layout_list.append([])
            box = QComboBox()
            box.addItems(self.menu['processList'])
            self.flowList.append(box)

            delete_button = QToolButton()
            delete_button.setText('-')
            self.delete_button_list.append(delete_button)

            add_button = QToolButton()
            add_button.setText('+')
            self.add_button_list.append(add_button)

            label = QLabel('   ↓')
            self.arrow_label_list.append(label)
            # 时域滤波

            temporal_filter_layout = QHBoxLayout()
            temp = QLabel('滤波方法')
            temporal_filter_layout.addWidget(temp)
            self.function_layout_list[i].append(temp)

            temp = QComboBox()  # 滤波方法
            temp.addItems(self.menu['temporalFilterList'])
            temp.setFixedSize(300, 40)
            self.function_layout_list[i].append(temp)
            self.processComboBoxes['temporalFilter'].append(temp)
            temporal_filter_layout.addWidget(temp)

            temp = QLabel('窗口长度')
            self.function_layout_list[i].append(temp)
            temporal_filter_layout.addWidget(temp)

            temp = QSpinBox()  # 窗口长度
            temp.setRange(1, 11)
            temp.setSingleStep(2)
            temp.setFixedSize(300, 40)
            self.function_layout_list[i].append(temp)
            self.processSpinBoxes['temporalFilterBuffer'].append(temp)
            temporal_filter_layout.addWidget(temp)
            temporal_filter_layout.addStretch()

            # 背景移除
            background_remove_layout = QHBoxLayout()
            temp = QLabel('移除方法')
            self.function_layout_list[i].append(temp)
            background_remove_layout.addWidget(temp)

            temp = QComboBox()  # 选项框，选择背景移除算法
            temp.addItems(self.menu['backgroundRemoveList'])
            temp.setFixedSize(300, 40)
            self.function_layout_list[i].append(temp)
            self.processComboBoxes['backgroundRemove'].append(temp)
            background_remove_layout.addWidget(temp)

            temp = QLabel('缓冲长度')
            self.function_layout_list[i].append(temp)
            background_remove_layout.addWidget(temp)

            temp = QSpinBox()  # 背景移除缓存区
            temp.setRange(5, 60)
            temp.setSingleStep(5)
            temp.setFixedSize(300, 40)
            self.function_layout_list[i].append(temp)
            self.processSpinBoxes['RBbuffer'].append(temp)
            background_remove_layout.addWidget(temp)
            background_remove_layout.addStretch()

            # 插值
            interpolate_layout = QHBoxLayout()
            temp = QLabel('插值点数')
            self.function_layout_list[i].append(temp)
            interpolate_layout.addWidget(temp)
            temp = QSpinBox()  # 选值框，设定插值点数
            temp.setRange(0, 15)  # 插值点数范围
            temp.setFixedSize(300, 40)
            self.function_layout_list[i].append(temp)
            self.processSpinBoxes['interPoints'].append(temp)
            interpolate_layout.addWidget(temp)
            interpolate_layout.addStretch()

            # 空间滤波
            spatial_filter_layout = QHBoxLayout()
            temp = QLabel('滤波方法')
            self.function_layout_list[i].append(temp)
            spatial_filter_layout.addWidget(temp)
            temp = QComboBox()
            temp.addItems(self.menu['filterList'])  # 根据系统设置文件添加选项
            temp.setFixedSize(300, 40)
            self.function_layout_list[i].append(temp)
            self.processComboBoxes['filter'].append(temp)
            spatial_filter_layout.addWidget(temp)
            temp = QLabel('窗口尺寸')
            self.function_layout_list[i].append(temp)
            spatial_filter_layout.addWidget(temp)
            temp = QSpinBox()  # 选值框，设定滤波参数
            temp.setRange(1, 15)
            temp.setSingleStep(2)
            temp.setFixedSize(300, 40)
            self.function_layout_list[i].append(temp)
            self.processSpinBoxes['filterPara'].append(temp)
            spatial_filter_layout.addWidget(temp)
            spatial_filter_layout.addStretch()

            # 阈值分割
            segment_layout = QHBoxLayout()
            temp = QLabel('分割方法')
            self.function_layout_list[i].append(temp)
            segment_layout.addWidget(temp)
            temp = QComboBox()
            temp.addItems(self.menu['segmentList'])  # 根据系统设置文件添加选项
            temp.setFixedSize(300, 40)
            self.function_layout_list[i].append(temp)
            self.processComboBoxes['segment'].append(temp)
            segment_layout.addWidget(temp)
            temp = QLabel('分割参数')
            self.function_layout_list[i].append(temp)
            segment_layout.addWidget(temp)
            temp = QDoubleSpinBox()  # 选值框，手动设定阈值
            temp.setRange(-20, 256)
            temp.setFixedSize(300, 40)
            self.function_layout_list[i].append(temp)
            self.processSpinBoxes['segmentPara'].append(temp)
            segment_layout.addWidget(temp)
            segment_layout.addStretch()

            function_layout = QVBoxLayout()
            function_layout.addLayout(temporal_filter_layout)
            function_layout.addLayout(background_remove_layout)
            function_layout.addLayout(interpolate_layout)
            function_layout.addLayout(spatial_filter_layout)
            function_layout.addLayout(segment_layout)

            tempLayout = QHBoxLayout()
            tempLayout.addWidget(box)
            tempLayout.addWidget(delete_button)
            tempLayout.addWidget(add_button)
            tempLayout.addLayout(function_layout)
            self.layout2.addLayout(tempLayout)
            self.layout2.addLayout(tempLayout)
            self.layout2.addWidget(label)

        self.layout2.addWidget(QLabel('output'))

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addLayout(self.layout1)
        self.mainLayout.addLayout(self.layout2)
        self.setLayout(self.mainLayout)
        self.setFont(QFont("Microsoft Yahei", 12, QFont.Normal))  # 设定字体及大小

    def boxes_init(self):
        # 设置各个框体的值
        # flow框
        for box, processName in zip(self.flowList, self.processPara['flow']):
            index = self.menu['processList'].index(processName)
            box.setCurrentIndex(index)

        for i, box in enumerate(self.flowList):
            self.show_function(i, box.currentText())

        # 选值框
        for name, list in self.processComboBoxes.items():
            for box in list:
                index = self.menu[name + 'List'].index(self.processPara[name])
                box.setCurrentIndex(index)
        # 输入框
        for name, list in self.processSpinBoxes.items():
            for box in list:
                box.setValue(self.processPara[name])

    def build_connection(self, num):
        # 框体信号连接

        # flow连接
        for i, box in enumerate(self.flowList):
            box.activated[str].connect(lambda value, _i=i: self.save_flow(_i, value))
            box.activated[str].connect(lambda value, _i=i: self.show_function(_i, value))

        # 减去流程按钮
        for i, button in enumerate(self.delete_button_list):
            button.clicked.connect(partial(self.delete_flow, i))
            button.clicked.connect(lambda _i=i: self.show_function(_i, 'None'))

        # 增加流程按钮
        for i, button in enumerate(self.add_button_list):
            button.clicked.connect(partial(self.add_flow, i))

        # comboBox连接
        for name, list in self.processComboBoxes.items():
            for box in list:
                box.activated[str].connect(lambda _value, _name=name, _num=num: self.save_box(_name, _value, _num,
                                                                                              'processPara.xml'))
                box.activated[str].connect(self.boxes_init)
        # spinBox连接
        for name, list in self.processSpinBoxes.items():
            for box in list:
                box.valueChanged.connect(lambda value, _name=name, _num=num: self.save_box(_name, value, _num,
                                                                                           'processPara.xml'))
                box.valueChanged.connect(self.boxes_init)

    def hide_none(self):
        # 重新排序，隐藏None流程，并把排序后的流程存入xml文件
        not_none_list = []
        none_list = []
        for i, box in enumerate(self.flowList):
            if box.currentIndex() == 0:
                self.flowList[i].hide()
                self.delete_button_list[i].hide()
                self.arrow_label_list[i].hide()
                self.add_button_list[i].hide()
                for j in range(18):
                    self.function_layout_list[i][j].hide()
                none_list.append(box)
            else:
                not_none_list.append(box)

        not_none_list.extend(none_list)
        self.flowList = not_none_list
        for i, box in enumerate(self.flowList):
            self.processPara['flow'][i] = self.flowList[i].currentText()
            fp.save_para('processPara.xml', 'flow' + str(self.num), self.flowList[i].currentText(), i)

        if len(none_list) == 10:
            self.flowList[0].show()
            self.delete_button_list[0].show()
            self.arrow_label_list[0].show()
            self.add_button_list[0].show()

    def add_flow(self, i):
        full_flag = True
        for j, box in enumerate(self.flowList):
            if box.isHidden():
                index = j
                full_flag = False
                break
        if full_flag:
            QMessageBox.warning(self, "警告", "最多设置10个处理流程")
        else:
            self.flowList[index].show()
            self.flowList[index].setCurrentIndex(0)
            self.delete_button_list[index].show()
            self.add_button_list[index].show()
            self.arrow_label_list[index].show()
            front_list = self.processPara['flow'][0:i + 1]
            back_list = self.processPara['flow'][i + 1:]
            back_list.insert(0, 'None')
            back_list.pop(-1)
            front_list.extend(back_list)
            self.processPara['flow'] = front_list
            for box, processName in zip(self.flowList, self.processPara['flow']):
                index = self.menu['processList'].index(processName)
                box.setCurrentIndex(index)
            for k, box in enumerate(self.flowList):
                self.save_flow(k, box.currentText())
        self.show()

    def delete_flow(self, i):
        front_list = self.processPara['flow'][0:i]
        back_list = self.processPara['flow'][i + 1:]
        back_list.append('None')
        front_list.extend(back_list)
        self.processPara['flow'] = front_list
        for box, processName in zip(self.flowList, self.processPara['flow']):
            index = self.menu['processList'].index(processName)
            box.setCurrentIndex(index)
        for k, box in enumerate(self.flowList):
            self.save_flow(k, box.currentText())

        for j, box in enumerate(self.flowList):
            if box.isHidden():
                index = j
                break
        if index == 1:
            QMessageBox.warning(self, "警告", "请保留一个处理流程，查看原始数据请设置为None")
        else:
            self.flowList[index - 1].hide()
            self.delete_button_list[index - 1].hide()
            self.arrow_label_list[index - 1].hide()
            self.add_button_list[index - 1].hide()
            for j in range(18):
                self.function_layout_list[index - 1][j].hide()

    def show_function(self, i, flow):
        for j in range(18):
            self.function_layout_list[i][j].hide()
        if flow == self.menu['processList'][2]:
            for j in range(0, 4):
                self.function_layout_list[i][j].show()
        elif flow == self.menu['processList'][4]:
            for j in range(4, 8):
                self.function_layout_list[i][j].show()
        elif flow == self.menu['processList'][5]:
            for j in range(8, 10):
                self.function_layout_list[i][j].show()
        elif flow == self.menu['processList'][6]:
            for j in range(10, 14):
                self.function_layout_list[i][j].show()
        elif flow == self.menu['processList'][7]:
            for j in range(14, 18):
                self.function_layout_list[i][j].show()
        else:
            pass

    def save_flow(self, i, value):
        """########################################
        保存流程至xml
        输入：
            1、流程编号i（整数）
            2、值value（字符串）
        输出：
            None
        备注：
        #######################################"""
        self.processPara['flow'][i] = value
        fp.save_para('processPara.xml', 'flow' + str(self.num), value, i)
        signal = '方案' + str(self.num) + ' 第' + str(i) + '流程修改为' + value
        self.settingChangeSignal.emit(signal)

    def save_box(self, name, value, num, xml_name):
        """########################################
        保存参数至xml
        输入：
            1、标签名name（字符串）
            2、值value（数字或者字符串）,
            3、位置num（整数）
            4、xml文件名xmlName（字符串）
        输出：
            None
        备注：
            1、num并不一定等于self.num，如果有某个参数是共有的，num将小于0
        #######################################"""
        self.processPara[name] = value
        fp.save_para(xml_name, name, value, num)
        signal = '方案' + str(num) + ' 的' + name + '修改为' + str(value)
        self.settingChangeSignal.emit(signal)
