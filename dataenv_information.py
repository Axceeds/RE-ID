import os

from PyQt5.QtWidgets import QWidget, QToolButton, QHBoxLayout, QVBoxLayout, QFormLayout, \
    QPushButton, QLabel, QSpinBox, QGridLayout, QProgressBar, QMessageBox, QProgressDialog, QLineEdit
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal
from button import Button
import file_process as fp

class EnvironmentInformation(QWidget):
    def __init__(self):
        super(EnvironmentInformation, self).__init__()

        self.env_information = {'name': [], 'time': [], 'temperature': [], 'weather': [], 'people_height': [], \
                                'sensor_distance': [], 'sensor_height': []}
        self.str = []
        self.environmentLayout = QHBoxLayout()

        self.label = QLineEdit()
        self.labelLayout = QFormLayout()
        self.labelLayout.addRow('标签', self.label)
        self.name = QLineEdit()
        self.nameLayout = QFormLayout()
        self.nameLayout.addRow('姓名', self.name)
        self.time1 = QLineEdit()
        self.timeLayout = QFormLayout()
        self.timeLayout.addRow('时间', self.time1)
        self.clothes = QLineEdit()
        self.clothesLayout = QFormLayout()
        self.clothesLayout.addRow('衣服', self.clothes)
        self.weather = QLineEdit()
        self.weatherLayout = QFormLayout()
        self.weatherLayout.addRow('天气', self.weather)
        self.people_height = QLineEdit()
        self.people_heightLayout = QFormLayout()
        self.people_heightLayout.addRow('身高', self.people_height)
        self.sensor_distance = QLineEdit()
        self.sensor_distanceLayout = QFormLayout()
        self.sensor_distanceLayout.addRow('传感器距离', self.sensor_distance)
        self.sensor_height = QLineEdit()
        self.sensor_heightLayout = QFormLayout()
        self.sensor_heightLayout.addRow('传感器高度', self.sensor_height)
        self.temperature = QLineEdit()
        self.temperatureLayout = QFormLayout()
        self.temperatureLayout.addRow('室温', self.temperature)
        self.label.setFont(QFont('Arial', 10))
        self.name.setFont(QFont('Arial', 10))
        self.time1.setFont(QFont('Arial', 10))
        self.clothes.setFont(QFont('Arial', 10))
        self.weather.setFont(QFont('Arial', 10))
        self.people_height.setFont(QFont('Arial', 10))
        self.sensor_height.setFont(QFont('Arial', 10))
        self.sensor_distance.setFont(QFont('Arial', 10))
        self.temperature.setFont(QFont('Arial', 10))

        self.label.textChanged.connect(self.text_changed_label)
        self.name.textChanged.connect(self.text_changed_name)
        self.time1.textChanged.connect(self.text_changed_time)
        self.clothes.textChanged.connect(self.text_changed_clothes)
        self.weather.textChanged.connect(self.text_changed_weather)
        self.people_height.textChanged.connect(self.text_changed_people_height)
        self.sensor_height.textChanged.connect(self.text_changed_sensor_height)
        self.sensor_distance.textChanged.connect(self.text_changed_sensor_distance)
        self.temperature.textChanged.connect(self.text_changed_temperature)

        self.environmentLayout.addLayout(self.labelLayout)
        self.environmentLayout.addLayout(self.nameLayout)
        self.environmentLayout.addLayout(self.timeLayout)
        self.environmentLayout.addLayout(self.temperatureLayout)
        self.environmentLayout.addLayout(self.clothesLayout)
        self.environmentLayout.addLayout(self.weatherLayout)
        self.environmentLayout.addLayout(self.people_heightLayout)
        self.environmentLayout.addLayout(self.sensor_heightLayout)
        self.environmentLayout.addLayout(self.sensor_distanceLayout)
        self.setLayout(self.environmentLayout)
        self.label.setText('None')
        self.name.setText('None')
        self.time1.setText('None')
        self.clothes.setText('None')
        self.temperature.setText('None')
        self.weather.setText('None')
        self.people_height.setText('None')
        self.sensor_height.setText('None')
        self.sensor_distance.setText('None')

    def text_changed_label(self):
        if str(self.label.text()) != '':
            fp.save_para('env_information.xml', 'label', str(self.label.text()))

    def text_changed_clothes(self):
        if str(self.clothes.text()) != '':
            fp.save_para('env_information.xml', 'clothes', str(self.clothes.text()))

    def text_changed_name(self):
        if str(self.name.text()) != '':
            fp.save_para('env_information.xml', 'name', str(self.name.text()))

    def text_changed_time(self):
        if str(self.time1.text()) != '':
            fp.save_para('env_information.xml', 'time', str(self.time1.text()))

    def text_changed_weather(self):
        if str(self.weather.text()) != '':
            fp.save_para('env_information.xml', 'weather', str(self.weather.text()))

    def text_changed_people_height(self):
        if str(self.people_height.text()) != '':
            fp.save_para('env_information.xml', 'people_height', str(self.people_height.text()))

    def text_changed_sensor_height(self):
        if str(self.sensor_height.text()) != '':
            fp.save_para('env_information.xml', 'sensor_height', str(self.sensor_height.text()))

    def text_changed_sensor_distance(self):
        if str(self.sensor_distance.text()) != '':
            fp.save_para('env_information.xml', 'sensor_distance', str(self.sensor_distance.text()))

    def text_changed_temperature(self):
        if str(self.temperature.text()) != '':
            fp.save_para('env_information.xml', 'temperature', str(self.temperature.text()))

    def set_text(self, info):
        # self.label.textChanged.disconnect(self.text_changed_label)
        # self.name.textChanged.disconnect(self.text_changed_name)
        # print('yes')
        # self.time1.textChanged.disconnect(self.text_changed_time)
        # self.weather.textChanged.disconnect(self.text_changed_weather)
        # self.people_height.textChanged.disconnect(self.text_changed_people_height)
        # self.sensor_height.textChanged.disconnect(self.text_changed_sensor_height)
        # self.sensor_distance.textChanged.disconnect(self.text_changed_sensor_distance)
        # self.temperature.textChanged.disconnect(self.text_changed_temperature)
        for keys in info:
            if keys == 'name':
                self.name.setText(info[keys])
            elif keys == 'clothes':
                self.clothes.setText(info[keys])
            elif keys == 'weather':
                self.weather.setText(info[keys])
            elif keys == 'time':
                self.time1.setText(info[keys])
            elif keys == 'people_height':
                self.people_height.setText(info[keys])
            elif keys == 'sensor_height':
                self.sensor_height.setText(info[keys])
            elif keys == 'sensor_distance':
                self.sensor_distance.setText(info[keys])
            elif keys == 'temperature':
                self.temperature.setText(info[keys])
            elif keys == 'label':
                self.label.setText(str(info[keys]))
            else:
                pass
        self.label.textChanged.connect(self.text_changed_label)
        self.name.textChanged.connect(self.text_changed_name)
        self.time1.textChanged.connect(self.text_changed_time)
        self.weather.textChanged.connect(self.text_changed_weather)
        self.people_height.textChanged.connect(self.text_changed_people_height)
        self.sensor_height.textChanged.connect(self.text_changed_sensor_height)
        self.sensor_distance.textChanged.connect(self.text_changed_sensor_distance)
        self.temperature.textChanged.connect(self.text_changed_temperature)
        pass

