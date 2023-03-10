import os
import numpy as np
from PyQt5.QtXml import QDomDocument
from xml.dom.minidom import Document
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QFile, QIODevice, QTextStream, pyqtSignal
import dataenv_information

def load_para(name, tag='None'):
    # 读取参数xml
    file = QFile(os.getcwd() + '/system/' + name)
    if not file.open(QIODevice.ReadOnly):
        # 如果文件打开失败
        # QMessageBox.warning("警告", "起始帧必须与结束帧相同")
        # print('参数xml打开失败，检测文件' + name + '是否存在')
        return 0
    else:
        # 打开系统参数文件成功，读取
        para = {}
        doc = QDomDocument()
        doc.setContent(file)
        root = doc.documentElement()
        para_node = root.firstChild()
        while para_node.toElement().tagName() != 'end':
            key = para_node.toElement().tagName()
            value = para_node.toElement().text()
            if ' ' in value:
                value = value.split(' ')
                para[key] = []
                for x in value:
                    para[key].append(str2num(x))
            else:
                para[key] = str2num(value)
            para_node = para_node.nextSibling()
        file.close()
    if tag != 'None':
        return para[tag]
    else:
        return para


def str2num(string):
    # 将字符串转换为数字
    if string.isdigit():  # 纯数字将会转换为int
        return int(string)
    else:
        try:
            return float(string)  # 尝试转换为浮点数
        except ValueError:
            return string


def save_para(name, tag, value, num=0):
    # 修改参数xml
    file = QFile(os.getcwd() + '/system/' + name)
    if not file.open(QIODevice.ReadOnly):
        # 如果文件打开失败
        print('参数xml打开失败，检测文件' + name + '是否存在')
        return 0
    else:
        doc = QDomDocument()
        doc.setContent(file)
        file.close()
        root = doc.documentElement()
        node = root.firstChild()
        while node.toElement().tagName() != 'end':
            nodeTag = node.toElement().tagName()
            if nodeTag == tag:
                oldValue = node.toElement().text()
                if ' ' in oldValue:
                    # 如果参数有多个，说明不同传感器不同，需要分开处理
                    newValue = oldValue.split(' ')
                    if len(newValue) > num:
                        newValue[num] = str(value)
                    elif len(newValue) == num:
                        newValue.append(str(value))
                        print('参数' + nodeTag + '只有' + str(len(newValue)) + '个，num=' + str(num) + '超过边界一个，所以添加了一个')
                    else:
                        print('参数' + nodeTag + '只有' + str(len(newValue)) + '个，num=' + str(num) + '超过边界一个以上，没有进行操作')
                else:
                    # 如果参数只有一个，则不需要进行分开处理
                    newValue = [str(value)]

                newNode = doc.createTextNode(' '.join(newValue))
                oldNode = node.firstChild()
                node.replaceChild(newNode, oldNode)
            node = node.nextSibling()
        if not file.open(QFile.WriteOnly):
            print('参数xml打开失败，检测文件' + name + '是否存在')
            return 0
        else:
            xmlWriter = QTextStream(file)
            doc.save(xmlWriter, 4)
            file.close()

def xml_process(file_path):
    # 读取xml文件
    file = QFile(file_path)
    if not file.open(QIODevice.ReadOnly):
        # 如果文件打开失败，不进行处理，返回0
        return 0
    else:
        # 文件打开成功，读取文件，转换格式，返回1
        data = {}
        doc = QDomDocument()
        doc.setContent(file)
        root = doc.documentElement()  # Root为xml文件根节点
        infoNode = root.firstChild()  # 根节点的第一个子节点是信息节点
        info = read_info(infoNode)  # 读取信息节点的信息，返回一个字典
        if 'TPASNum' in info.keys():
            sensor_num = info['TPASNum']  # 传感器个数
        elif 'sensorNum' in info.keys():
            sensor_num = info['sensorNum']
        else:
            sensor_num = 4
        sensorNode = infoNode.nextSibling()  # 数据节点
        for i in range(sensor_num):  # 根据传感器个数读取
            t_matrix_list = []
            name = sensorNode.toElement().tagName()  # 名字
            frameNode = sensorNode.firstChild()  # 传感器数据节点的子节点是各帧数据
            tag = frameNode.toElement().tagName()  # 获取当前节点标签
            tag_front = tag[0:9]  # 获取标签前九个字符
            while tag_front == 'DataFrame':  # 如果标签前九个字符是'DataFrame'，则读取数据
                childNode = frameNode.firstChild()
                childtag = childNode.toElement().tagName()[0:7]
                while childtag !="Datarow":
                    childNode = childNode.nextSibling()
                    childtag = childNode.toElement().tagName()[0:7]
                if childtag =="Datarow":
                    t_matrix = read_matrix(childNode)  # 获取温度矩阵
                    t_matrix_list.append(t_matrix)  # 将转换得到的矩阵放入矩阵列表中
                    frameNode = frameNode.nextSibling()
                    tag = frameNode.toElement().tagName()
                    tag_front = tag[0:9]
            t_matrix_list = np.array(t_matrix_list)  # 转换为三维张量
            if name == 'Side':  # 把名字改一下
                name = 'side'
            if name == 'Top':
                name = 'top'
            data[name] = t_matrix_list
            sensorNode = sensorNode.nextSibling()  # 下一个节点
        file.close()
        length = []
        for sub_data in data.values():
            length.append(sub_data.shape[0])
        length.sort()
        return data, length[0], info


def read_matrix(data_node):
    row = 24
    col = 32
    tMatrix = np.zeros((row, col))
    for i in range(row):
        dataLineStr = data_node.toElement().text()
        dataLineList = dataLineStr.split(' ')
        for j in range(col):
            tMatrix[i, j] = float(dataLineList[j])
        data_node = data_node.nextSibling()
    return tMatrix

def read_info(info_node):
    # 读取xml文件中信息节点的信息，以字典的形式返回
    info = {}
    node = info_node.firstChild()
    # node = info_node
    while not node.isNull():  # 如果不是Null就读取其信息
        name = node.toElement().tagName()  # 标签名
        if not node.firstChild().hasChildNodes():  # 如果没有下一层，则直接转换数据
            value = str2num(node.toElement().text())  # 字符串如果能够转换为数字，则转换为数字
            info[name] = value
        else:  # 如果有下一层，则递归调用，获取子字典
            info[name] = read_info(node)
        node = node.nextSibling()
    return info


def save_data(data, path=''):
    # 将数据以xml的格式保存 data字典
    if path != '':  # 如果有文件名，则保存，否则不保存
        para = load_para('env_information.xml')
        xml_file = Document()  # 创建文件
        root_node = xml_file.createElement('TPAS_Data')  # 根节点
        xml_file.appendChild(root_node)  # 将根节点加入到文件中
        information_node = xml_file.createElement('Information')  # 信息节点
        root_node.appendChild(information_node)  # 将信息节点添加到根节点中

        sensor_num_node = xml_file.createElement('TPASNum')  # 信息：传感器个数
        information_node.appendChild(sensor_num_node)  # 将传感器个数节点添加到信息节点中

        label = xml_file.createElement('label')  # 信息：传感器个数
        information_node.appendChild(label)  # 将传感器个数节点添加到信息节点中

        name = xml_file.createElement('name')  # 信息：传感器个数
        information_node.appendChild(name)  # 将传感器个数节点添加到信息节点中

        time = xml_file.createElement('time')  # 信息：传感器个数
        information_node.appendChild(time)  # 将传感器个数节点添加到信息节点中

        weather = xml_file.createElement('weather')  # 信息：传感器个数
        information_node.appendChild(weather)  # 将传感器个数节点添加到信息节点中
        people_height = xml_file.createElement('people_height')  # 信息：传感器个数
        information_node.appendChild(people_height)  # 将传感器个数节点添加到信息节点中
        sensor_height = xml_file.createElement('sensor_height')  # 信息：传感器个数
        information_node.appendChild(sensor_height)  # 将传感器个数节点添加到信息节点中
        sensor_distance = xml_file.createElement('sensor_distance')  # 信息：传感器个数
        information_node.appendChild(sensor_distance)  # 将传感器个数节点添加到信息节点中
        temperature = xml_file.createElement('temperature')  # 信息：传感器个数
        information_node.appendChild(temperature)  # 将传感器个数节点添加到信息节点中
        clothes = xml_file.createElement('clothes')  # 信息：传感器个数
        information_node.appendChild(clothes)  # 将传感器个数节点添加到信息节点中

        sensor_num_text = xml_file.createTextNode(str(len(data)))  # 根据tMatrixList的长度得到传感器个数
        sensor_num_node.appendChild(sensor_num_text)
        label_text = xml_file.createTextNode(str(para['label']))  # 根据tMatrixList的长度得到传感器个数
        label.appendChild(label_text)
        name_text = xml_file.createTextNode(str(para['name']))  # 根据tMatrixList的长度得到传感器个数
        name.appendChild(name_text)
        time_text = xml_file.createTextNode(str(para['time']))  # 根据tMatrixList的长度得到传感器个数
        time.appendChild(time_text)
        weather_text = xml_file.createTextNode(str(para['weather']))  # 根据tMatrixList的长度得到传感器个数
        weather.appendChild(weather_text)
        people_height_text = xml_file.createTextNode(str(para['people_height']))  # 根据tMatrixList的长度得到传感器个数
        people_height.appendChild(people_height_text)
        sensor_height_text = xml_file.createTextNode(str(para['sensor_height']))  # 根据tMatrixList的长度得到传感器个数
        sensor_height.appendChild(sensor_height_text)
        sensor_distance_text = xml_file.createTextNode(str(para['sensor_distance']))  # 根据tMatrixList的长度得到传感器个数
        sensor_distance.appendChild(sensor_distance_text)
        temperature_text = xml_file.createTextNode(str(para['temperature']))  # 根据tMatrixList的长度得到传感器个数
        temperature.appendChild(temperature_text)
        clothes_text = xml_file.createTextNode(str(para['clothes']))  # 根据tMatrixList的长度得到传感器个数
        clothes.appendChild(clothes_text)

        for name, subData in data.items():
            subData = np.array(subData)
            sensor_data_node = xml_file.createElement(name)  # 创建传感器数据节点
            root_node.appendChild(sensor_data_node)
            if len(subData.shape) == 2:
                dataFrameNode = xml_file.createElement('DataFrame1')  # 创建传感器数据帧节点
                sensor_data_node.appendChild(dataFrameNode)
                row, col = subData.shape[0], subData.shape[1]
                for j in range(row):
                    data_row_node = xml_file.createElement('Datarow' + str(j + 1))
                    dataFrameNode.appendChild(data_row_node)
                    data_row_text = []
                    for k in range(col):
                        data_row_text.append(str(subData[j, k]))
                    data_row_text = ' '.join(data_row_text)
                    data_row_text = xml_file.createTextNode(data_row_text)
                    data_row_node.appendChild(data_row_text)
            elif len(subData.shape) == 3:
                for i in range(subData.shape[0]):
                    dataFrameNode = xml_file.createElement('DataFrame' + str(i + 1))  # 创建传感器数据帧节点
                    sensor_data_node.appendChild(dataFrameNode)
                    row, col = subData.shape[1], subData.shape[2]
                    for j in range(row):
                        data_row_node = xml_file.createElement('Datarow' + str(j + 1))
                        dataFrameNode.appendChild(data_row_node)
                        data_row_text = []
                        for k in range(col):
                            data_row_text.append(str(subData[i, j, k]))
                        data_row_text = ' '.join(data_row_text)
                        data_row_text = xml_file.createTextNode(data_row_text)
                        data_row_node.appendChild(data_row_text)
        with open(path, 'w') as f:
            xml_file.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
            f.close()


