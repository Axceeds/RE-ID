import pymysql
from PyQt5.QtGui import QFont
import file_process as fp
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QTableWidget, QComboBox, QHBoxLayout
from PyQt5.QtWidgets import QVBoxLayout, QAbstractItemView, QLabel
from PyQt5.QtWidgets import QToolButton, QTableWidgetItem


class Database(QWidget):
    """########################################
    数据库查询界面
    功能：
        1、建立一个查询界面，能够返回文件路径
        2、能够直接对数据进行处理，在表格里直接显示其结果
    ########################################"""
    selectFile = pyqtSignal(str)  # 文件选定信号，返回文件路径

    def __init__(self):
        super().__init__()
        self.setWindowTitle('数据库操作')
        self.sql = Sql()
        self.menu = fp.load_para('menu.xml')  # 载入选项列表menu

        self.tester_nums = ['all'] + [str(i) for i in range(1, 7)]  # 可选项目
        self.actionList = ['all'] + self.menu['action'] + ['walkingSlowly or walkingFast']
        self.directionList = ['all', '0', '45', '90', '135', '180']
        self.header = ['被测者编号', '动作', '方向',
                       '文件路径']  # , '侧视最大变化值', '顶视最大变化值', '纵轴长度', '横轴长度', '横纵比', ]  # 表头

        self.testerNumComboBox = QComboBox()  # 选择被测者编号
        self.testerNumComboBox.addItems(self.tester_nums)
        self.actionComboBox = QComboBox()  # 选择动作
        self.actionComboBox.addItems(self.actionList)
        self.directionComboBox = QComboBox()  # 选择朝向
        self.directionComboBox.addItems(self.directionList)
        self.getResultBt = QToolButton()
        self.getResultBt.setText('获取文件')

        self.selectLayout = QVBoxLayout()
        self.selectLayout.addWidget(QLabel('被测者编号'))
        self.selectLayout.addWidget(self.testerNumComboBox)
        self.selectLayout.addWidget(QLabel('动作'))
        self.selectLayout.addWidget(self.actionComboBox)
        self.selectLayout.addWidget(QLabel('方向'))
        self.selectLayout.addWidget(self.directionComboBox)
        self.selectLayout.addWidget(self.getResultBt)
        self.selectLayout.addStretch()

        self.showTable = QTableWidget()  # 表格，用于显示结果
        self.showTable.setRowCount(3000)
        self.showTable.setColumnCount(len(self.header))
        self.showTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.showTable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.showTable.setHorizontalHeaderLabels(self.header)
        self.showTable.horizontalHeader().setFont(QFont("黑体", 12, QFont.Bold))

        self.mainLayout = QHBoxLayout()
        self.mainLayout.addLayout(self.selectLayout)
        self.mainLayout.addWidget(self.showTable)

        self.setLayout(self.mainLayout)
        self.setFont(QFont("Microsoft Yahei", 12, QFont.Normal))  # 设定字体及大小
        self.setMinimumSize(900, 900)
        self.move(0, 0)

        self.getResultBt.clicked.connect(self.result_bt_clicked)  # 单击获取文件按钮事件
        self.showTable.itemDoubleClicked.connect(self.get_file_path)  # 双击单元格事件
        self.showTable.horizontalHeader().sectionClicked.connect(self.order)  # 单击表头事件

    def display(self, content):
        """########################################
        在表格中显示
        输入：
            1、显示内容content（二维，字符串组成的列表）
        输出：
            None
        备注：
        #######################################"""
        self.showTable.clearContents()
        for i, row in enumerate(content):
            for j, item in enumerate(row):
                self.showTable.setItem(i, j, QTableWidgetItem(str(item)))

    def order(self, index):
        """########################################
        单击表头，进行排序
        输入：
            1、index（表头编号）
        输出：
            None
        备注：
        #######################################"""
        self.showTable.sortItems(index, Qt.DescendingOrder)

    def result_bt_clicked(self):
        """########################################
        获取结果按钮按下，查询数据库，显示结果
        输入：
            None
        输出：
            None
        备注：
        #######################################"""
        tester_num = self.testerNumComboBox.currentText()  # 获取编号，动作，方向，如果是all，则特殊处理一下
        if tester_num == 'all':
            tester_nums = self.tester_nums[1:]
        else:
            tester_nums = [tester_num]
        action = self.actionComboBox.currentText()
        if action == 'all':
            actions = self.actionList[1:]
        else:
            actions = [action]
        direction = self.directionComboBox.currentText()
        if direction == 'all':
            directions = self.directionList[1:]
        else:
            directions = [direction]

        content = []  # 获取结果
        for tester_num in tester_nums:
            for action in actions:
                for direction in directions:
                    paths = self.sql.search(columns=['type', 'direction', 'testeeNumber'],
                                            values=[action, direction, tester_num])
                    for path in paths:
                        row = [tester_num, action, direction, path]
                        content.append(row)

        self.display(content)  # 显示结果

    def get_file_path(self):
        """########################################
        双击某一行，发出信号
        输入：
            None
        输出：
            None
        备注：
        #######################################"""
        row = self.showTable.selectionModel().selection().indexes()[0].row()
        path = self.showTable.item(row, 3).text()
        self.selectFile.emit(path)


class Sql:
    """########################################
    数据库
    功能：
        1、查询数据库，返回文件路径
    ########################################"""

    @staticmethod
    def search(columns, values):
        """########################################
        从数据库中获取文件路径
        输入：
            1、列columns（字符串组成的列表）
            2、值values（字符串组成的列表）
        输出：
            1、迭代输出文件路径（字符串）
        备注：
            1、获取满足所有条件的数据
            2、如果某个value是'all'，那么就会选取整列
            3、如果某个value中有or这个关键词，那么会返回or连接的多个量
        ########################################"""
        if len(columns) != len(values):
            raise Exception('error')
        else:
            order = "select path from main where "
            for column, value in zip(columns, values):
                if value == 'all':
                    continue
                if 'or' in value:
                    splited_values = value.split(' or ')
                    order = order + '('
                    for splitedValue in splited_values:
                        order = order + column + " = " + "\'" + str(splitedValue) + "\'" + " or "
                    order = order[:-4]
                    order = order + ') and '
                else:
                    order = order + column + " = " + "\'" + str(value) + "\'" + " and "
            has_non_all = False
            for value in values:
                if value != 'all':
                    has_non_all = True
            if has_non_all:
                order = order[:-5]
            else:
                order = "select path from main"

        result = []
        conn = pymysql.connect(host='localhost', user='root', password='123456', database='TPASdataset', charset='utf8')
        cursor = conn.cursor()
        cursor.execute(order)  # 执行查询的SQL语句
        query = cursor.fetchall()
        # 在此处将列表里套的元组改为列表
        for i in range(len(query)):
            result.append(query[i][0])
        conn.commit()
        conn.close()
        for path in result:
            yield path
