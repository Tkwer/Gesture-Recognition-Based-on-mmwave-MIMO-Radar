# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled备份.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from re import X
from turtle import left
from pyqtgraph import GraphicsLayoutWidget
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QVBoxLayout, QSizePolicy, QMessageBox, QWidget,QListWidget
import pyqtgraph as pg
import globalvar as gl
import os
import sys
import itertools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as  np
import time
gl._init()

trandir = []
tran_ratio = []

def printlog(textEdit,string,fontcolor='green'):
    textEdit.moveCursor(QtGui.QTextCursor.End)
    gettime = time.strftime("%H:%M:%S", time.localtime())
    textEdit.append("<font color="+fontcolor+">"+str(gettime)+"-->"+string+"</font>")


def get_filelist(dir,Filelist):
    newDir=dir
    #注意看dir是文件名还是路径＋文件名！！！！！！！！！！！！！！
    if os.path.isfile(dir):
        dir_ = os.path.basename(dir)  
        if (dir_[:2] == 'DT') and (dir_[-4:] == '.npy'):
            Filelist[0].append(dir)
        elif (dir_[:2] == 'RT') and (dir_[-4:] == '.npy'):
            Filelist[1].append(dir)
        elif (dir_[:3] == 'RDT') and (dir_[-4:] == '.npy'):
            Filelist[2].append(dir)
        elif (dir_[:3] == 'ART') and (dir_[-4:] == '.npy'):
            Filelist[3].append(dir)    
        elif (dir_[:3] == 'ERT') and (dir_[-4:] == '.npy'):
            Filelist[4].append(dir)  
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir=os.path.join(dir,s)
            get_filelist(newDir,Filelist)
    return Filelist

class ViolationItem (QtWidgets.QListWidget):
    def __init__(self, parent=None, radioBtnname='test', download_path='',textEditwidget=None):
        super(ViolationItem, self).__init__(parent)
        self.layoutWidget = QtWidgets.QWidget(self)
        self.layoutWidget.resize(320,40)
        # self.layoutWidget.setGeometry(QtCore.QRect(0, 100, 0, 40))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.radioButton = QtWidgets.QRadioButton(self.layoutWidget)
        self.radioButton.setObjectName("radioButton")
        self.radioButton.setText(radioBtnname)
        self.file_folder = download_path+'/'+ radioBtnname
        filelist = [[] for i in range(5)]#5代表5个特征
        filelist = get_filelist(self.file_folder,filelist)

        self.horizontalLayout.addWidget(self.radioButton)
        self.horizontalSlider = QtWidgets.QSlider(self.layoutWidget)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalLayout.addWidget(self.horizontalSlider)
        self.lineEdit = QtWidgets.QLineEdit(self.layoutWidget)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.horizontalSlider.setMaximum(len(filelist[0])-1)
        self.horizontalSlider.setValue(len(filelist[0])*0.8)
        self.lineEdit.setText(str(self.horizontalSlider.value()))
        # if radioBtnname[0]=='p':
        #     self.horizontalSlider.setValue(len(filelist[0]))
        #     self.lineEdit.setText(str(self.horizontalSlider.value()))
        #     self.horizontalSlider.setDisabled(True)
        #     self.lineEdit.setDisabled(True)
        self.horizontalLayout.setStretch(0, 2)
        self.horizontalLayout.setStretch(1, 6)
        self.horizontalLayout.setStretch(2, 1)
        self.horizontalSlider.valueChanged.connect(self.valChange)
        self.lineEdit.editingFinished.connect(self.valChange)
        self.radioButton.clicked.connect(lambda:self.radbtnSelected(textEditwidget,radioBtnname))

    def valChange(self):
        # print("ss")
        self.lineEdit.setText(str(self.horizontalSlider.value()))
        self.horizontalSlider.setValue(int(self.lineEdit.text()))

    def radbtnSelected(self, name, string):
        if(self.radioButton.isChecked()):
            trandir.append(self.file_folder)
            tran_ratio.append(self.horizontalSlider.value()/self.horizontalSlider.maximum())
        else:
            tran_ratio.pop(trandir.index(self.file_folder))
            trandir.remove(self.file_folder)
        gl.set_value('train_validir',trandir)
        gl.set_value('trian_ratio',tran_ratio)
        printf_str =''
        for trandir_ in trandir:
            printf_str = printf_str + os.path.basename(trandir_)+', '+str(tran_ratio[trandir.index(trandir_)]*100)+'%'
        printlog(name,'select dataset:'+printf_str)




class MplCanvas(FigureCanvas):
    """这是一个窗口部件，即QWidget（当然也是FigureCanvasAgg）"""
    def __init__(self, parent=None, width=3, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(left=0.15,top=0.95,bottom = 0.15)
        self.axes = fig.add_subplot(111)
        # 每次plot()调用的时候，我们希望原来的坐标轴被清除(所以False)
        # self.axes.hold(False)
        self.compute_initial_figure()
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

class DrawMplCanvas(MplCanvas):
    def __init__(self,parent=None, width=3, height=4, dpi=100,set_xlim=[0,50],set_ylim=[0,2.0],xlabel='Epcho',
                    ylabel='Loss',label1='train_loss',label2='vali_loss'):
        self.label1 = label1
        self.label2 = label2
        super().__init__(parent, width, height, dpi)
        # 设置坐标轴范围
        self.axes.set_xlim(set_xlim)
        self.axes.set_ylim(set_ylim)
        # 设置坐标轴名称
        self.axes.set_xlabel(xlabel)
        self.axes.set_ylabel(ylabel)
        # # 设置坐标轴刻度
        # x_ticks = np.arange(set_xlim[0], set_xlim[1], 1)
        # self.axes.set_xticks(x_ticks)
        # self.axes.grid(axis='x')

    def compute_initial_figure(self):
        self.hl1, = self.axes.plot([], [], '-', label=self.label1,linewidth=2)   #绘制线对象，plot返回值类型，要加逗号
        self.hl2, = self.axes.plot([], [], '-', label=self.label2,linewidth=2)   #绘制线对象，plot返回值类型，要加逗号
        self.axes.legend()

# 混淆矩阵默认是这样的：classes = ['back','dblclick','down', 'front', 'left', 'right','up'],
# 为了方便对比 需要转换成： classes = ['front', 'back', 'up', 'down', 'left', 'right','dblclick'],[3,0,6,2,4,5,1]

class DrawheartCanvas(MplCanvas):
    def __init__(self, parent=None, width=3, height=4, dpi=100, cm = None, classes = ['front', 'back', 'up', 'down', 'left', 'right','dblclick'], 
                    textEdit = None, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
        self.normalize = normalize
        self.cm = cm
        self.cmap = cmap
        self.classes = classes
        self.textEdit = textEdit
        super().__init__(parent, width, height, dpi)

     # 绘制混淆矩阵   
    def compute_initial_figure(self):
        '''
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        Input
        - cm : 计算出的混淆矩阵的值
        - classes : 混淆矩阵中每一行每一列对应的列
        - normalize : True:显示百分比, False:显示个数
        '''
        if self.normalize:
            cm = self.cm.astype('float') / self.cm.sum(axis=1)[:, np.newaxis]
            # printlog(self.textEdit,"Normalized confusion matrix",fontcolor='blue')
        else:
            pass
            # printlog(self.textEdit,"Confusion matrix, without normalization",fontcolor='blue')
        # printlog(self.textEdit,str(cm),fontcolor='blue')
        self.axes.imshow(cm, interpolation='nearest', cmap=self.cmap)
        # self.axes.colorbar()
        # plt.switch_backend('agg')
        tick_marks = np.arange(len(self.classes))
        self.axes.set_xticks(tick_marks)
        self.axes.set_yticks(tick_marks)
        self.axes.set_xticklabels(self.classes)
        self.axes.set_yticklabels(self.classes)
        # 。。。。。。。。。。。。新增代码开始处。。。。。。。。。。。。。。。。
        # x,y轴长度一致(问题1解决办法）
        self.axes.axis("equal")
        # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
        # ax = self.axes.gca()  # 获得当前axis
        left, right = self.axes.get_xlim()  # 获得x轴最大最小值
        self.axes.spines['left'].set_position(('data', left))
        self.axes.spines['right'].set_position(('data', right))
        for edge_i in ['top', 'bottom', 'right', 'left']:
            self.axes.spines[edge_i].set_edgecolor("white")
        # 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            num = '{:.2f}'.format(cm[i, j]) if self.normalize else int(cm[i, j])
            self.axes.text(j, i, num,
                    verticalalignment='center',
                    horizontalalignment="center",
                    color="white" if float(num) > thresh else "black")
        # self.axes.tight_layout()
        self.axes.set_ylabel('True label')
        self.axes.set_xlabel('Predicted label')

    def updatashow(self,cm1):
        self.axes.clear()
        # 调换矩阵位置方便比较
        cm = cm1[:,[3,0,6,2,4,5,1]]
        cm = cm[[3,0,6,2,4,5,1],:]
        if self.normalize:
            mycm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        else:
            pass
        self.axes.imshow(mycm, interpolation='nearest', cmap=self.cmap)
        tick_marks = np.arange(len(self.classes))
        self.axes.set_xticks(tick_marks)
        self.axes.set_yticks(tick_marks)
        self.axes.set_xticklabels(self.classes)
        self.axes.set_yticklabels(self.classes)
        # 。。。。。。。。。。。。新增代码开始处。。。。。。。。。。。。。。。。
        # x,y轴长度一致(问题1解决办法）
        self.axes.axis("equal")
        # x轴处理一下，如果x轴或者y轴两边有空白的话(问题2解决办法）
        # ax = self.axes.gca()  # 获得当前axis
        left, right = self.axes.get_xlim()  # 获得x轴最大最小值
        self.axes.spines['left'].set_position(('data', left))
        self.axes.spines['right'].set_position(('data', right))
        for edge_i in ['top', 'bottom', 'right', 'left']:
            self.axes.spines[edge_i].set_edgecolor("white")
        # 。。。。。。。。。。。。新增代码结束处。。。。。。。。。。。。。。。。
        thresh = mycm.max() / 2.
        for i, j in itertools.product(range(mycm.shape[0]), range(mycm.shape[1])):
            num = '{:.2f}'.format(mycm[i, j]) if self.normalize else int(mycm[i, j])
            self.axes.text(j, i, num,
                    verticalalignment='center',
                    horizontalalignment="center",
                    color="white" if float(num) > thresh else "black")
        self.axes.set_ylabel('True label')
        self.axes.set_xlabel('Predicted label')
        self.draw()
     

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1300, 900)
        pg.setConfigOption('background', '#dcdcdc')
        pg.setConfigOption('foreground', 'd')  
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.tab)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.groupBox_3 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_12 = QtWidgets.QLabel(self.groupBox_3)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_8.addWidget(self.label_12)
        self.line_14 = QtWidgets.QFrame(self.groupBox_3)
        self.line_14.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_14.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_14.setObjectName("line_14")
        self.horizontalLayout_8.addWidget(self.line_14)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.groupBox_3)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.horizontalLayout_8.addWidget(self.lineEdit_2)
        self.line_15 = QtWidgets.QFrame(self.groupBox_3)
        self.line_15.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_15.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_15.setObjectName("line_15")
        self.horizontalLayout_8.addWidget(self.line_15)
        self.pushButton_5 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_5.setObjectName("pushButton_5")
        self.horizontalLayout_8.addWidget(self.pushButton_5)
        self.verticalLayout_4.addLayout(self.horizontalLayout_8)
        self.line_12 = QtWidgets.QFrame(self.groupBox_3)
        self.line_12.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_12.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_12.setObjectName("line_12")
        self.verticalLayout_4.addWidget(self.line_12)
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_4.setEnabled(True)
        self.groupBox_4.setMouseTracking(False)
        self.groupBox_4.setTabletTracking(False)
        self.groupBox_4.setFocusPolicy(QtCore.Qt.NoFocus)
        self.groupBox_4.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.groupBox_4.setAcceptDrops(False)
        self.groupBox_4.setAutoFillBackground(False)
        self.groupBox_4.setInputMethodHints(QtCore.Qt.ImhNone)
        self.groupBox_4.setAlignment(int(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter))
        self.groupBox_4.setFlat(False)
        self.groupBox_4.setCheckable(False)
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.listView = QtWidgets.QListWidget(self.groupBox_4)
        self.listView.setObjectName("listView")
        self.verticalLayout_22 = QtWidgets.QVBoxLayout(self.listView)
        self.verticalLayout_6.addWidget(self.listView)
        self.verticalLayout_4.addWidget(self.groupBox_4)

        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.label_15 = QtWidgets.QLabel(self.groupBox_3)
        self.label_15.setObjectName("label_15")
        self.horizontalLayout_11.addWidget(self.label_15)
        self.line_17 = QtWidgets.QFrame(self.groupBox_3)
        self.line_17.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_17.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_17.setObjectName("line_17")
        self.horizontalLayout_11.addWidget(self.line_17)
        self.comboBox_5 = QtWidgets.QComboBox(self.groupBox_3)
        self.comboBox_5.setObjectName("comboBox_5")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.comboBox_5.addItem("")
        self.horizontalLayout_11.addWidget(self.comboBox_5)
        self.label_14 = QtWidgets.QLabel(self.groupBox_3)
        self.label_14.setObjectName("label_14")
        self.horizontalLayout_11.addWidget(self.label_14)
        self.line_16 = QtWidgets.QFrame(self.groupBox_3)
        self.line_16.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_16.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_16.setObjectName("line_16")
        self.horizontalLayout_11.addWidget(self.line_16)
        self.comboBox_4 = QtWidgets.QComboBox(self.groupBox_3)
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("CNN")
        self.horizontalLayout_11.addWidget(self.comboBox_4)
        self.verticalLayout_4.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_14.addItem(spacerItem1)
        self.pushButton_8 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_8.setObjectName("pushButton_8")
        self.horizontalLayout_14.addWidget(self.pushButton_8)
        self.verticalLayout_4.addLayout(self.horizontalLayout_14)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem2)
        self.line_19 = QtWidgets.QFrame(self.groupBox_3)
        self.line_19.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_19.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_19.setObjectName("line_19")
        self.verticalLayout_4.addWidget(self.line_19)
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.label_22 = QtWidgets.QLabel(self.groupBox_3)
        self.label_22.setObjectName("label_22")
        self.horizontalLayout_19.addWidget(self.label_22)
        self.line_20 = QtWidgets.QFrame(self.groupBox_3)
        self.line_20.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_20.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_20.setObjectName("line_20")
        self.horizontalLayout_19.addWidget(self.line_20)
        self.comboBox_6 = QtWidgets.QComboBox(self.groupBox_3)
        self.comboBox_6.setObjectName("comboBox_6")

        self.horizontalLayout_19.addWidget(self.comboBox_6)
    
        self.comboBox_7 = QtWidgets.QComboBox(self.groupBox_3)
        self.comboBox_7.setObjectName("comboBox_7")
        self.comboBox_7.addItems(['RT_CNN','DT_CNN','RT+DT+ART_CNN','RT+DT+ART_CNN-LSTM','ALL_CNN-LSTM','ALL_FeatureFusionNet'])
        # self.comboBox_7.addItems(['RT_CNN','DT_CNN','ART_CNN-LL','RT+DT+ART_CNN','RT+DT+ART_CNN-LSTM','ALL_CNN-LSTM','ALL_FeatureFusionNet'])
        self.horizontalLayout_19.addWidget(self.comboBox_7)
        self.pushButton_9 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_9.setObjectName("pushButton_9")
        self.horizontalLayout_19.addWidget(self.pushButton_9)
        self.verticalLayout_4.addLayout(self.horizontalLayout_19)
        self.label_17 = QtWidgets.QLabel(self.groupBox_3)
        self.label_17.setObjectName("label_17")
        self.verticalLayout_4.addWidget(self.label_17)
        self.textEdit = QtWidgets.QTextEdit(self.groupBox_3)
        self.textEdit.setObjectName("textEdit")
        self.verticalLayout_4.addWidget(self.textEdit)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.label_16 = QtWidgets.QLabel(self.groupBox_3)
        self.label_16.setObjectName("label_16")
        self.horizontalLayout_13.addWidget(self.label_16)
        self.progressBar = QtWidgets.QProgressBar(self.groupBox_3)
        # self.progressBar.setValue(0)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_13.addWidget(self.progressBar)
        self.verticalLayout_4.addLayout(self.horizontalLayout_13)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_12.addItem(spacerItem3)
        self.pushButton_7 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_7.setMinimumSize(QtCore.QSize(1, 0))
        self.pushButton_7.setObjectName("pushButton_7")
        self.horizontalLayout_12.addWidget(self.pushButton_7)
        self.line_18 = QtWidgets.QFrame(self.groupBox_3)
        self.line_18.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_18.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_18.setObjectName("line_18")
        self.horizontalLayout_12.addWidget(self.line_18)
        self.pushButton_6 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_6.setMinimumSize(QtCore.QSize(1, 0))
        self.pushButton_6.setObjectName("pushButton_6")
        self.horizontalLayout_12.addWidget(self.pushButton_6)
        self.verticalLayout_4.addLayout(self.horizontalLayout_12)
        self.horizontalLayout_5.addWidget(self.groupBox_3)
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.label_18 = QtWidgets.QLabel(self.groupBox_2)
        self.label_18.setAlignment(QtCore.Qt.AlignCenter)
        self.label_18.setObjectName("label_18")
        self.horizontalLayout_15.addWidget(self.label_18)
        self.label_19 = QtWidgets.QLabel(self.groupBox_2)
        self.label_19.setAlignment(QtCore.Qt.AlignCenter)
        self.label_19.setObjectName("label_19")
        self.horizontalLayout_15.addWidget(self.label_19)
        self.verticalLayout_5.addLayout(self.horizontalLayout_15)
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        # self.graphicsView_7 = QtWidgets.QGraphicsView(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.canvas = DrawMplCanvas(self.groupBox_2, width=1, height=1, dpi=100)
        self.canvas1 = DrawMplCanvas(self.groupBox_2, width=1, height=1, dpi=100,set_xlim=[0,50],set_ylim=[0,1.0],xlabel='Epcho',
                    ylabel='Accuracy',label1='train_accuracy',label2='vali__accuracy')
        self.horizontalLayout_16.addWidget(self.canvas)
        self.horizontalLayout_16.addWidget(self.canvas1)
        self.verticalLayout_5.addLayout(self.horizontalLayout_16)
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.label_20 = QtWidgets.QLabel(self.groupBox_2)
        self.label_20.setAlignment(QtCore.Qt.AlignCenter)
        self.label_20.setObjectName("label_20")
        self.horizontalLayout_17.addWidget(self.label_20)
        self.label_21 = QtWidgets.QLabel(self.groupBox_2)
        self.label_21.setAlignment(QtCore.Qt.AlignCenter)
        self.label_21.setObjectName("label_21")
        self.horizontalLayout_17.addWidget(self.label_21)
        self.verticalLayout_5.addLayout(self.horizontalLayout_17)
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        # self.graphicsView_8 = QtWidgets.QGraphicsView(self.groupBox_2)
        # self.graphicsView_8.setMaximumSize(QtCore.QSize(289, 289))
        # self.graphicsView_8.setObjectName("graphicsView_8")
        self.con_matrix1 = DrawheartCanvas(self.groupBox_2, width=1, height=1, dpi=100, cm=np.eye(7),textEdit=self.textEdit)
        self.con_matrix2 = DrawheartCanvas(self.groupBox_2, width=1, height=1, dpi=100, cm=np.eye(7),textEdit=self.textEdit)
        self.horizontalLayout_18.addWidget(self.con_matrix1)
        # self.graphicsView_10 = QtWidgets.QGraphicsView(self.groupBox_2)
        # self.graphicsView_10.setMaximumSize(QtCore.QSize(289, 289))
        # self.graphicsView_10.setObjectName("graphicsView_10")
        self.horizontalLayout_18.addWidget(self.con_matrix2)
        self.verticalLayout_5.addLayout(self.horizontalLayout_18)
        self.horizontalLayout_5.addWidget(self.groupBox_2)
        self.horizontalLayout_5.setStretch(0, 1)
        self.horizontalLayout_5.setStretch(1, 2)
        self.horizontalLayout_7.addLayout(self.horizontalLayout_5)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.verticalLayout.setContentsMargins(-1, 0, -1, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        
        self.groupBox_8 = QtWidgets.QGroupBox(self.tab_2)
        self.groupBox_8.setMinimumSize(QtCore.QSize(100, 100))
        self.groupBox_8.setFlat(False)
        self.groupBox_8.setObjectName("groupBox_8")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_8)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.graphicsView_3 = GraphicsLayoutWidget(self.groupBox_8)
        self.graphicsView_3.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graphicsView_3.sizePolicy().hasHeightForWidth())
        self.graphicsView_3.setSizePolicy(sizePolicy)
        self.graphicsView_3.setMinimumSize(QtCore.QSize(255, 255))
        self.graphicsView_3.setMaximumSize(QtCore.QSize(255, 255))
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.gridLayout_2.addWidget(self.graphicsView_3, 2, 2, 1, 1)
        self.graphicsView_2 = GraphicsLayoutWidget(self.groupBox_8)
        self.graphicsView_2.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graphicsView_2.sizePolicy().hasHeightForWidth())
        self.graphicsView_2.setSizePolicy(sizePolicy)
        self.graphicsView_2.setMinimumSize(QtCore.QSize(255, 255))
        self.graphicsView_2.setMaximumSize(QtCore.QSize(255, 255))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.gridLayout_2.addWidget(self.graphicsView_2, 2, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.groupBox_8)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 5, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.groupBox_8)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.gridLayout_2.addWidget(self.label_8, 0, 2, 1, 1)
        self.graphicsView_6 = GraphicsLayoutWidget(self.groupBox_8)
        self.graphicsView_6.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(10)
        sizePolicy.setHeightForWidth(self.graphicsView_6.sizePolicy().hasHeightForWidth())
        self.graphicsView_6.setSizePolicy(sizePolicy)
        self.graphicsView_6.setMinimumSize(QtCore.QSize(255, 255))
        self.graphicsView_6.setMaximumSize(QtCore.QSize(255, 255))
        self.graphicsView_6.setObjectName("graphicsView_6")
        self.gridLayout_2.addWidget(self.graphicsView_6, 6, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.groupBox_8)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 5, 2, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.groupBox_8)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 0, 1, 1, 1)
        self.graphicsView = GraphicsLayoutWidget(self.groupBox_8)
        self.graphicsView.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(10)
        sizePolicy.setHeightForWidth(self.graphicsView.sizePolicy().hasHeightForWidth())
        self.graphicsView.setSizePolicy(sizePolicy)
        self.graphicsView.setMinimumSize(QtCore.QSize(255, 255))
        self.graphicsView.setMaximumSize(QtCore.QSize(255, 255))
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout_2.addWidget(self.graphicsView, 2, 0, 1, 1)
        self.graphicsView_4 = GraphicsLayoutWidget(self.groupBox_8)
        self.graphicsView_4.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graphicsView_4.sizePolicy().hasHeightForWidth())
        self.graphicsView_4.setSizePolicy(sizePolicy)
        self.graphicsView_4.setMinimumSize(QtCore.QSize(255, 255))
        self.graphicsView_4.setMaximumSize(QtCore.QSize(255, 255))
        self.graphicsView_4.setObjectName("graphicsView_4")
        self.gridLayout_2.addWidget(self.graphicsView_4, 6, 1, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.groupBox_8)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.gridLayout_2.addWidget(self.label_9, 5, 1, 1, 1)
        self.graphicsView_5 = QtWidgets.QLabel(self.groupBox_8)
        self.graphicsView_5.setStyleSheet('border-width: 1px;border-style: solid;border-color: rgb(255, 170, 0);background-color: rgb(180,180,180);')
        self.graphicsView_5.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graphicsView_5.sizePolicy().hasHeightForWidth())
        self.graphicsView_5.setSizePolicy(sizePolicy)
        self.graphicsView_5.setMinimumSize(QtCore.QSize(255, 255))
        self.graphicsView_5.setMaximumSize(QtCore.QSize(255, 255))
        self.graphicsView_5.setObjectName("graphicsView_5")
        self.gridLayout_2.addWidget(self.graphicsView_5, 6, 2, 1, 1)
        self.graphicsView.raise_()
        self.graphicsView_6.raise_()
        self.graphicsView_2.raise_()
        self.graphicsView_4.raise_()
        self.graphicsView_3.raise_()
        self.label_7.raise_()
        self.label.raise_()
        self.graphicsView_5.raise_()
        self.label_8.raise_()
        self.label_11.raise_()
        self.label_9.raise_()
        self.label_10.raise_()
        self.horizontalLayout.addWidget(self.groupBox_8)
        self.groupBox = QtWidgets.QGroupBox(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(6)
        sizePolicy.setHeightForWidth(self.groupBox.sizePolicy().hasHeightForWidth())
        self.groupBox.setSizePolicy(sizePolicy)
        self.groupBox.setFlat(False)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_2.addWidget(self.label_6)
        self.line_5 = QtWidgets.QFrame(self.groupBox)
        self.line_5.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.horizontalLayout_2.addWidget(self.line_5)
        self.pushButton_3 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout_2.addWidget(self.pushButton_3)
        self.horizontalLayout_2.setStretch(0, 1)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.line_7 = QtWidgets.QFrame(self.groupBox)
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.verticalLayout_2.addWidget(self.line_7)
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setObjectName("lineEdit")
        self.verticalLayout_2.addWidget(self.lineEdit)
        self.line_8 = QtWidgets.QFrame(self.groupBox)
        self.line_8.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.verticalLayout_2.addWidget(self.line_8)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.line_6 = QtWidgets.QFrame(self.groupBox)
        self.line_6.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.horizontalLayout_3.addWidget(self.line_6)
        self.comboBox_2 = QtWidgets.QComboBox(self.groupBox)
        self.comboBox_2.setObjectName("comboBox_2")
        self.horizontalLayout_3.addWidget(self.comboBox_2)
        self.horizontalLayout_3.setStretch(0, 2)
        self.horizontalLayout_3.setStretch(2, 4)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.line_9 = QtWidgets.QFrame(self.groupBox)
        self.line_9.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_9.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_9.setObjectName("line_9")
        self.verticalLayout_2.addWidget(self.line_9)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_4.addWidget(self.label_4)
        self.line_2 = QtWidgets.QFrame(self.groupBox)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.horizontalLayout_4.addWidget(self.line_2)
        self.spinBox = QtWidgets.QSpinBox(self.groupBox)
        self.spinBox.setObjectName("spinBox")
        self.horizontalLayout_4.addWidget(self.spinBox)
        self.horizontalLayout_4.setStretch(0, 2)
        self.horizontalLayout_4.setStretch(1, 1)
        self.horizontalLayout_4.setStretch(2, 4)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.line_10 = QtWidgets.QFrame(self.groupBox)
        self.line_10.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_10.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_10.setObjectName("line_10")
        self.verticalLayout_2.addWidget(self.line_10)
        spacerItem4 = QtWidgets.QSpacerItem(20, 448, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem4)
        self.horizontalLayout.addWidget(self.groupBox)
        self.horizontalLayout.setSpacing(10)
        self.horizontalLayout.setStretch(0, 6)
        self.horizontalLayout.setStretch(1, 2)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.groupBox_10 = QtWidgets.QGroupBox(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_10.sizePolicy().hasHeightForWidth())
        self.groupBox_10.setSizePolicy(sizePolicy)
        self.groupBox_10.setMinimumSize(QtCore.QSize(0, 60))
        self.groupBox_10.setFlat(False)
        self.groupBox_10.setCheckable(False)
        self.groupBox_10.setObjectName("groupBox_10")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.groupBox_10)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_23 = QtWidgets.QLabel(self.groupBox_10)
        self.label_23.setObjectName("label_23")
        self.horizontalLayout_6.addWidget(self.label_23)
        self.comboBox = QtWidgets.QComboBox(self.groupBox_10)
        self.comboBox.setObjectName("comboBox")
        self.horizontalLayout_6.addWidget(self.comboBox)
        self.line = QtWidgets.QFrame(self.groupBox_10)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout_6.addWidget(self.line)
        self.splitter_7 = QtWidgets.QSplitter(self.groupBox_10)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.splitter_7.sizePolicy().hasHeightForWidth())
        self.splitter_7.setSizePolicy(sizePolicy)
        self.splitter_7.setMinimumSize(QtCore.QSize(20, 0))
        self.splitter_7.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_7.setObjectName("splitter_7")
        self.pushButton = QtWidgets.QPushButton(self.splitter_7)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setMaximumSize(QtCore.QSize(200, 200))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setCheckable(True)
        self.horizontalLayout_6.addWidget(self.splitter_7)
        self.label_3 = QtWidgets.QLabel(self.groupBox_10)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_6.addWidget(self.label_3)
        self.line_3 = QtWidgets.QFrame(self.groupBox_10)
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.horizontalLayout_6.addWidget(self.line_3)
        self.horizontalSlider = QtWidgets.QSlider(self.groupBox_10)
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(11)
        self.horizontalSlider.setSingleStep(1)  # 步长
        self.horizontalSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)  # 设置刻度位置，在下方
        self.horizontalSlider.setTickInterval(1)  # 设置刻度间隔        
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(6)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalSlider.sizePolicy().hasHeightForWidth())
        self.horizontalSlider.setSizePolicy(sizePolicy)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalLayout_6.addWidget(self.horizontalSlider)
        self.line_4 = QtWidgets.QFrame(self.groupBox_10)
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.horizontalLayout_6.addWidget(self.line_4)
        self.label_5 = QtWidgets.QLabel(self.groupBox_10)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_6.addWidget(self.label_5)
        self.lcdNumber = QtWidgets.QLCDNumber(self.groupBox_10)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lcdNumber.sizePolicy().hasHeightForWidth())
        self.lcdNumber.setSizePolicy(sizePolicy)
        self.lcdNumber.setObjectName("lcdNumber")
        self.horizontalLayout_6.addWidget(self.lcdNumber)
        spacerItem5 = QtWidgets.QSpacerItem(40, 21, QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem5)
        self.pushButton_2 = QtWidgets.QPushButton(self.groupBox_10)
        self.pushButton_2.setMinimumSize(QtCore.QSize(1, 0))
        self.pushButton_2.setObjectName("pushButton_2")

        self.pushButton_23 = QtWidgets.QPushButton(self.groupBox_10)
        self.pushButton_23.setMinimumSize(QtCore.QSize(1, 0))
        self.pushButton_23.setObjectName("pushButton_23")
        self.horizontalLayout_6.addWidget(self.pushButton_2)
        self.horizontalLayout_6.addWidget(self.pushButton_23)
        self.line_11 = QtWidgets.QFrame(self.groupBox_10)
        self.line_11.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")
        self.horizontalLayout_6.addWidget(self.line_11)
        self.pushButton_4 = QtWidgets.QPushButton(self.groupBox_10)
        self.pushButton_4.setMinimumSize(QtCore.QSize(1, 0))
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout_6.addWidget(self.pushButton_4)
        self.verticalLayout.addWidget(self.groupBox_10)
        self.verticalLayout.setStretch(0, 6)
        self.verticalLayout_3.addLayout(self.verticalLayout)
        self.tabWidget.addTab(self.tab_2, "")
        self.gridLayout.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1155, 25))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionload = QtWidgets.QAction(MainWindow)
        self.actionload.setObjectName("actionload")
        self.menu.addSeparator()
        self.menu.addAction('&About',self.about)
        self.menu.addSeparator()
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        self.pushButton.clicked.connect(self.showbtntext)
        self.pushButton_3.clicked.connect(self.setBrowerPath)
        self.pushButton_5.clicked.connect(self.setBrowerPath1)
        self.pushButton_8.clicked.connect(self.startTraining)
        self.pushButton_7.clicked.connect(self.changeMatrix)
        self.comboBox_2.currentIndexChanged.connect(self.selectionChange)
        self.comboBox_4.currentIndexChanged.connect(self.selectionMethod)
        self.comboBox_7.currentIndexChanged.connect(self.selectionTestdataset)
        self.comboBox_5.activated[str].connect(self.selectionFeature)
        
        self.horizontalSlider.valueChanged.connect(self.valChange)
        # self.horizontalSlider_2.valueChanged.connect(self.changeTraindatasetpre)
        self.tabWidget.setCurrentIndex(0)
        self.horizontalSlider.actionTriggered['int'].connect(self.lcdNumber.setDecMode)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def selectionTestdataset(self):
        if len(self.comboBox_6.currentText()) and len(self.comboBox_7.currentText()):
            gl.set_value('testrecognize',[[gl.get_value('data_path1')+'/'+self.comboBox_6.currentText()],[self.comboBox_7.currentText()]])
            printlog(self.textEdit,'testrecognize:'+self.comboBox_6.currentText()+','+self.comboBox_7.currentText(),fontcolor='chocolate')

    def selectionMethod(self):
        if len(self.comboBox_4.currentText()):
            gl.set_value('recognizemethod',self.comboBox_5.currentText()+'_'+self.comboBox_4.currentText())
            printlog(self.textEdit,'selectMethod:'+gl.get_value('recognizemethod'),fontcolor='chocolate')

    def selectionFeature(self,s):
        self.comboBox_4.clear()
        if s == "RT":
            self.comboBox_4.addItem('CNN')  
        elif s == "DT":
            self.comboBox_4.addItem('CNN')  
        elif s == "RT+DT+ART":
            self.comboBox_4.addItems(['CNN','CNN-LSTM'])  
        elif s == "ALL":
            self.comboBox_4.addItems(['CNN-LSTM','FeatureFusionNet']) 
        else:
            self.comboBox_4.addItem("请重选")
        # gl.set_value('recognizemethod',self.comboBox_4.currentText())
        # printlog(self.textEdit,self.comboBox_4.currentText(),fontcolor='yellow')

    def changeMatrix(self):
        self.con_matrix1.updatashow(np.random.random(size=(7,7)))
        # gl.set_value('file_name',filename)

    def selectionChange(self):
        filename = self.comboBox_2.currentText()
        # gl.set_value('file_name',filename)

    def showbtntext(self):
        if self.pushButton.isChecked():
            self.pushButton.setText("autoing...")
        else:
            self.pushButton.setText("AUTO")

    def setBrowerPath(self):
        download_path = QtWidgets.QFileDialog.getExistingDirectory(None,"打开文件夹","./")
        gl.set_value('data_path',download_path)
        self.lineEdit.setText(download_path)
        self.comboBox_2.clear()
        self.comboBox_2.addItem("--select--")
        list = []
        if (os.path.exists(download_path)):
            files = os.listdir(download_path)
            for file in files:
                m = os.path.join(download_path, file)
                if (os.path.isdir(m)):
                    h = os.path.split(m)
                    list.append(h[1])
        self.comboBox_2.addItems(list)

    

    def setBrowerPath1(self):
        download_path = QtWidgets.QFileDialog.getExistingDirectory(None,"打开文件夹","./")
        gl.set_value('data_path1',download_path)
        self.lineEdit_2.setText(download_path)
        list = []
        if (os.path.exists(download_path)):
            files = os.listdir(download_path)
            for file in files:
                m = os.path.join(download_path, file)
                if (os.path.isdir(m)):
                    h = os.path.split(m)
                    list.append(h[1])
        self.listView.clear()
        self.comboBox_6.clear()

        for data in sorted(list):
            self.comboBox_6.addItem(data)
            listWidget = ViolationItem(radioBtnname=data,download_path=download_path,textEditwidget = self.textEdit)
            listWidgetItem = QtWidgets.QListWidgetItem(self.listView)
            listWidgetItem.setSizeHint(QtCore.QSize(self.listView.width()-30, 40))
            self.listView.addItem(listWidgetItem)
            self.listView.setItemWidget(listWidgetItem, listWidget)

            

    def changeTraindatasetpre(self):
        self.lineEdit_3.setText(str(self.horizontalSlider_2.value()))

    def valChange(self):
        # print(self.horizontalSlider.value())
        self.lcdNumber.display(self.horizontalSlider.value())

    def startTraining(self):
    
        self.comboBox_6.setEnabled(False)
        self.comboBox_7.setEnabled(False)
        self.pushButton_9.setEnabled(False)    
        self.groupBox_4.setEnabled(False)
        self.pushButton_5.setEnabled(False)
        self.comboBox_5.setEnabled(False)
        self.comboBox_4.setEnabled(False)
        self.pushButton_8.setEnabled(False)
        # x = np.linspace(0, 100, 100)
        # y = np.random.random(100)

        # self.canvas.hl1.set_data([x,y])
        # self.canvas.draw()


    def about(self):
        QMessageBox.about(MainWindow, "About",
        """
        作者：Tkwer
        QQ:1837830365
        """
        )

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_3.setTitle(_translate("MainWindow", "File"))
        self.label_12.setText(_translate("MainWindow", "File:"))
        self.pushButton_5.setText(_translate("MainWindow", "open"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Train/Validate Dataset:"))
        self.label_15.setText(_translate("MainWindow", "Feature:"))
        self.comboBox_5.setItemText(0, _translate("MainWindow", "RT"))
        self.comboBox_5.setItemText(1, _translate("MainWindow", "DT"))
        self.comboBox_5.setItemText(2, _translate("MainWindow", "RT+DT+ART"))
        self.comboBox_5.setItemText(3, _translate("MainWindow", "ALL"))
        self.label_14.setText(_translate("MainWindow", "Method:"))
        self.label_23.setText(_translate("MainWindow", "color:"))
        self.pushButton_8.setText(_translate("MainWindow", "Start"))
        self.label_22.setText(_translate("MainWindow", "Test Dataset:"))
        self.pushButton_9.setText(_translate("MainWindow", "Recognize"))
        self.label_17.setText(_translate("MainWindow", "PrintLog"))
        self.label_16.setText(_translate("MainWindow", "Processing:"))
        self.pushButton_7.setText(_translate("MainWindow", "Save"))
        self.pushButton_6.setText(_translate("MainWindow", "EXIT"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Training data"))
        self.label_18.setText(_translate("MainWindow", "损失曲线"))
        self.label_19.setText(_translate("MainWindow", "平均准确率曲线"))
        self.label_20.setText(_translate("MainWindow", "训练混淆矩阵"))
        self.label_21.setText(_translate("MainWindow", "验证（测试）混淆矩阵"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Training"))
        self.groupBox_8.setTitle(_translate("MainWindow", "Radar_Data_Browse "))
        self.label.setText(_translate("MainWindow", "Range-Time Image"))
        self.label_11.setText(_translate("MainWindow", "Range-Doppler Image"))
        self.label_8.setText(_translate("MainWindow", "Estimate-Range Image"))
        self.label_10.setText(_translate("MainWindow", "Gesture_Type"))
        self.label_7.setText(_translate("MainWindow", "Doppler-Time Image"))
        self.label_9.setText(_translate("MainWindow", "Azimuth-Range Image"))
        self.groupBox.setTitle(_translate("MainWindow", "File"))
        self.label_6.setText(_translate("MainWindow", "File folder:"))
        self.pushButton_3.setText(_translate("MainWindow", "open"))
        self.label_2.setText(_translate("MainWindow", "Gesture_Type:"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "LEFT"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "RIGHT"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "FRONT"))
        self.comboBox_2.setItemText(3, _translate("MainWindow", "BACK"))
        self.label_4.setText(_translate("MainWindow", "Number:"))
        self.groupBox_10.setTitle(_translate("MainWindow", "Control"))
        self.pushButton.setText(_translate("MainWindow", "AUTO"))
        self.label_5.setText(_translate("MainWindow", "Frame:"))
        self.pushButton_2.setText(_translate("MainWindow", "Recognize"))
        self.pushButton_23.setText(_translate("MainWindow", "Dele"))
        self.pushButton_4.setText(_translate("MainWindow", "EXIT"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Visualize"))
        self.menu.setTitle(_translate("MainWindow", "菜单"))
        self.actionload.setText(_translate("MainWindow", "load"))
        self.actionload.setToolTip(_translate("MainWindow", "load"))
        self.actionload.setStatusTip(_translate("MainWindow", "你好"))
        self.actionload.setWhatsThis(_translate("MainWindow", "你好"))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    MainWindow.show()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    sys.exit(app.exec_())