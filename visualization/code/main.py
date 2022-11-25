from app_interface import Ui_MainWindow
import time
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.ptime as ptime
import numpy as np
import sys
import os
import pyqtgraph as pg
import globalvar as gl
from cnn.test1 import CNet
import cnn.test1 as ctt
import sys
import os
from queue import Queue 
import matplotlib.pyplot as plt
import matplotlib.cm
from colortrans import pg_get_cmap
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from cnn.train import GetTrainINFO
from cnn.train import GetTestINFO

filepath = ""
filelist = [[] for i in range(5)]  # 创建的是多行三列的二维列表

# Queue for access data
print_data = Queue() 
traloss_data = Queue() 
valiloss_data = Queue() 
traacc_data = Queue() 
valiacc_data = Queue() 
tra_confusion = Queue() 
vali_confusion = Queue() 
test_confusion = Queue() 


train_validir = []
train_ratio = []

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


cnt = 0
def update_figure():
    global img_rdi, img_rai, img_rti, img_rei, img_dti, updateTime, cnt

    if autobtn.isChecked():
        Slider.setValue(cnt)
        cnt = (cnt +1)%12
        charge_frame()
        QtCore.QTimer.singleShot(200, update_figure)
    else:
        pass

def getcolor():
    values=matplotlib.cm.cmap_d.keys()
    color_.addItem("--select--")
    color_.addItem("customize")
    for value in values:
        color_.addItem(value)

def setcolor():
    if(color_.currentText()!='--select--' and color_.currentText()!=''):
        if color_.currentText() == 'customize':
            pgColormap = pg_get_cmap(color_.currentText())
        else:
            cmap=plt.cm.get_cmap(color_.currentText())
            pgColormap = pg_get_cmap(cmap)
        lookup_table = pgColormap.getLookupTable(0.0, 1.0, 256)
        img_rdi.setLookupTable(lookup_table)
        img_rai.setLookupTable(lookup_table)
        img_rti.setLookupTable(lookup_table)
        img_dti.setLookupTable(lookup_table)
        img_rei.setLookupTable(lookup_table)
        charge_frame()

def startTraining():
    global train_validir,train_ratio
    train_validir = gl.get_value('train_validir')
    train_ratio = gl.get_value('trian_ratio')
    if train_validir and gl.get_value('recognizemethod'):
        collector = GetTrainINFO('Listener', train_validir, train_ratio, print_data, traloss_data, traacc_data, valiloss_data, valiacc_data, tra_confusion, vali_confusion)
        collector.start()
        update_printf()
    else:
        tr_printlog('请检查训练路径和选中方法！')
        # # 可以重新训练
        # comboBox_6.setEnabled(True)
        # comboBox_7.setEnabled(True)
        # startTestbtn.setEnabled(True)
        # groupBox_4.setEnabled(True)
        # pushButton_5.setEnabled(True)
        # comboBox_5.setEnabled(True)
        # comboBox_4.setEnabled(True)
        # startTrainbtn.setEnabled(True)



def startTesting():
    testrecognize = gl.get_value('testrecognize')
    if testrecognize :
        collector1 = GetTestINFO('Listener1', testrecognize,  print_data, vali_confusion)
        collector1.start()
        update_printf1()
    else:
        tr_printlog('请检查路径和选中方法！')

def update_printf1():
    print_str = print_data.get()
    tr_printlog(print_str)
    if(bool(1-vali_confusion.empty())):
        valia_conf = vali_confusion.get()
        con_matrix2.updatashow(np.array(valia_conf))
    if (print_str!='----test over!----'):
        QtCore.QTimer.singleShot(20, update_printf1)
    else:
        canvas.hl1.set_data([],[])
        canvas.hl2.set_data([],[])
        canvas.draw()
        canvas1.hl1.set_data([],[])
        canvas1.hl2.set_data([],[])
        canvas1.draw()
        con_matrix1.updatashow(np.ones((7,7)))

def update_printf():
    print_str = print_data.get()
    tr_printlog(print_str)
    if(bool(1-traloss_data.empty())):
        traloss = traloss_data.get()
        canvas.hl1.set_data(range(0,len(traloss)),traloss)
        progressBar.setValue(len(traloss)*2)
        canvas.draw()
    if(bool(1-valiloss_data.empty())):
        valiloss = valiloss_data.get()
        canvas.hl2.set_data(range(0,len(valiloss)),valiloss)
        canvas.draw()
    if(bool(1-traacc_data.empty())):
        traacc = traacc_data.get()
        canvas1.hl1.set_data(range(0,len(traacc)),traacc)
        canvas1.draw()
    if(bool(1-valiacc_data.empty())):
        valiacc = valiacc_data.get()
        canvas1.hl2.set_data(range(0,len(valiacc)),valiacc)
        canvas1.draw()
    if(bool(1-tra_confusion.empty())):
        tra_conf = tra_confusion.get()
        con_matrix1.updatashow(np.array(tra_conf))

    if(bool(1-vali_confusion.empty())):
        valia_conf = vali_confusion.get()
        con_matrix2.updatashow(np.array(valia_conf))


    if (print_str!='----trian over!----'):
        QtCore.QTimer.singleShot(20, update_printf)
    else:
        # 可以重新训练

        comboBox_6.setEnabled(True)
        comboBox_7.setEnabled(True)
        startTestbtn.setEnabled(True)    
        groupBox_4.setEnabled(True)
        pushButton_5.setEnabled(True)
        comboBox_5.setEnabled(True)
        comboBox_4.setEnabled(True)
        startTrainbtn.setEnabled(True)



def openfile():
    pass

def charge_frame():
    global img_dti_data,img_rti_data,img_rdi_data,img_rai_data,img_rei_data
    i  = Spinbox.value()
    j = Slider.value()
    if len(filelist[0]):
        img_dti_data = np.load(filelist[0][i])
        img_rti_data = np.load(filelist[1][i])
        img_rdi_data = np.load(filelist[2][i])
        img_rai_data = np.load(filelist[3][i])
        img_rei_data = np.load(filelist[4][i])
        img_rti.setImage(img_rti_data, levels=[0, 1e4])
        img_rdi.setImage(img_rdi_data[j,:,:].T,levels=[2e4, 4e5])
        img_rei.setImage(img_rei_data[j,:,:].T,levels=[0, 6])
        img_dti.setImage(img_dti_data,levels=[0, 1000])
        img_rai.setImage(img_rai_data[j,:,:],levels=[0, 4])  

def selectionfile():
    global filelist,selcombox,gesture_filepath
    filepath = gl.get_value('data_path')
    if(selcombox.currentText()!='--select--' and filepath!='' and selcombox.currentText()!=''):
        gesture_filepath = filepath+'/'+selcombox.currentText()
        filelist = [[] for i in range(5)]
        filelist = get_filelist(gesture_filepath,filelist)
        Spinbox.setMaximum(len(filelist[0])-1)
        charge_frame()
    # print(len(filelist[0]))

def Recognition_Gesture():
    global gesture_filepath
    Spinbox.setFocus()
    fanhui = ctt.recognize1(net,[i[Spinbox.value()] for i in filelist])
    view_gesture.setPixmap(QtGui.QPixmap("visualization/gesture_icons/"+str(fanhui)+".jpg"))

def tr_printlog(string,fontcolor='blue'):
    global textEdit1
    textEdit1.moveCursor(QtGui.QTextCursor.End)
    gettime = time.strftime("%H:%M:%S", time.localtime())
    if string[0]=='-':
        fontcolor='red'
    textEdit1.append("<font color="+fontcolor+">"+str(gettime)+"-->"+string+"</font>")

def delfeatures():
    global filelist, gesture_filepath
    Spinbox.setFocus()
    i = Spinbox.value()
    if os.path.exists(filelist[0][i]):
        os.remove(filelist[0][i])
        os.remove(filelist[1][i])
        os.remove(filelist[2][i])
        os.remove(filelist[3][i])
        os.remove(filelist[4][i])
        filelist[0].remove(filelist[0][i])
        filelist[1].remove(filelist[1][i])
        filelist[2].remove(filelist[2][i])
        filelist[3].remove(filelist[3][i])
        filelist[4].remove(filelist[4][i])
        Spinbox.setMaximum(len(filelist[0])-1)
        charge_frame()

    else:
        pass


def application():
    global img_rdi, img_rai, img_rti, img_dti, img_rei,view_gesture, updateTime,autobtn,selcombox,Slider,Spinbox, progressBar,textEdit1,canvas,canvas1,color_,con_matrix1,con_matrix2
    global comboBox_4,comboBox_5,comboBox_6,comboBox_7,groupBox_4,pushButton_5,startTestbtn,startTrainbtn
    # ---------------------------------------------------
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    MainWindow.show()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    # 改了D:\Applications\anaconda3\Lib\site-packages\pyqtgraph\graphicsItems\ViewBox
    # 里的ViewBox.py第919行padding = self.suggestPadding(ax)改成padding = 0
    view_rdi = ui.graphicsView_6.addViewBox()
    ui.graphicsView_6.setCentralWidget(view_rdi)#去边界
    view_rai = ui.graphicsView_4.addViewBox()
    ui.graphicsView_4.setCentralWidget(view_rai)#去边界
    view_rti = ui.graphicsView.addViewBox()
    ui.graphicsView.setCentralWidget(view_rti)#去边界
    view_dti = ui.graphicsView_2.addViewBox()
    ui.graphicsView_2.setCentralWidget(view_dti)#去边界
    view_rei = ui.graphicsView_3.addViewBox()
    ui.graphicsView_3.setCentralWidget(view_rei)#去边界
    view_gesture = ui.graphicsView_5
    # view_gestures = ui.graphicsView_3
    view_gesture.setPixmap(QtGui.QPixmap("gesture_icons/7.jpg"))
    # view_gestures.setPixmap(QtGui.QPixmap("gesture_icons/8.jpg"))

    
    autobtn = ui.pushButton
    Recbtn = ui.pushButton_2
    openbtn = ui.pushButton_3
    exitbtn = ui.pushButton_4
    exitbtn1 = ui.pushButton_6
    startTrainbtn = ui.pushButton_8
    startTestbtn = ui.pushButton_9
    selcombox = ui.comboBox_2
    Slider = ui.horizontalSlider
    Spinbox = ui.spinBox
    textEdit1 = ui.textEdit
    canvas = ui.canvas
    canvas1 = ui.canvas1
    progressBar = ui.progressBar
    con_matrix1 = ui.con_matrix1
    con_matrix2 = ui.con_matrix2
    comboBox_4 = ui.comboBox_4
    comboBox_5 = ui.comboBox_5
    comboBox_6 = ui.comboBox_6
    comboBox_7 = ui.comboBox_7
    groupBox_4 = ui.groupBox_4
    pushButton_5 = ui.pushButton_5
    delbtn = ui.pushButton_23

    color_  = ui.comboBox

    
    img_rdi = pg.ImageItem(border=None)
    img_rai = pg.ImageItem(border=None)
    img_rti = pg.ImageItem(border=None)
    img_dti = pg.ImageItem(border=None)
    img_rei = pg.ImageItem(border=None)
    getcolor()

    pgColormap = pg_get_cmap('customize')

    lookup_table = pgColormap.getLookupTable(0.0, 1.0, 256)
    img_rdi.setLookupTable(lookup_table)
    img_rai.setLookupTable(lookup_table)
    img_rti.setLookupTable(lookup_table)
    img_dti.setLookupTable(lookup_table)
    img_rei.setLookupTable(lookup_table)

    view_rdi.addItem(img_rdi)
    view_rai.addItem(img_rai)
    view_rti.addItem(img_rti)
    view_dti.addItem(img_dti)
    view_rei.addItem(img_rei)


    updateTime = ptime.time()
    startTrainbtn.clicked.connect(startTraining)
    startTestbtn.clicked.connect(startTesting)
    autobtn.clicked.connect(update_figure)
    selcombox.currentIndexChanged.connect(selectionfile)
    Slider.valueChanged.connect(charge_frame)
    Spinbox.valueChanged.connect(charge_frame)
    Recbtn.clicked.connect(Recognition_Gesture)
    openbtn.clicked.connect(openfile)
    color_.currentIndexChanged.connect(setcolor)
    delbtn.clicked.connect(delfeatures)
    exitbtn1.clicked.connect(app.instance().exit)
    exitbtn.clicked.connect(app.instance().exit)
    
    app.instance().exec_()

if __name__ == '__main__':
    net = ctt.load_module()
    application()
    sys.exit()