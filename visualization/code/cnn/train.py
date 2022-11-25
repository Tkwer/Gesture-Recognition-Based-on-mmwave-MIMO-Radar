from cProfile import label
import os
import shutil
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torchvision import transforms, utils
import torch.nn.functional as F
from torch.utils.data.dataset import T, Dataset

from cnn.dataset import loadedDataset
from cnn.model import F3FusionNet
from cnn.model import D23CNN
from cnn.model import FeatureFusionNet
from cnn.model import AMSoftmax
from cnn.utils import AverageMeter

from torch.autograd import Variable
import threading as th
import globalvar as gl
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtWidgets import QApplication
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description = 'Training')
parser.add_argument('--model', default='./save_model/', type=str, help = 'path to model')
parser.add_argument('--arch', default = 'FeatureFusionNet', help = 'model architecture')
parser.add_argument('--lstm-layers', default=1, type=int, help='number of lstm layers')
parser.add_argument('--hidden-size', default=128, type=int, help='output size of LSTM hidden layers')
parser.add_argument('--fc-size', default=64, type=int, help='size of fully connected layer before LSTM')					
parser.add_argument('--epochs', default=50, type=int, help='manual epoch number')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--lr-step', default=100, type=float, help='learning rate decay frequency')
parser.add_argument('--batch-size', default=10, type=int, help='mini-batch size')						
parser.add_argument('--workers', default=1, type=int, help='number of data loading workers')
args = parser.parse_args()

labeldict = {0:2, 1:0, 2:4, 3:2, 4:3, 5:3, 6:4}
loss1 = nn.CrossEntropyLoss()


class GetTestINFO(th.Thread):
    def __init__(self, name, testdir, print_quence, confusion):

        th.Thread.__init__(self, name=name)
        self.testdir = testdir
        self.print_quence = print_quence
        self.confusion = confusion


    def run(self):
        startTesting_(self.testdir,self.print_quence, self.confusion)
        self.print_quence.put('----test over!----')


def startTesting_(testInfo,print_queue,confusion):
    global train_loss,train_accuracy,validate_loss,validate_accuracy
    train_loss = []
    train_accuracy = []
    validate_loss = []
    validate_accuracy = []
    test_dataset = loadedDataset(5, testInfo[0],[1],'train')

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    args.arch = gl.get_value('testrecognize')[1][0]

    if os.path.exists(os.path.join(args.model, testInfo[1][0]+'checkpoint.pth.tar')):
        # load existing model
        model_info = torch.load(os.path.join(args.model, testInfo[1][0]+'checkpoint.pth.tar'))
        print_queue.put("==> loading existing model '{}' ".format(model_info['arch']))
        criterion = None
        if model_info['arch'] == "RT_CNN":
            model = D23CNN(len(test_dataset.classes[0]), 'RT', 1, args.fc_size)
        elif model_info['arch'] == "DT_CNN":
            model = D23CNN(len(test_dataset.classes[0]), 'DT', 1,  args.fc_size)
        elif model_info['arch'] == "RT+DT+ART_CNN":
            model = F3FusionNet(len(test_dataset.classes[0]), 'CNN', args.lstm_layers, args.hidden_size, args.fc_size)
        elif model_info['arch'] == "RT+DT+ART_CNN-LSTM":
            model = F3FusionNet(len(test_dataset.classes[0]), 'CNN-LSTM', args.lstm_layers, args.hidden_size, args.fc_size)
        elif model_info['arch'] == "ALL_CNN-LSTM":
            model = FeatureFusionNet(len(test_dataset.classes[0]), 'CNN-LSTM', args.lstm_layers, args.hidden_size, args.fc_size)
        elif model_info['arch'] == "ALL_FeatureFusionNet":
            model = FeatureFusionNet(len(test_dataset.classes[0]), 'FeatureFusionNet', args.lstm_layers, args.hidden_size, args.fc_size)
            criterion = AMSoftmax(20,len(test_dataset.classes[0])).cuda()
            criterion_info = torch.load(os.path.join('./save_model/', 'AMsoftmax.pkl'))
            criterion.load_state_dict(criterion_info.state_dict() )
        model.cuda()
        model.load_state_dict(model_info['state_dict'])
        # print('Total params: %.6fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
        print_queue.put("------Test------")
        # evaluate on validation set
        prec1, _,_, _, cn2 = validate(test_loader, len(test_dataset.classes[0]), model, criterion, print_queue)
        confusion.put(cn2)
        print_queue.put("------Test Result------")
        print_queue.put("------Top1 accuracy: {prec: .2f} %".format(prec=prec1))
        # print_queue.put("   Top5 accuracy: {prec: .2f} %".format(prec=prec5))
        print_queue.put("-----------------------------")

    else:
        print_queue.put("not exist model!")




class GetTrainINFO(th.Thread):
    def __init__(self, name, train_validir, train_ratio, print_quence, loss1, acc1, loss2, acc2, confusion1,confusion2):

        th.Thread.__init__(self, name=name)
        self.train_validir = train_validir
        self.train_ratio = train_ratio
        self.print_quence = print_quence
        self.loss1 = loss1
        self.loss2 = loss2
        self.acc1 = acc1
        self.acc2 = acc2
        self.confusion1 = confusion1
        self.confusion2 = confusion2
        

    def run(self):
        
        startTraining_(self.train_validir,self.train_ratio,self.print_quence, self.loss1, self.acc1, self.loss2, self.acc2, self.confusion1, self.confusion2)
        self.print_quence.put('----trian over!----')

def minmaxscaler(data):
    mean = data.min()
    var = data.max() 
    return (data - mean)/(var-mean)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join('./save_model/', filename))
    if is_best:
        shutil.copyfile(os.path.join('./save_model/', filename), './save_model/'+state['arch']+'_model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch):
    if not epoch % args.lr_step and epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# 更新混淆矩阵
def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix



def train(train_loader, model, classes, criterion, optimizer, epoch, print_queue):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()	# switch to train mode
    conf_matrix = torch.zeros(classes, classes)
    for i, tr_data in enumerate(train_loader):
        # input_var, target_var = Variable(inputs, requires_grad=True), Variable(target)
        ART_feature, DT_feature, ERT_feature, RDT_feature, RT_feature, target = tr_data
        for m in range(len(ART_feature)):
            ART_feature[m,...] = minmaxscaler(ART_feature[m,...])
            DT_feature = minmaxscaler(DT_feature)
            ERT_feature[m,...] = minmaxscaler(ERT_feature[m,...])
            RDT_feature[m,...] = minmaxscaler(RDT_feature[m,...])
            RT_feature = minmaxscaler(RT_feature)
        # input_var = Variable([ART_feature, DT_feature, ERT_feature, RDT_feature, RT_feature], requires_grad=True) 
        # input_var1 = Variable(ART_feature, requires_grad=True) 
        # input_var2 = Variable(DT_feature, requires_grad=True) 
        # input_var3 = Variable(ERT_feature, requires_grad=True) 
        # input_var4 = Variable(RDT_feature, requires_grad=True) 
        # input_var5 = Variable(RT_feature, requires_grad=True) 
        # target_var = Variable(target)
        input_var1 = Variable(ART_feature, requires_grad=True).cuda()
        input_var2 = Variable(DT_feature, requires_grad=True).cuda() 
        input_var3 = Variable(ERT_feature, requires_grad=True).cuda() 
        input_var4 = Variable(RDT_feature, requires_grad=True).cuda() 
        input_var5 = Variable(RT_feature, requires_grad=True).cuda() 
        target_var = Variable(target).cuda()
        # compute output
        output = model(input_var1,input_var2,input_var3,input_var4,input_var5)
        # output = output[:, -1, :]
        # zero the parameter gradients
        optimizer.zero_grad()

        if args.arch == "ALL_FeatureFusionNet":
            # output[0] is output,output[1] is attention
            target1 = target.squeeze().numpy()
            label1 = [labeldict[x] for x in target1]
            loss,output1 = criterion(output[0],target_var.squeeze())
            #添加第二标签
            

            loss = loss+loss1(output[1],torch.LongTensor(label1).cuda())
            loss = loss.requires_grad_()
            losses.update(loss.item()/10, 1)

            loss.backward()
            output = output1

        else:
            loss = criterion(output, target_var.squeeze())
            losses.update(loss.item(), 1)
            # compute gradient
            loss.backward()

        # compute accuracy
        prec1, prec5 = accuracy(output.data.cpu(), target, topk=(1, 4))#求tensor中某个dim的前k大或者前k小的值以及对应的index
        prediction = torch.max(output, 1)[1]
        conf_matrix = confusion_matrix(prediction, labels=target, conf_matrix=conf_matrix)	
        top1.update(prec1[0].item(), 1)
        top5.update(prec5[0].item(), 1)




        optimizer.step()

        print_queue.put('Epoch: [{0}][{1}/{2}]\t'
            'lr {lr:.5f}\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Top1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            epoch, i, len(train_loader),
            lr=optimizer.param_groups[-1]['lr'],
            loss=losses,
            top1=top1))
    
    train_loss.append(losses.avg)
    train_accuracy.append(top1.avg/100)
    return (train_loss, train_accuracy, conf_matrix)

def validate(val_loader, classes, model, criterion,print_queue):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    conf_matrix = torch.zeros(classes, classes)
    for i, tr_data in enumerate(val_loader):
        # input_var, target_var = Variable(inputs, requires_grad=True), Variable(target)
        ART_feature, DT_feature, ERT_feature, RDT_feature, RT_feature, target = tr_data
        for m in range(len(ART_feature)):
            ART_feature[m,...] = minmaxscaler(ART_feature[m,...])
            DT_feature = minmaxscaler(DT_feature)
            ERT_feature[m,...] = minmaxscaler(ERT_feature[m,...])
            RDT_feature[m,...] = minmaxscaler(RDT_feature[m,...])
            RT_feature = minmaxscaler(RT_feature)
        # input_var = Variable([ART_feature, DT_feature, ERT_feature, RDT_feature, RT_feature], requires_grad=True) 
        # input_var1 = Variable(ART_feature, requires_grad=True) 
        # input_var2 = Variable(DT_feature, requires_grad=True) 
        # input_var3 = Variable(ERT_feature, requires_grad=True) 
        # input_var4 = Variable(RDT_feature, requires_grad=True) 
        # input_var5 = Variable(RT_feature, requires_grad=True) 
        # target_var = Variable(target)
        input_var1 = Variable(ART_feature, requires_grad=True).cuda()
        input_var2 = Variable(DT_feature, requires_grad=True).cuda() 
        input_var3 = Variable(ERT_feature, requires_grad=True).cuda() 
        input_var4 = Variable(RDT_feature, requires_grad=True).cuda() 
        input_var5 = Variable(RT_feature, requires_grad=True).cuda() 
        target_var = Variable(target).cuda()		
        # compute output
        with torch.no_grad():
            output = model(input_var1,input_var2,input_var3,input_var4,input_var5)
            # output = output[:, -1, :]
            if criterion !=None:
                if args.arch == "ALL_FeatureFusionNet":
                    # output[0] is output,output[1] is attention
                    # target1 = target.squeeze().numpy()
                    #添加第二标签
                    # label1 = [labeldict[x] for x in target1]
                    loss,output1 = criterion(output[0],target_var.squeeze(axis=1))
                    # loss = loss+loss1(output[1],torch.LongTensor(label1).cuda())
                    # loss = loss.requires_grad_()
                    losses.update(loss.item()/10, 1)
                    output = output1

                else:
                    loss = criterion(output, target_var.squeeze())
                    losses.update(loss.item(), 1)
                    # compute gradient

        # compute accuracy
        prec1, prec5 = accuracy(output.data.cpu(), target, topk=(1, 4))
        prediction = torch.max(output, 1)[1]
        conf_matrix = confusion_matrix(prediction, labels=target, conf_matrix=conf_matrix)		
        top1.update(prec1[0].item(), 1)
        top5.update(prec5[0].item(), 1)

        print_queue.put('Test: [{0}/{1}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Top1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader),
                loss=losses,
                top1=top1))
    validate_loss.append(losses.avg)
    validate_accuracy.append(top1.avg/100)

    return (top1.avg, top5.avg, validate_loss, validate_accuracy, conf_matrix)




def startTraining_(train_validir,train_ratio,print_queue,loss1,acc1,loss2,acc2,con1,con2):
    global train_loss,train_accuracy,validate_loss,validate_accuracy
    train_loss = []
    train_accuracy = []
    validate_loss = []
    validate_accuracy = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.arch = gl.get_value('recognizemethod')
    # print(args.arch)
    print_queue.put("Device being used: "+str(device))
    # Data Transform and data loading

    train_dataset = loadedDataset(5, train_validir,train_ratio,'train')
    val_dataset = loadedDataset(5, train_validir,[1-x for x in train_ratio],'vali')

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    

    if os.path.exists(os.path.join(args.model, args.arch+'checkpoint.pth.tar')):
        # load existing model
        model_info = torch.load(os.path.join(args.model, args.arch+'checkpoint.pth.tar'))
        print_queue.put("==> loading existing model '{}' ".format(model_info['arch']))
        if args.arch == "RT_CNN":
            model = D23CNN(len(train_dataset.classes[0]), 'RT', 1, args.fc_size)
        elif args.arch == "DT_CNN":
            model = D23CNN(len(train_dataset.classes[0]), 'DT', 1,  args.fc_size)
        elif args.arch == "RT+DT+ART_CNN":
            model = F3FusionNet(len(train_dataset.classes[0]), 'CNN', args.lstm_layers, args.hidden_size, args.fc_size)
        elif args.arch == "RT+DT+ART_CNN-LSTM":
            model = F3FusionNet(len(train_dataset.classes[0]), 'CNN-LSTM', args.lstm_layers, args.hidden_size, args.fc_size)
        elif args.arch == "ALL_CNN-LSTM":
            model = FeatureFusionNet(len(train_dataset.classes[0]), 'CNN-LSTM', args.lstm_layers, args.hidden_size, args.fc_size)
        elif args.arch == "ALL_FeatureFusionNet":
            model = FeatureFusionNet(len(train_dataset.classes[0]), 'FeatureFusionNet', args.lstm_layers, args.hidden_size, args.fc_size)
        model.cuda()
        model.load_state_dict(model_info['state_dict'])
        best_prec = model_info['best_prec']
        cur_epoch = model_info['epoch']
    else:
        if not os.path.isdir(args.model):
            os.makedirs(args.model)
        # load and create model
        print_queue.put("==> creating model '{}' ".format(args.arch))
        if args.arch == "RT_CNN":
            model = D23CNN(len(train_dataset.classes[0]), 'RT', 1, args.fc_size)
        elif args.arch == "DT_CNN":
            model = D23CNN(len(train_dataset.classes[0]), 'DT', 1,  args.fc_size)
        elif args.arch == "RT+DT+ART_CNN":
            model = F3FusionNet(len(train_dataset.classes[0]), 'CNN', args.lstm_layers, args.hidden_size, args.fc_size)
        elif args.arch == "RT+DT+ART_CNN-LSTM":
            model = F3FusionNet(len(train_dataset.classes[0]), 'CNN-LSTM', args.lstm_layers, args.hidden_size, args.fc_size)
        elif args.arch == "ALL_CNN-LSTM":
            model = FeatureFusionNet(len(train_dataset.classes[0]), 'CNN-LSTM', args.lstm_layers, args.hidden_size, args.fc_size)
        elif args.arch == "ALL_FeatureFusionNet":
            model = FeatureFusionNet(len(train_dataset.classes[0]), 'FeatureFusionNet', args.lstm_layers, args.hidden_size, args.fc_size)
            criterion = AMSoftmax(20,len(train_dataset.classes[0])).cuda()
        # print(model)
        model.cuda()
        cur_epoch = 0
        #0.28M大小

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

    print_queue.put('Total params: %.6fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # optimizer = torch.optim.Adam([{'params': model.fc_pre1.parameters()},
    # 								{'params': model.fc_pre3.parameters()},
    # 								{'params': model.fc_pre4.parameters()},
    # 							{'params': model.rnn.parameters()},
    # 							{'params': model.fc.parameters()}],
    # 							lr=args.lr)

    best_prec = 0
# Training on epochs
    for epoch in range(cur_epoch, args.epochs):

        optimizer = adjust_learning_rate(optimizer, epoch)

        print_queue.put("------Training------")

        # train on one epoch
        t_loss, t_acc, cn1 = train(train_loader, model, len(train_dataset.classes[0]), criterion, optimizer, epoch, print_queue)
        loss1.put(t_loss)
        acc1.put(t_acc)
        con1.put(cn1)

        print_queue.put("------Validation------")

        # evaluate on validation set
        prec1, prec5,v_loss, v_acc, cn2 = validate(val_loader, len(train_dataset.classes[0]), model, criterion, print_queue)
        loss2.put(v_loss)
        acc2.put(v_acc)
        con2.put(cn2)
        print_queue.put("------Validation Result------")
        print_queue.put("------Top1 accuracy: {prec: .2f} %".format(prec=prec1))
        # print_queue.put("   Top5 accuracy: {prec: .2f} %".format(prec=prec5))
        print_queue.put("-----------------------------")


        # remember best top1 accuracy and save checkpoint
        is_best = prec1 > best_prec
        best_prec = max(prec1, best_prec)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'num_classes': len(train_dataset.classes[0]),
            'lstm_layers': args.lstm_layers,
            'hidden_size': args.hidden_size,
            'fc_size': args.fc_size,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer' : optimizer.state_dict(),}, is_best,filename=args.arch+'checkpoint.pth.tar')
        if args.arch == "ALL_CNN-LSTM":
            torch.save(model, os.path.join('./save_model/', 'model.pkl'))
        if args.arch == "ALL_FeatureFusionNet":
            torch.save(criterion, os.path.join('./save_model/', 'AMsoftmax.pkl'))