import os
import torch
import numpy as np
from torch.utils.data import Dataset


# l = {0:'Back',1:'Dblclick',2:'Down',3:'Front',4:'Left',5:'Right',6:'Up',7:'Unkown'}

class loadedDataset(Dataset):
    def __init__(self, num_feature, root_dir, _ratio, dataset_type, transform=None):
        self.num_feature = num_feature
        self.root_dir = root_dir
        self.ratio = _ratio
        self.transform = transform
        self.dataset_type = dataset_type
        self.classes = [[] for x in range(len(self.root_dir))]
        self.count = [[] for x in range(len(self.root_dir))]
        self.acc_count = [[] for x in range(len(self.root_dir))]
        if len(self.root_dir)>0:
            for i,root_dir_ in enumerate(self.root_dir):
                self.classes[i] = sorted(os.listdir(root_dir_))
                self.count[i] = [len(os.listdir(root_dir_ + '/' + c)) for c in self.classes[i]]
                # 计数每一组（一组5个）特征
                self.count[i] = [int(x / self.num_feature * float(self.ratio[i])) for x in self.count[i]]
                for j in range(len(self.count[i])):
                    if i<1:
                        self.acc_count[i].append(np.sum(np.array(self.count[i][:j+1])))
                    else:
                        self.acc_count[i].append(self.acc_count[i-1][-1]+np.sum(np.array(self.count[i][:j+1])))
        else:
            print('目前enumerate()要求至少为2长')
        # self.acc_count = [self.count[i] + self.acc_count[i-1] for i in range(1, len(self.count))]

    def __len__(self):
        l = np.sum(np.sum(np.array(self.count)))
        return int(l)

    # idx 的值由 __len__方法返回所决定
    def __getitem__(self, idx):
        for j in range(len(self.root_dir)):
            for i in range(len(self.acc_count[j])):#类数
                if idx < self.acc_count[j][i]:
                    which_data = j
                    label = i
                    break
            else:
                continue
            break

        class_path = self.root_dir[which_data] + '/' + self.classes[which_data][label] 
        if self.dataset_type == 'train':
            if label:
                file_path = class_path + '/' + sorted(os.listdir(class_path))[idx-self.acc_count[which_data][label-1]]
                # file_path = class_path + '/' + sorted(os.listdir(class_path))[idx]
            elif which_data>0:
                file_path = class_path + '/' + sorted(os.listdir(class_path))[idx-self.acc_count[which_data-1][-1]]
            else:
                file_path = class_path + '/' + sorted(os.listdir(class_path))[idx]

            _, file_name = os.path.split(file_path)
        elif self.dataset_type == 'vali':
            Offset = int(self.count[which_data][label]/float(self.ratio[which_data])*(1-float(self.ratio[which_data])))
            if label:
                file_path = class_path + '/' + sorted(os.listdir(class_path))[Offset+idx-self.acc_count[which_data][label-1]]
                # file_path = class_path + '/' + sorted(os.listdir(class_path))[idx]
            elif which_data>0:
                file_path = class_path + '/' + sorted(os.listdir(class_path))[Offset+idx-self.acc_count[which_data-1][-1]]
            else:
                file_path = class_path + '/' + sorted(os.listdir(class_path))[Offset+idx]

            _, file_name = os.path.split(file_path)
        else:
            print('输入dataset_type字符错误！')
        # print("label:%10s,真实:%10s,特征:%12s,集合:%5s,位置:%5s"%(l[label],self.classes[which_data][label],file_name[:12],self.dataset_type,file_name[-9:-4]))
        ART_feature = np.load(file_path)
        ART_feature = np.float32(ART_feature)
        ART_feature = np.delete(ART_feature,[i for i in range(40,64)],axis=2)
        ART_feature = np.expand_dims(ART_feature,1).repeat(3,axis=1)
        ART_feature = torch.from_numpy(ART_feature)
        DT_feature = np.load(class_path +'/'+ 'DT' +file_name[-18:])
        DT_feature = np.float32(DT_feature)
        DT_feature = np.expand_dims(DT_feature,0).repeat(3,axis=0)
        DT_feature = torch.from_numpy(DT_feature)
        ERT_feature = np.load(class_path +'/'+ 'ERT' +file_name[-18:])
        ERT_feature = np.float32(ERT_feature)
        # ERT_feature = np.delete(ERT_feature,[88,89,90,91],axis=1)
        ERT_feature = np.delete(ERT_feature,[i for i in range(40,64)],axis=2)
        ERT_feature = np.expand_dims(ERT_feature,1).repeat(3,axis=1)
        ERT_feature = torch.from_numpy(ERT_feature)
        RDT_feature = np.load(class_path +'/'+ 'RDT' +file_name[-18:])
        RDT_feature = np.float32(RDT_feature)
        RDT_feature = np.expand_dims(RDT_feature,1).repeat(3,axis=1)
        RDT_feature = torch.from_numpy(RDT_feature)
        RT_feature = np.load(class_path +'/'+ 'RT' +file_name[-18:])
        RT_feature = np.float32(RT_feature)
        RT_feature = np.expand_dims(RT_feature,0).repeat(3,axis=0)
        RT_feature = torch.from_numpy(RT_feature)
        result = torch.LongTensor(label*np.ones(1))

        return ART_feature, DT_feature, ERT_feature, RDT_feature, RT_feature, result
