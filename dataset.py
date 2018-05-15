# -*- coding: utf-8 -*-
"""
Created on Tue May 15 15:19:50 2018

@author: spinbjy
"""

import torch
import torch.utils.data as Data
import os
import numpy as np

class MfccDataset(Data.Dataset):
    def __init__(self,datasetpath):
        self.audiodata = []
        self.label = []
        for name in os.listdir(datasetpath):
            for file in os.listdir(os.path.join(datasetpath,name)):
                data = np.loadtxt(os.path.join(datasetpath,name,file))
                self.audiodata.append(data)
                self.label.append(np.array(int(name)))
    def __getitem__(self,index):
        data = torch.from_numpy(self.audiodata[index])
        label = torch.from_numpy(self.label[index])
        return data,label
    def __len__(self):
        return len(self.audiodata)
