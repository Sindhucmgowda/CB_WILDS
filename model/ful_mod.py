import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as mod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
from tqdm import tqdm
import pdb
import numpy as np
import pandas as pd 
from torch.autograd import Variable
import pickle
import pdb

class Conv_conf_emb(nn.Module):

    def __init__(self,dataset):
        super(Conv_conf_emb, self).__init__()

        if dataset == 'CelebA':
            self.model_conv = mod.resnet18(pretrained=False) # (checking if using pretrained weights is causing the problem) 
            self.num_ftrs = self.model_conv.fc.in_features
            self.num_ftrs = self.num_ftrs + 1 ## to add the confounding 
            self.model_conv.fc = nn.Identity()
        
        elif dataset == 'NIH': 
            self.model_conv = mod.densenet121(pretrained=False) # (checking if using pretrained weights is causing the problem) 
            self.num_ftrs = self.model_conv.classifier.in_features
            self.num_ftrs =  self.num_ftrs + 1
            self.model_conv.classifier = nn.Identity()

        # self.model_conv_only = torch.nn.Sequential(*(list(self.model_conv.children())[:-1]))
        self.class_conf =  nn.Linear(self.num_ftrs,2)

    def forward(self,x,conf):

        conf = conf.unsqueeze(-1)
        img_conv_out = self.model_conv(x)
        img_conf = torch.cat((img_conv_out, conf), -1)
        out = self.class_conf(img_conf)

        return out

class Conv_conf_med_emb(nn.Module):

    def __init__(self,dataset):
        super(Conv_conf_med_emb, self).__init__()

        if dataset == 'CelebA':
            self.model_conv = mod.resnet18(pretrained=False) # (checking if using pretrained weights is causing the problem) 
            self.num_ftrs = self.model_conv.fc.in_features
            self.num_ftrs = self.num_ftrs + 2 ## to add the confounding 
            self.model_conv.fc = nn.Identity()
        
        elif dataset == 'NIH': 
            self.model_conv = mod.densenet121(pretrained=False) # (checking if using pretrained weights is causing the problem) 
            self.num_ftrs = self.model_conv.classifier.in_features
            self.num_ftrs =  self.num_ftrs + 2
            self.model_conv.classifier = nn.Identity()

        # self.model_conv_only = torch.nn.Sequential(*(list(self.model_conv.children())[:-1]))
        self.class_conf =  nn.Linear(self.num_ftrs,2)

    def forward(self,x,conf,med):

        conf = conf.unsqueeze(-1)
        med = med.unsqueeze(-1)
        img_conv_out = self.model_conv(x)
        img_conf = torch.cat((img_conv_out, conf, med), -1)
        out = self.class_conf(img_conf)

        return out