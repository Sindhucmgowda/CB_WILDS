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
from model import resnet_multispectral

def get_model(dataset, causal_type, data_type, use_pretrained):
    if data_type == 'IF':
        if causal_type == 'back':
            n_extras = 1
        else:
            n_extras = 2
    else:
        n_extras = 0
        
    if dataset in ['camelyon', 'CXR']:
        return Conv_conf_emb(use_pretrained, n_extras)
    elif dataset == 'poverty':
        return resnet_multispectral.ResNet18(n_extras)
    

class Conv_conf_emb(nn.Module):
    def __init__(self, use_pretrained, num_extra):
        super(Conv_conf_emb, self).__init__()
        self.num_extra = num_extra
        self.model_conv = mod.densenet121(pretrained= use_pretrained)         
        self.num_ftrs = self.model_conv.classifier.in_features + num_extra
        self.model_conv.classifier = nn.Identity()
        self.class_conf =  nn.Linear(self.num_ftrs,2)

    def forward(self,x,*args):
        assert(len(args) <= 1) 
        img_conv_out = self.model_conv(x)
        if self.num_extra:
            assert(args[0].shape[1] == self.num_extra)        
            img_conv_out = torch.cat((img_conv_out, args[0]), -1)
        out = self.class_conf(img_conv_out)
        return out
