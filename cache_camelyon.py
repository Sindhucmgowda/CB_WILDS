#!/h/haoran/anaconda3/bin/python
import numpy as np
import argparse
import sys  
import pickle 
import time 
import os 
import logging 
import random 
import os.path as osp 
sys.path.append(os.getcwd())

from data.data_cam import Camelyon, split_train_test, split_n_label, transform   

parser = argparse.ArgumentParser()
parser.add_argument('--domain', type=int, choices = list(range(5)))
args = parser.parse_args()

for split in ['train', 'valid', 'test']:
    index_n_labels = split_n_label(split = split, domains = [args.domain], data = 'camelyon')
    dataset = Camelyon(labels = index_n_labels.to_numpy(), cache = True)
    for i in range(len(dataset)):
        dataset[i]        