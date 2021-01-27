import os
import os.path as osp 
import torch
import pandas as pd
import numpy as np
import random
from scipy.misc import imread 
from sklearn.model_selection import StratifiedShuffleSplit

import pdb

import PIL
import matplotlib.pyplot as plt 
import torch 
from torch.utils.data import Dataset
from torchvision.transforms import transforms  
import skimage.io as sko 
from PIL import Image 

from tqdm import tqdm 
import multiprocessing as mp
from multiprocessing import Pool

def transform(x):
    trans = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
    return trans(x)

class Camelyon(Dataset):
    
    def __init__(self, labels, transform, img_pth = "/scratch/gobi2/sindhu/datasets/WILDS/camelyon/camelyon17_v1.0"):
        
        self.labels = labels
        self.img_pth = img_pth
        self.tras = transform
        
    def __len__(self):
        # return 600
        return len(self.labels)
    
    def load_img(self, idx):

        img_idx = self.labels[idx][0]
        img = imread(osp.join(self.img_pth, img_idx)) 

        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)
        if len(img.shape)>2:
            img = img[:,:,0]
            img = img[:, :, np.newaxis]
            img = np.concatenate([img, img, img], axis=2)

        img = PIL.Image.fromarray(img)
        
        # plt.imshow(img)
        # plt.savefig(f"sam_bef_{self.labels[idx][2]}_{self.labels[idx][1]}")        
        
        img = self.tras(img)
        
        # im_path = osp.join(f'sample_{self.labels[idx][2]}_{self.labels[idx][1]}.jpg')
        # print(f'{self.labels[idx][0]}')
        # print(f'{self.labels[idx][2]}_{self.labels[idx][1]}')
        # sko.imsave(im_path, img.permute(1,2,0).cpu().numpy())

        return img

    def __getitem__(self, idx):

        img = self.load_img(idx)
        lab = torch.tensor(int(self.labels[idx][1]))
        conf = int(self.labels[idx][2])

        return img, lab, conf  

## loading and splitting metadata according to convinience
def split_n_label(split, domains, data = 'camelyon', root_dir='/scratch/gobi2/sindhu/datasets/WILDS'):
    if data == 'camelyon': 
        split_dir = os.path.join(root_dir,'camelyon/camelyon17_v1.0',f'{split}.csv')
        metadata = pd.read_csv(split_dir)
        req_fields = ['filename', 'tumor', 'center']
        metadata = metadata[req_fields]
        metadata = metadata[metadata['center'].isin(domains)] 
        metadata = metadata.rename(columns={'center': 'conf', 'tumor':'label'})
        return metadata

## loading and splitting metadata according to convinience 
def split_train_test(train_ratio, root_dir='/scratch/gobi2/sindhu/datasets/WILDS'): 
    
    data_dir = os.path.join(root_dir, 'camelyon/camelyon17_v1.0')
    # read in metadata 
    metadata_df = pd.read_csv(
            os.path.join(data_dir, 'metadata.csv'),
            index_col=0,
            dtype={'patient': 'str'})
    
    # get filename 
    file_name = [
            f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x_cord}_y_{y_cord}.png'
            for patient, node, x_cord, y_cord in
            metadata_df.loc[:, ['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False, name=None)]
    
    # add filename to metadata info 
    metadata_df['filename'] =  pd.Series(file_name, index=metadata_df.index)

    # get splits and req information 
    domain_array = metadata_df['center'].unique().tolist()
    y_uni = metadata_df['tumor'].unique().tolist()

    train_ind = []; val_ind = []; test_ind = []     
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size= 1-train_ratio, random_state=1234)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=1234)

    import pdb; pdb.set_trace()
    # within each uni_y (0,1) split the domains equally in train, val and test 
    indices = []
    for y in y_uni:
        metadata_y_df = metadata_df.loc[metadata_df['tumor'] == y]
        idx_org = metadata_y_df.index.values
        indices.append(idx_org)
        train_idx, val_test_idx = list(sss1.split(metadata_y_df['center'].tolist(), metadata_y_df['center'].tolist()))[0]
        val_idx, test_idx = list(sss2.split(metadata_y_df.iloc[val_test_idx]['filename'].tolist(), metadata_y_df.iloc[val_test_idx]['center'].tolist()))[0]
        train_ind += list(idx_org[train_idx]); val_ind += list(idx_org[val_test_idx[val_idx]]); test_ind += list(idx_org[val_test_idx[test_idx]]) 
    random.shuffle(train_ind); random.shuffle(val_ind) ; random.shuffle(test_ind)
    split_ind = {"train": train_ind, "valid": val_ind , "test": test_ind}

    import pdb; pdb.set_trace()
    # saving the split data
    for sp in split_ind.keys():
        met_spl = metadata_df.iloc[split_ind[sp]]
        met_spl.to_csv(os.path.join(data_dir, f'{sp}.csv'), index=False)
