import torch
import os
import numpy as np
from PIL import Image
import Constants
from data import cxr_preprocess as preprocess
import pandas as pd
from torchvision import transforms
import pickle
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset

def get_dfs(envs = [], split = None, only_frontal = False):
    dfs = []
    for e in envs:
        func = preprocess.get_process_func(e)
        paths = Constants.df_paths[e]
        
        if split is not None:    
            splits = [split]
        else:
            splits = ['train', 'val', 'test']
            
        dfs += [func(pd.read_csv(paths[i]), only_frontal) for i in splits]   
        
    return pd.concat(dfs, ignore_index = True, sort = False).sample(frac=1) #shuffle        
        

def prepare_df_for_cb(df, positive_env = 'CXP'):
    df2 = df.copy()
    df2 = df2.rename(columns = {'path': 'filename', 'Atelectasis': 'label', 'env': 'conf'})
    df2['conf'] = (df2['conf'] == positive_env).astype(int)
    df2['label'] = (df2['label']).astype(int)
    return df2
    
def dataset_from_cb_output(orig_df, labels_gen, split, envs = ['MIMIC', 'CXP']):
    '''
    massages output from labels_gen (which is only filename, label, conf) into a more informative
        dataframe format to allow for generalized caching in dataloader    
    '''
    augmented_dfs = {i: {} for i in envs}
    for i in envs:
        assert(len(np.unique(labels_gen[:, 0])) == len(labels_gen))
        augmented_dfs[i][split] = orig_df[(orig_df.path.isin(labels_gen[:, 0])) & (orig_df.env == i)]    
    
    dataset = get_dataset(envs, split, only_frontal = False, imagenet_norm = True, augment = 1 if split == 'train' else 0, 
               cache = True, subset_label = 'Atelectasis', augmented_dfs = augmented_dfs)    
    
    return dataset
    
def get_dataset(envs = [], split = None, only_frontal = False, imagenet_norm = True, augment = 0, cache = True, subset_label = None,
               augmented_dfs = None):
      
    if split in ['val', 'test']:
        assert(augment in [0, -1])
    
    if augment == 1: # image augmentations
        image_transforms = [transforms.RandomHorizontalFlip(), 
                            transforms.RandomRotation(10),     
                            transforms.RandomResizedCrop(size = 224, scale = (0.75, 1.0)),
                        transforms.ToTensor()]
    elif augment == 0: 
        image_transforms = [transforms.ToTensor()]
    elif augment == -1: # only resize, just return a dataset with PIL images; don't ToTensor()
        image_transforms = []        
   
    if imagenet_norm and augment != -1:
        image_transforms.append(transforms.Normalize(Constants.IMAGENET_MEAN, Constants.IMAGENET_STD))             
    
    datasets = []
    for e in envs:
        func = preprocess.get_process_func(e)
        paths = Constants.df_paths[e]
        
        if split is not None:    
            splits = [split]
        else:
            splits = ['train', 'val', 'test']
            
        if augmented_dfs is not None: # use provided dataframes instead of loading 
            dfs = [augmented_dfs[e][i] for i in splits]
        else:            
            dfs = [func(pd.read_csv(paths[i]), only_frontal) for i in splits]            
            
        for c, s in enumerate(splits):
            cache_dir = Path(Constants.cache_dir)/ f'{e}_{s}/'
            cache_dir.mkdir(parents=True, exist_ok=True)
            datasets.append(AllDatasetsShared(dfs[c], transform = transforms.Compose(image_transforms)
                                      , split = split, cache = cache, cache_dir = cache_dir, subset_label = subset_label)) 
                
    if len(datasets) == 0:
        return None
    elif len(datasets) == 1:
        ds = datasets[0]
    else:
        ds = ConcatDataset(datasets)
        ds.dataframe = pd.concat([i.dataframe for i in datasets])
    
    return ds


class AllDatasetsShared(Dataset):
    def __init__(self, dataframe, transform=None, split = None, cache = True, cache_dir = '', subset_label = None):
        super().__init__()
        self.dataframe = dataframe
        self.dataset_size = self.dataframe.shape[0]
        self.transform = transform
        self.split = split
        self.cache = cache
        self.cache_dir = Path(cache_dir)
        self.subset_label = subset_label # (str) select one label instead of returning all Constants.take_labels

    def get_cache_path(self, cache_dir, meta):
        path = Path(meta['path'])
        if meta['env'] in ['PAD', 'NIH']:
            return cache_dir / (path.stem + '.pkl')
        elif meta['env'] in ['MIMIC', 'CXP']:
            return (cache_dir / '_'.join(path.parts[-3:])).with_suffix('.pkl')  
        
    def __getitem__(self, idx):
        item = self.dataframe.iloc[idx]
        cache_path = self.get_cache_path(self.cache_dir, item)
        
        if self.cache and cache_path.is_file():
            img, label, meta = pickle.load(cache_path.open('rb'))
        else:            
            img = np.array(Image.open(item["path"]))

            if img.dtype == 'int32':
                img = np.uint8(img/(2**16)*255)
            elif img.dtype == 'bool':
                img = np.uint8(img)
            else: #uint8
                pass

            if len(img.shape) == 2:
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2)            
            elif len(img.shape)>2:
                # print('different shape', img.shape, item)
                img = img[:,:,0]
                img = img[:, :, np.newaxis]
                img = np.concatenate([img, img, img], axis=2) 

            img = Image.fromarray(img)
            resize_transform = transforms.Resize(size = [224, 224])            
            img = transforms.Compose([resize_transform])(img)            

            label = torch.FloatTensor(np.zeros(len(Constants.take_labels), dtype=float))
            for i in range(0, len(Constants.take_labels)):
                if (self.dataframe[Constants.take_labels[i].strip()].iloc[idx].astype('float') > 0):
                    label[i] = self.dataframe[Constants.take_labels[i].strip()].iloc[idx].astype('float')

            meta = item.to_dict()
            
            if self.cache:
                pickle.dump((img, label, meta), cache_path.open('wb'))
        
        if self.transform is not None: # apply image augmentations after caching
            img = self.transform(img)
        
        if self.subset_label:
            label = int(label[Constants.take_labels.index(self.subset_label)])
                
        return img, label, meta
            

    def __len__(self):
        return self.dataset_size
