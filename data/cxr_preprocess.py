import Constants
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import pandas as pd
import torch
from pathlib import Path

def preprocess_MIMIC(split, only_frontal):    
    details = pd.read_csv(Constants.MIMIC_details)
    details = details.drop(columns=['dicom_id', 'study_id', 'religion', 'race', 'insurance', 'marital_status', 'gender'])
    details.drop_duplicates(subset="subject_id", keep="first", inplace=True)
    df = pd.merge(split, details)
    
    copy_sunbjectid = df['subject_id'] 
    df.drop(columns = ['subject_id'])
    
    df = df.replace(
            [[None], -1, "[False]", "[True]", "[ True]", 'UNABLE TO OBTAIN', 'UNKNOWN', 'MARRIED', 'LIFE PARTNER',
             'DIVORCED', 'SEPARATED', '0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90',
             '>=90'],
            [0, 0, 0, 1, 1, 0, 0, 'MARRIED/LIFE PARTNER', 'MARRIED/LIFE PARTNER', 'DIVORCED/SEPARATED',
             'DIVORCED/SEPARATED', '0-20', '0-20', '20-40', '20-40', '40-60', '40-60', '60-80', '60-80', '80-', '80-'])
    
    df['subject_id'] = copy_sunbjectid.astype(str)
    df['Age'] = df["age_decile"]
    df['Sex'] = df["gender"]
    df = df.drop(columns=["age_decile", 'gender'])
    df = df.rename(
        columns = {
            'Pleural Effusion':'Effusion',   
        })
    df['study_id'] = df['path'].apply(lambda x: x[x.index('p'):x.rindex('/')])
    df['path'] = Constants.image_paths['MIMIC'] + df['path'].astype(str)
    df['frontal'] = (df.view == 'frontal')
    if only_frontal:
        df = df[df.frontal]
        
    df['env'] = 'MIMIC'  
    df.loc[df.Age == 0, 'Age'] = '0-20'
    
    return df[['subject_id','path','Sex',"Age", 'env', 'frontal', 'study_id'] + Constants.take_labels]

def preprocess_NIH(split, only_frontal = True):
    split['Patient Age'] = np.where(split['Patient Age'].between(0,19), 19, split['Patient Age'])
    split['Patient Age'] = np.where(split['Patient Age'].between(20,39), 39, split['Patient Age'])
    split['Patient Age'] = np.where(split['Patient Age'].between(40,59), 59, split['Patient Age'])
    split['Patient Age'] = np.where(split['Patient Age'].between(60,79), 79, split['Patient Age'])
    split['Patient Age'] = np.where(split['Patient Age']>=80, 81, split['Patient Age'])
    
    copy_sunbjectid = split['Patient ID'] 
    split.drop(columns = ['Patient ID'])
    
    split = split.replace([[None], -1, "[False]", "[True]", "[ True]", 19, 39, 59, 79, 81], 
                            [0, 0, 0, 1, 1, "0-20", "20-40", "40-60", "60-80", "80-"])
   
    split['subject_id'] = copy_sunbjectid.astype(str)
    split['Sex'] = split['Patient Gender'] 
    split['Age'] = split['Patient Age']
    split = split.drop(columns=["Patient Gender", 'Patient Age'])
    split['path'] = Constants.image_paths['NIH'] + split['Image Index'].astype(str)
    split['env'] = 'NIH'
    split['frontal'] = True
    split['study_id'] = split['subject_id'].astype(str)
    return split[['subject_id','path','Sex',"Age", 'env', 'frontal','study_id'] + Constants.take_labels]


def preprocess_CXP(split, only_frontal):
    split['Age'] = np.where(split['Age'].between(0,19), 19, split['Age'])
    split['Age'] = np.where(split['Age'].between(20,39), 39, split['Age'])
    split['Age'] = np.where(split['Age'].between(40,59), 59, split['Age'])
    split['Age'] = np.where(split['Age'].between(60,79), 79, split['Age'])
    split['Age'] = np.where(split['Age']>=80, 81, split['Age'])
    
    copy_sunbjectid = split['subject_id'] 
    split.drop(columns = ['subject_id'])
    split = split.replace([[None], -1, "[False]", "[True]", "[ True]", 19, 39, 59, 79, 81], 
                            [0, 0, 0, 1, 1, "0-20", "20-40", "40-60", "60-80", "80-"])
    
    split['subject_id'] = copy_sunbjectid.astype(str)
    split['Sex'] = np.where(split['Sex']=='Female', 'F', split['Sex'])
    split['Sex'] = np.where(split['Sex']=='Male', 'M', split['Sex'])
    split = split.rename(
        columns = {
            'Pleural Effusion':'Effusion',
            'Lung Opacity': 'Airspace Opacity'        
        })
    split['path'] = Constants.image_paths['CXP'] + split['Path'].astype(str)
    split['frontal'] = (split['Frontal/Lateral'] == 'Frontal')
    if only_frontal:
        split = split[split['frontal']]
    split['env'] = 'CXP'
    split['study_id'] = split['path'].apply(lambda x: x[x.index('patient'):x.rindex('/')])
    return split[['subject_id','path','Sex',"Age", 'env', 'frontal','study_id'] + Constants.take_labels]


def preprocess_PAD(split, only_frontal):
    split['Age'] = np.where(split['Age'].between(0,19), 19, split['Age'])
    split['Age'] = np.where(split['Age'].between(20,39), 39, split['Age'])
    split['Age'] = np.where(split['Age'].between(40,59), 59, split['Age'])
    split['Age'] = np.where(split['Age'].between(60,79), 79, split['Age'])
    split['Age'] = np.where(split['Age']>=80, 81, split['Age'])
    
    split = split.replace([[None], -1, "[False]", "[True]", "[ True]", 19, 39, 59, 79, 81], 
                            [0, 0, 0, 1, 1, "0-20", "20-40", "40-60", "60-80", "80-"])
    
    split.loc[split['Age'] == 0.0, 'Age'] = '0-20'
    split = split.rename(columns = {
        'PatientID': 'subject_id',
        'StudyID': 'study_id',
        'PatientSex_DICOM' :'Sex'        
    })
    
    split.loc[~split['Sex'].isin(['M', 'F', 'O']), 'Sex'] = 'O'
    split['path'] =  split['ImageID'].astype(str).apply(lambda x: os.path.join(Constants.image_paths['PAD'], x))
    if only_frontal:
        split = split[split['frontal']]
    split['env'] = 'PAD'
    return split[['subject_id','path','Sex',"Age", 'env', 'frontal','study_id'] + Constants.take_labels]

def get_process_func(env):
    if env == 'MIMIC':
        return preprocess_MIMIC
    elif env == 'NIH':
        return preprocess_NIH
    elif env == 'CXP':
        return preprocess_CXP
    elif env == 'PAD':
        return preprocess_PAD
    else:
        raise NotImplementedError        