from pathlib import Path

camelyon_cache_dir = '/scratch/ssd001/home/haoran/wilds_cache/camelyon'
wilds_root_dir = Path('/scratch/ssd001/home/haoran/wilds/')
camelyon_path = wilds_root_dir / 'camelyon/camelyon17_v1.0/'

train_N = {
    'camelyon': 15*4700,
    'CXR': 5*4700
}

df_paths = {
    'MIMIC': {
        'train': "/scratch/hdd001/projects/ml4h/projects/mimic_access_required/MIMIC-CXR/laleh/new_split/8-1-1/new_train.csv",
        'val': "/scratch/hdd001/projects/ml4h/projects/mimic_access_required/MIMIC-CXR/laleh/new_split/8-1-1/new_valid.csv",
        'test': "/scratch/hdd001/projects/ml4h/projects/mimic_access_required/MIMIC-CXR/laleh/new_split/8-1-1/new_test.csv"        
    },
    'CXP':{
        'train': "/scratch/hdd001/projects/ml4h/projects/CheXpert/split/July19/new_train.csv",
        'val': "/scratch/hdd001/projects/ml4h/projects/CheXpert/split/July19/new_valid.csv",
        'test': "/scratch/hdd001/projects/ml4h/projects/CheXpert/split/July19/new_test.csv"
    },
    'NIH':{
        'train': "/scratch/hdd001/projects/ml4h/projects/NIH/split/July16/train.csv",
        'val': "/scratch/hdd001/projects/ml4h/projects/NIH/split/July16/valid.csv",
        'test': "/scratch/hdd001/projects/ml4h/projects/NIH/split/July16/test.csv"
    },
    'PAD':{
        'train': "/scratch/hdd001/projects/ml4h/projects/padchest/PADCHEST/haoran_split/train.csv",
        'val': "/scratch/hdd001/projects/ml4h/projects/padchest/PADCHEST/haoran_split/valid.csv",
        'test': "/scratch/hdd001/projects/ml4h/projects/padchest/PADCHEST/haoran_split/test.csv"            
    }
}

image_paths = {
    'MIMIC':  "/scratch/hdd001/projects/ml4h/projects/mimic_access_required/MIMIC-CXR/",
    'CXP': "/scratch/hdd001/projects/ml4h/projects/CheXpert/",
    'NIH': "/scratch/hdd001/projects/ml4h/projects/NIH/images/",
    'PAD': '/scratch/hdd001/projects/ml4h/projects/padchest/PADCHEST/images-224'
}

MIMIC_details = "/scratch/hdd001/projects/ml4h/projects/mimic_access_required/MIMIC-CXR/vin_new_split/8-1-1/mimic-cxr-metadata-detail.csv"
PAD_details = "/scratch/hdd001/projects/ml4h/projects/padchest/PADCHEST/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
cache_dir = '/scratch/ssd001/home/haoran/projects/IRM_Clinical/cache'

IMAGENET_MEAN = [0.485, 0.456, 0.406]         # Mean of ImageNet dataset (used for normalization)
IMAGENET_STD = [0.229, 0.224, 0.225]          # Std of ImageNet dataset (used for normalization)

take_labels = ['No Finding', 'Atelectasis', 'Cardiomegaly',  'Effusion',  'Pneumonia', 'Pneumothorax', 'Consolidation','Edema' ]