from pathlib import Path

camelyon_cache_dir = '/scratch/ssd001/home/haoran/wilds_cache/camelyon'
wilds_root_dir = Path('/scratch/ssd001/home/haoran/wilds/')
camelyon_path = wilds_root_dir / 'camelyon/camelyon17_v1.0/'

train_N = {
    'camelyon': 15*4700,
    'CXR': 15*4700
}