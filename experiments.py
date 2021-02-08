import numpy as np
    
def combinations(grid):
    keys = list(grid.keys())
    limits = [len(grid[i]) for i in keys]
    all_args = []
    
    index = [0]*len(keys)
    
    while True:
        args = {}
        for c, i in enumerate(index):
            key = keys[c]
            args[key] = grid[key][i]
        all_args.append(args)
        
        # increment index by 1
        carry = False
        index[-1] += 1
        ind = len(index) -1
        while ind >= 0:
            if carry:
                index[ind] += 1
            
            if index[ind] == limits[ind]:
                index[ind] = 0
                carry = True
            else:
                carry = False                 
            ind -= 1
       
        if carry:
            break
        
    return all_args
        
def get_hparams(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment].hparams()    


def get_script_name(experiment):
    if experiment not in globals():
        raise NotImplementedError
    return globals()[experiment].fname
 
    
#### write experiments here    
class Camelyon1:
    fname = 'train.py'
    
    @staticmethod
    def hparams():
        grid = {
           'type': ['back', 'front', 'back_front', 'label_flip'],
           'data': ['camelyon'],
           'data_type': ['Conf', 'Deconf', 'DA', 'IF'],
           'domains': ((2, 3),),           
           'corr-coff': list(np.linspace(0.65, 0.95, 4)),
           'seed': list(range(5)),
        }
        
        return combinations(grid)

class CXR1:
    fname = 'train.py'
    
    @staticmethod
    def hparams():
        grid = {
           'type': ['back', 'front', 'back_front', 'label_flip'],
           'data': ['CXR'],
           'data_type': ['Conf', 'Deconf', 'DA', 'IF'],    
           'corr-coff': list(np.linspace(0.65, 0.95, 4)),
           'seed': list(range(5)),
           'samples': [6500],
           '': ['--use_pretrained'] 
        }
        
        return combinations(grid)


class Poverty1:
    fname = 'train.py'
    
    @staticmethod
    def hparams():
        grid = {
           'type': ['back', 'front', 'back_front', 'label_flip'],
           'data': ['poverty'],
           'data_type': ['Conf', 'Deconf', 'DA', 'IF'],    
           'corr-coff': list(np.linspace(0.65, 0.95, 4)),
           'seed': list(range(5)),
           'samples': [300],
           'domains': (('malawi', 'kenya', 'tanzania', 'nigeria'),),    
        }
        
        return combinations(grid)
    
class EnvClf:
    fname = 'train_env.py'
    
    @staticmethod
    def hparams():
        grid = {
           'data': ['CXR', 'camelyon', 'poverty', 'NIH', 'MNIST', 'CelebA'],
           'seed': list(range(5)),
            '': ['--use_pretrained'] 
        }
        
        return combinations(grid)    