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
 
    
#### write experiments here    
class Camelyon1:
    @staticmethod
    def hparams():
        grid = {
           'data': ['camelyon'],
           'data_type': ['Conf', 'Deconf'],
           'domains': ((2, 3), (2, 4)),           
           'qzy': list(np.linspace(0.65, 0.95, 4)),
           'seed': list(range(5)),
        }
        
        return combinations(grid)
