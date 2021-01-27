import numpy as np
import sys 
import numpy.random as rand 
from scipy.stats import bernoulli 
import pdb
import matplotlib.pyplot as plt 
import copy 
import pandas as pd 

def unison_shuffled_copies(a, b, c):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p]

def cb_backdoor(index_n_labels,p,qyu,N):

    pu_y = np.array([qyu, 1-qyu]) 

    if qyu < 0: 
        pu_y = np.array([-qyu, 1+qyu]) 

    filename = np.array(index_n_labels['filename'].tolist())
    Y_all = np.array(index_n_labels['label'].tolist())
    U_all = np.array(index_n_labels['conf'].tolist())

    la_all = pd.DataFrame(data={'Y_all':Y_all, 'U_all':U_all})  
    
    Y = rand.binomial(1,p,N)
    U = rand.binomial(1,pu_y[Y])
    
    yr = np.unique(Y); ur = np.unique(U); 
    ur_r = np.unique(U_all); yr_r = np.unique(Y_all)
    la = pd.DataFrame(data={'Y':Y,'U':U})

    Ns = []; Ns_real = []; idn = []; idx = []
    for y in yr:
        for u in ur: 
            ns = len(la.index[(la['Y']==y) & (la['U']==u)].tolist())
            Ns.append(ns)
            idn += la.index[(la['Y']==y) & (la['U']==u)].tolist()
            Ns_real.append(len(la_all.index[(la_all['Y_all']==yr_r[y]) & (la_all['U_all']==ur_r[u])].tolist()))
            idx += la_all.index[(la_all['Y_all']==yr_r[y]) & (la_all['U_all']==ur_r[u])].tolist()[:ns]

    Y = Y[idn]; U = U[idn]
    U = np.array(U, dtype=int); Y = np.array(Y, dtype=int) ## to make sure that they can be used as indices in later part of the code
    
    ## Step 1: estimate f(u,y), f(y) and f(u|y)
    Nyu,_,_ = np.histogram2d(Y,U, bins = [len(yr),len(ur)])
    pyu_emp = Nyu/N 
    pu_emp = np.sum(pyu_emp, axis=0)
    py_emp = np.sum(pyu_emp, axis=1)
    py_u_emp = pyu_emp/pu_emp 

    ## Step 2: for each y in range of values of Y variable  
    i = np.arange(0,len(idx)) # indices 
    w = np.zeros(len(idx)) # weights for the indices 
    i_new = [] 

    for m in range(len(yr)):

        j = np.where(Y==yr[m])[0]
        w[j] = (((Y==yr[m])/py_u_emp[m,U])/N)[j]
      
        # Step 3: Resample Indices according to weight w 
        i_new = i_new + list(rand.choice(j,size=j.shape[0],replace=True,p=w[j]))

    i_new.sort()
    
    # Step 4: New indices for unbiased data 
    idx = np.array(idx, dtype=int)    
    idx_new = idx[i_new]
    
    # confounded data 
    filename_conf = filename[idx]
    Y_conf = Y; U_conf = U
    filename_conf,Y_conf,U_conf = unison_shuffled_copies(filename_conf,Y_conf,U_conf)
    labels_conf = np.array([filename_conf, Y_conf, U_conf]).transpose(1,0)

    # unconfounded data 
    filename_deconf = filename[idx_new]
    Y_deconf = Y[i_new]; U_deconf = U[i_new]
    filename_deconf,Y_deconf,U_deconf = unison_shuffled_copies(filename_deconf,Y_deconf,U_deconf)
    labels_deconf = np.array([filename_deconf, Y_deconf, U_deconf]).transpose(1,0)
    
    return labels_conf, labels_deconf