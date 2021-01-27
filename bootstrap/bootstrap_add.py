import numpy as np
import sys 
import numpy.random as rand 
print(sys.executable)
print(sys.path)
from scipy.stats import bernoulli 
import pdb
import matplotlib.pyplot as plt 
import copy 
import pandas as pd 

def cb_frontdoor(index_n_labels,p,qyu,qzy,N): 
 
    pz_y = np.array([1-qzy, qzy]) # p(z/y) (correlated)
    
    if qyu<0:
        pu_y = np.array([-qyu, 1+qyu])
    else:    
        pu_y = np.array([1-qyu, qyu])
    
    lab_all = np.array(list(index_n_labels.values()))
    filename = np.array(list(index_n_labels.keys()))        
    
    N_bar = len(lab_all)

    # sampling U (guassian with variance 5)
    Y = rand.binomial(1,p,N)
    U = rand.binomial(1,pu_y[Y])
    Z = rand.binomial(1,pz_y[Y])
    # Y_all_de = copy.copy(Y_all)

    p = sum(Z)/N 
    Ns1 = int(p*N)
    Ns0 = N - Ns1 

    idn = list(np.where(Z==0)[0]) + list(np.where(Z==1)[0])
    idx = list(np.where(lab_all==0)[0][:Ns0]) + list(np.where(lab_all==1)[0][:Ns1])

    Y = Y[idn]; Z = Z[idn]; U = U[idn]
    
    Z = np.array(Z, dtype=int)
    Y = np.array(Y, dtype=int) ## to make sure that they can be used as indices in later part of the code
    yr = np.unique(Y)
    zr = np.unique(Z)
    ur = np.unique(U)

    ## Step 1: estimate f(u,y), f(y) and f(u|y)

    Nyz,_,_ = np.histogram2d(Y,Z,bins= [2,2])
    pyz_emp = Nyz/N 
    pz_emp = np.sum(pyz_emp, axis=0)
    py_emp = np.sum(pyz_emp, axis=1)
    pz_y_emp = np.transpose(pyz_emp)/py_emp 

    Nyu,_,_ = np.histogram2d(Y,U,bins=[len(yr),len(ur)])
    pyu_emp = Nyu/N 
    pu_emp = np.sum(pyu_emp, axis=0)
    py_emp = np.sum(pyu_emp, axis=1)
    py_u_emp = np.transpose(pyu_emp)/py_emp 

    ## estimate the f(z,y,u) to get f(z/y,u)  

    mat = np.array([Z,Y,U]).transpose(1,0)  
    H, [by, bu, bz]= np.histogramdd(mat,bins=[len(yr),len(ur),len(zr)])
    iz, iy, iu = np.where(H)
    pzyu_emp = H/N
    pu_emp = np.sum(np.sum(pzyu_emp, axis=0),axis=0)
    pz_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=1)
    py_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=0)
    pyu_emp = np.sum(pzyu_emp, axis=0)    
    pz_yu_emp = pzyu_emp/np.expand_dims(pyu_emp, axis=0)

    ## Step 2: for each y in range of values of Y variable  

    i = np.arange(0,N) # indices 
    k = 0
    w = np.zeros(N) # weights for the indices 
    i_new = [] 
    Y_new = []

    for m in range(len(yr)):

        j = np.where(Y==yr[m])[0]
        w = (pz_y_emp[Z,m]/pz_y_emp[Z,Y]/N)
        
        # Step 3: Resample Indices according to weight w 
        i_new = i_new + list(rand.choice(N,size=j.shape[0],replace=True,p=w))
        Y_new += [m]*j.shape[0]

    i_new.sort()

    idx = np.array(idx)    
    idx_new = idx[i_new]

    # confounded data 
    filename_conf = filename[idx]
    Y_conf = Y; Z_conf = Z; U_conf = U
    labels_conf = np.array([filename_conf, Y_conf, U_conf, Z_conf]).transpose(1,0)
    # new_vals_conf = np.array([Y_conf, U_conf, Z_conf], dtype=int).transpose(1,0)

    # unconfounded data 
    filename_deconf = filename[idx_new]
    Y_deconf = np.array(Y_new); Z_deconf = Z[i_new]; U_deconf = U[i_new]
    labels_deconf = np.array([filename_deconf, Y_deconf, U_deconf, Z_deconf]).transpose(1,0)
    # new_vals_deconf = np.array([Y_deconf, U_deconf, Z_deconf], dtype=int).transpose(1,0)

    # filenames = list(index_n_labels.keys())
    # labels_conf = dict(zip(filename_conf, new_vals_conf))
    # labels_deconf = dict(zip(filename_deconf, new_vals_deconf))
    
    return labels_conf, labels_deconf


def cb_front_n_back(index_n_labels,p,qyu,qzy,N): 

    pz_y = np.array([1-qzy, qzy]) # p(z/y) (correlated)
    
    if qyu<0:
        pu_y = np.array([-qyu, 1+qyu])
    else:    
        pu_y = np.array([1-qyu, qyu])
    
    lab_all = np.array(list(index_n_labels.values()))
    filename = np.array(list(index_n_labels.keys()))        
    
    N_bar = len(lab_all)

    # sampling U (guassian with variance 5)
    Y = rand.binomial(1,p,N)
    U = rand.binomial(1,pu_y[Y])
    Z = rand.binomial(1,pz_y[Y])
    # Y_all_de = copy.copy(Y_all)

    p = sum(Z)/N 
    Ns1 = int(p*N)
    Ns0 = N - Ns1 

    idn = list(np.where(Z==0)[0]) + list(np.where(Z==1)[0])
    idx = list(np.where(lab_all==0)[0][:Ns0]) + list(np.where(lab_all==1)[0][:Ns1])

    Y = Y[idn]; Z = Z[idn]; U = U[idn]
    
    Z = np.array(Z, dtype=int)
    Y = np.array(Y, dtype=int)
    U = np.array(U, dtype=int) ## to make sure that they can be used as indices in later part of the code
    
    yr = np.unique(Y)
    zr = np.unique(Z)
    ur = np.unique(U)

    ## Step 1: estimate f(z,y), f(y) and f(z|y)

    Nyz,_,_ = np.histogram2d(Y,Z,bins=[len(yr),len(zr)])
    pyz_emp = Nyz/N 
    pz_emp = np.sum(pyz_emp, axis=0)
    py_emp = np.sum(pyz_emp, axis=1) 
    pz_y_emp = np.transpose(pyz_emp)/py_emp

    Nyu,_,_ = np.histogram2d(Y,U,bins=[len(yr),len(ur)])
    pyu_emp = Nyu/N 
    pu_emp = np.sum(pyu_emp, axis=0)
    py_emp = np.sum(pyu_emp, axis=1)
    pu_y_emp = np.transpose(pyu_emp)/py_emp

    Nuz,_,_ = np.histogram2d(U,Z,bins=[len(ur),len(zr)])
    puz_emp = Nuz/N 
    pz_emp = np.sum(puz_emp, axis=0)
    pu_emp = np.sum(puz_emp, axis=1) 
    pz_u_emp = np.transpose(puz_emp)/pu_emp 
    ## estimate the f(z,y,u) to get f(z/y,u)  

    mat = np.array([Z,Y,U]).transpose(1,0)  
    H, [by, bu, bz]= np.histogramdd(mat,bins=[len(yr),len(ur),len(zr)])
    iz, iy, iu = np.where(H)
    pzyu_emp = H/N
    pu_emp = np.sum(np.sum(pzyu_emp, axis=0),axis=0)
    pz_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=1)
    py_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=0)
    pyu_emp = np.sum(pzyu_emp, axis=0)    
    pz_yu_emp = pzyu_emp/np.expand_dims(pyu_emp, axis=0)

    # pdb.set_trace()
    
    ## Step 2: for each y in range of values of Y variable  

    i = np.arange(0,N) # indices 
    k = 0
    w = np.zeros(N) # weights for the indices 
    i_new = [] 
    Y_new = []

    for m in range(len(yr)):

        j = np.where(Y==yr[m])[0]
        w = (pz_y_emp[Z,m]/(pz_u_emp[Z,U])/N) ## conditional distribution is done by taking samples from p(x,y,z) 
                                                                ## and normalising by p(y,u)
                                                                # 
        # print(sum(w))
        w = w/sum(w) ##  to renormalise 0.99999999976 
   
        # Step 3: Resample Indices according to weight w 
        i_new = i_new + list(rand.choice(N,size=j.shape[0],replace=True,p=w))
        Y_new += [m]*j.shape[0]

    # for m in range(len(yr)):

    #     j = np.where(Y==yr[m])[0]
    #     w = (pz_y_emp[Z,m]/(pzyu_emp[Z,Y,U]/pyu_emp[Y,U])/N) ## conditional distribution is done by taking samples from p(x,y,z) 
    #                                                             ## and normalising by p(y,u)
    #                                                             # 
    #     w = w/sum(w) ##  to renormalise 0.99999999976 
   
    #     # Step 3: Resample Indices according to weight w 
    #     i_new = i_new + list(rand.choice(N,size=j.shape[0],replace=True,p=w))
    #     Y_new += [m]*j.shape[0]

    i_new.sort()

    idx = np.array(idx)    
    idx_new = idx[i_new]

    # pdb.set_trace()

    # confounded data 
    filename_conf = filename[idx]
    Y_conf = Y; Z_conf = Z; U_conf = U
    labels_conf = np.array([filename_conf, Y_conf, U_conf, Z_conf]).transpose(1,0)
    # new_vals_conf = np.array([Y_conf, U_conf, Z_conf], dtype=int).transpose(1,0)

    # unconfounded data 
    filename_deconf = filename[idx_new]
    Y_deconf = np.array(Y_new); Z_deconf = Z[i_new]; U_deconf = U[i_new]
    labels_deconf = np.array([filename_deconf, Y_deconf, U_deconf, Z_deconf]).transpose(1,0)
    # new_vals_deconf = np.array([Y_deconf, U_deconf, Z_deconf], dtype=int).transpose(1,0)
    
    return labels_conf, labels_deconf

def cb_par_front_n_back(index_n_labels,p,qyu,qzy,N): 

    pz_y = np.array([1-qzy, qzy]) # p(z/y) (correlated)
    
    if qyu<0:
        pu_y = np.array([-qyu, 1+qyu])
        pv_y = np.array([1+qyu, -qyu])

    else:    
        pu_y = np.array([1-qyu, qyu])
        pv_y = np.array([qyu, 1-qyu])

    lab_all = np.array(list(index_n_labels.values()))
    filename = np.array(list(index_n_labels.keys()))        
    
    N_bar = len(lab_all)

    # sampling U (guassian with variance 5)
    Y = rand.binomial(1,p,N)
    U = rand.binomial(1,pu_y[Y])
    V = rand.binomial(1,pv_y[Y])
    Z = rand.binomial(1,pz_y[Y])
    # Y_all_de = copy.copy(Y_all)

    p = sum(Z)/N 
    Ns1 = int(p*N)
    Ns0 = N - Ns1 

    idn = list(np.where(Z==0)[0]) + list(np.where(Z==1)[0])
    idx = list(np.where(lab_all==0)[0][:Ns0]) + list(np.where(lab_all==1)[0][:Ns1])

    Y = Y[idn]; Z = Z[idn]; U = U[idn]; V = V[idn]
    
    Z = np.array(Z, dtype=int)
    Y = np.array(Y, dtype=int)
    U = np.array(U, dtype=int) ## to make sure that they can be used as indices in later part of the code
    V = np.array(V, dtype=int)

    yr = np.unique(Y)
    zr = np.unique(Z)
    ur = np.unique(U)
    vr = np.unique(V)

    ## Step 1: estimate f(z,y), f(y) and f(z|y)

    Nyz,_,_ = np.histogram2d(Y,Z,bins=[len(yr),len(zr)])
    pyz_emp = Nyz/N 
    pz_emp = np.sum(pyz_emp, axis=0)
    py_emp = np.sum(pyz_emp, axis=1)
    pz_y_emp = np.transpose(pyz_emp)/py_emp 

    ## estimate the f(z,y,u), f(y) and f()  
    mat = np.array([Z,Y,U]).transpose(1,0)  
    H, [by, bu, bz]= np.histogramdd(mat,bins=[len(yr),len(ur),len(zr)])
    iz, iy, iu = np.where(H)
    pzyu_emp = H/N
    pu_emp = np.sum(np.sum(pzyu_emp, axis=0),axis=0)
    pz_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=1)
    py_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=0)
    pyu_emp = np.sum(pzyu_emp, axis=0)    
    pz_yu_emp = pzyu_emp/np.expand_dims(pyu_emp, axis=0)

    # pdb.set_trace()
    
    ## Step 2: for each y in range of values of Y variable  
    i = np.arange(0,N) # indices 
    k = 0
    w = np.zeros(N) # weights for the indices 
    i_new = [] 
    Y_new = []

    for m in range(len(yr)):

        j = np.where(Y==yr[m])[0]
        w = (pz_y_emp[Z,m]/(pzyu_emp[Z,Y,U]/pyu_emp[Y,U])/N) ## conditional distribution is done by taking samples from p(x,y,z) 
                                                                ## and normalising by p(y,u) 
        
        w = w/sum(w) ##  to renormalise 0.99999999976 

        # Step 3: Resample Indices according to weight w 
        i_new = i_new + list(rand.choice(N,size=j.shape[0],replace=True,p=w))
        Y_new += [m]*j.shape[0]

    i_new.sort()

    idx = np.array(idx)    
    idx_new = idx[i_new]

    # confounded data 
    filename_conf = filename[idx]
    Y_conf = Y; Z_conf = Z; U_conf = U; V_conf = V 
    labels_conf = np.array([filename_conf, Y_conf, U_conf, Z_conf, V_conf]).transpose(1,0)
    # new_vals_conf = np.array([Y_conf, U_conf, Z_conf], dtype=int).transpose(1,0)

    # unconfounded data 
    filename_deconf = filename[idx_new]
    Y_deconf = np.array(Y_new); Z_deconf = Z[i_new]; U_deconf = U[i_new]; V_deconf = V[i_new]
    labels_deconf = np.array([filename_deconf, Y_deconf, U_deconf, Z_deconf, V_deconf]).transpose(1,0)
    # new_vals_deconf = np.array([Y_deconf, U_deconf, Z_deconf], dtype=int).transpose(1,0)

    # filenames = list(index_n_labels.keys())
    # labels_conf = dict(zip(filename_conf, new_vals_conf))
    # labels_deconf = dict(zip(filename_deconf, new_vals_deconf))
    
    return labels_conf, labels_deconf

def cb_backdoor(index_n_labels,p,qyu,N):

    # pdb.set_trace()
    # pu_y = np.array([1-qyu, qyu])

    if qyu<0:
        pu_y = np.array([-qyu, 1+qyu])
    else:    
        pu_y = np.array([1-qyu, qyu])
    
    lab_all = np.array(list(index_n_labels.values()))
    filename = np.array(list(index_n_labels.keys()))
    
    Y = rand.binomial(1,p,N)
    U = rand.binomial(1,pu_y[Y])
    
    Ns1 = int(N*p)
    Ns0 = N - Ns1  

    idn = list(np.where(Y==0)[0]) + list(np.where(Y==1)[0])
    idx = list(np.where(lab_all==0)[0][:Ns0]) + list(np.where(lab_all==1)[0][:Ns1])
    
    Y = Y[idn]; U = U[idn]

    U = np.array(U, dtype=int); Y = np.array(Y, dtype=int) ## to make sure that they can be used as indices in later part of the code
    yr = np.unique(Y)
    ur = np.unique(U)

    ## Step 1: estimate f(u,y), f(y) and f(u|y)

    Nyu,_,_ = np.histogram2d(Y,U, bins= [2,2])
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
        # Y_new += [m]*j.shape[0]

    i_new.sort()

    idx = np.array(idx, dtype=int)    
    idx_new = idx[i_new]
    
    # confounded data 
    filename_conf = filename[idx]
    Y_conf = Y; U_conf = U
    labels_conf = np.array([filename_conf, Y_conf, U_conf]).transpose(1,0)

    # unconfounded data 
    filename_deconf = filename[idx_new]
    Y_deconf = Y[i_new]; U_deconf = U[i_new]
    labels_deconf = np.array([filename_deconf, Y_deconf, U_deconf]).transpose(1,0)
    
    # pdb.set_trace()
    # part where I figured making dictonaries is making deconf case miss a lot of repeated samples that actually correct the probabilities 
    # new_vals_deconf = np.array([Y_deconf, U_deconf], dtype=int).transpose(1,0)
    # new_vals_deconf = np.array([Y_deconf, U_deconf], dtype=int).transpose(1,0)
    # labels_conf = dict(zip(filename_conf, new_vals_conf))
    # labels_deconf = dict(zip(filename_deconf, new_vals_deconf))
    
    return labels_conf, labels_deconf

def cb_label_flip(index_n_labels,p,qyu,qzu0,qzu1,N):

    lab_all = np.array(list(index_n_labels.values()))
    filename = np.array(list(index_n_labels.keys()))        
    
    N_bar = len(lab_all)

    Y = rand.binomial(1,p,N)
    
    if qyu<0:
        pu_y = np.array([-qyu, 1+qyu])
    else:    
        pu_y = np.array([1-qyu, qyu])
    U = rand.binomial(1,pu_y[Y])
     
    pz_yu = np.array([[1-qzu0, 1-qzu1], [qzu0, qzu1]])
    
    Z = rand.binomial(1,pz_yu[Y,U])

    p = sum(Y)/N 
    Ns1 = int(p*N)
    Ns0 = N - Ns1 

    idn = list(np.where(Y==0)[0]) + list(np.where(Y==1)[0])
    idx = list(np.where(lab_all==0)[0][:Ns0]) + list(np.where(lab_all==1)[0][:Ns1])

    Y = Y[idn]; Z = Z[idn]; U = U[idn]
    
    Z = np.array(Z, dtype=int)
    Y = np.array(Y, dtype=int)
    U = np.array(U, dtype=int) ## to make sure that they can be used as indices in later part of the code

    yr = np.unique(Y)
    zr = np.unique(Z)
    ur = np.unique(U)

    Nyu,_,_ = np.histogram2d(Y,U, bins= [len(yr),len(ur)])
    pyu_emp = Nyu/N 
    pu_emp = np.sum(pyu_emp, axis=0)
    py_emp = np.sum(pyu_emp, axis=1)
    py_u_emp = pyu_emp/pu_emp 

    mat = np.array([Z,Y,U]).transpose(1,0)  
    H, [by, bu, bz]= np.histogramdd(mat,bins=[len(yr),len(ur),len(zr)])
    iz, iy, iu = np.where(H)
    pzyu_emp = H/N
    pu_emp = np.sum(np.sum(pzyu_emp, axis=0),axis=0)
    pz_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=1)
    py_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=0)
    pyu_emp = np.sum(pzyu_emp, axis=0)    
    pz_yu_emp = pzyu_emp/np.expand_dims(pyu_emp, axis=0)

    i = np.arange(0,N) # indices 
    k = 0
    w = np.zeros(N) # weights for the indices 
    w1 = np.zeros(N) # weights for the indices 
    i_new = [] 
    Y_new = []

    # for m in range(len(yr)):
    #     j = np.where(Y==yr[m])[0]
        
    #     w = (((pz_yu_emp[Z,yr[m],U])*(Y==yr[m])/py_u_emp[yr[m],U])/N)
    #     w = w/sum(w)

    #     # Step 3: Resample Indices according to weight w 
    #     i_new = i_new + list(rand.choice(N,size=j.shape[0],replace=True,p=w))
    #     Y_new += [m]*j.shape[0]

    for m in range(len(yr)):
    
        j = np.where(Y==yr[m])[0]
        
        w = (((pz_yu_emp[0,yr[m],U])*(Y==yr[m])/py_u_emp[yr[m],U])/N) + (((pz_yu_emp[1,yr[m],U])*(Y==yr[m])/py_u_emp[yr[m],U])/N) 
    
        # Step 3: Resample Indices according to weight w 
        i_new = i_new + list(rand.choice(N,size=j.shape[0],replace=True,p=w))
        Y_new += [m]*j.shape[0]

    i_new.sort()

    idx = np.array(idx)    
    idx_new = idx[i_new]

    # confounded data 
    filename_conf = filename[idx]
    Y_conf = Y; Z_conf = Z; U_conf = U
    labels_conf = np.array([filename_conf, Y_conf, U_conf, Z_conf]).transpose(1,0)

    # unconfounded data 
    filename_deconf = filename[idx_new]
    Y_deconf = np.array(Y_new); Z_deconf = Z[i_new]; U_deconf = U[i_new]
    labels_deconf = np.array([filename_deconf, Y_deconf, U_deconf, Z_deconf]).transpose(1,0)

    ## sanity check 
    # Nyu_de,_,_ = np.histogram2d(Y_deconf,U_deconf, bins= [len(yr),len(ur)])
    # pyu_emp_de = Nyu_de/N 
    # pu_emp_de = np.sum(pyu_emp_de, axis=0)
    # py_emp_de = np.sum(pyu_emp_de, axis=1)
    # py_u_emp_de = pyu_emp_de/pu_emp_de
    # print(f"deconf correlations p(Y/U)\n: {py_u_emp_de}")

    # mat_de = np.array([Z_deconf,Y_deconf,U_deconf]).transpose(1,0)  
    # H_de, [by, bu, bz]= np.histogramdd(mat_de,bins=[len(yr),len(ur),len(zr)])
    # iz, iy, iu = np.where(H_de)
    # pzyu_emp_de = H_de/N
    # pu_emp_de = np.sum(np.sum(pzyu_emp_de, axis=0),axis=0)
    # pz_emp_de = np.sum(np.sum(pzyu_emp_de, axis=2),axis=1)
    # py_emp_de = np.sum(np.sum(pzyu_emp_de, axis=2),axis=0)
    # pyu_emp_de = np.sum(pzyu_emp_de, axis=0)    
    # pz_yu_emp_de = pzyu_emp_de/np.expand_dims(pyu_emp_de, axis=0)
    # print(f"deconf correlations p(Z/Y,U)\n: {pz_yu_emp_de}")

    # Nyz,_,_ = np.histogram2d(Y,Z,bins=[len(yr),len(zr)])
    # pyz_emp = Nyz/N 
    # pz_emp = np.sum(pyz_emp, axis=0)
    # py_emp = np.sum(pyz_emp, axis=1)
    # pz_y_emp = np.transpose(pyz_emp)/py_emp 
    # print(f"conf corr p(Z/Y)\n: {pz_y_emp}")

    # Nyz_de,_,_ = np.histogram2d(Y_deconf,Z_deconf,bins=[len(yr),len(zr)])
    # pyz_emp_de = Nyz_de/N 
    # pz_emp_de = np.sum(pyz_emp_de, axis=0)
    # py_emp_de = np.sum(pyz_emp_de, axis=1)
    # pz_y_emp_de = np.transpose(pyz_emp_de)/py_emp_de 
    # print(f"conf corr p(Z/Y)\n: {pz_y_emp_de}")

    return labels_conf, labels_deconf


def cb_eg2(index_n_labels,p,qyu,qzu0,qzu1,N): 
 
    if qyu<0:
        pu_y = np.array([-qyu, 1+qyu])
    else:    
        pu_y = np.array([1-qyu, qyu])
    
    lab_all = np.array(list(index_n_labels.values()))
    filename = np.array(list(index_n_labels.keys()))        
    
    N_bar = len(lab_all)

    # sampling U (guassian with variance 5)
    Y = rand.binomial(1,p,N)
    U = rand.binomial(1,pu_y[Y])
    
    # pz_yu = np.array([ [ [0.8, 0.95],[0.2,0.05]  ], [ [0.2,0.05], [0.8, 0.95]  ] ])
    ## keeping only  z=1 probabilities
    # pz_yu = np.array([[0.2,0.05], [0.8, 0.95]])
    pz_yu = np.array([[1-qzu0, 1-qzu1], [qzu0, qzu1]])
    
    # pdb.set_trace()  
    Z = rand.binomial(1,pz_yu[Y,U])
    # Y_all_de = copy.copy(Y_all)

    p = sum(Z)/N 
    Ns1 = int(p*N)
    Ns0 = N - Ns1 

    idn = list(np.where(Z==0)[0]) + list(np.where(Z==1)[0])
    idx = list(np.where(lab_all==0)[0][:Ns0]) + list(np.where(lab_all==1)[0][:Ns1])

    Y = Y[idn]; Z = Z[idn]; U = U[idn]
    
    Z = np.array(Z, dtype=int)
    Y = np.array(Y, dtype=int)
    U = np.array(U, dtype=int) ## to make sure that they can be used as indices in later part of the code
    
    # pdb.set_trace()

    yr = np.unique(Y)
    zr = np.unique(Z)
    ur = np.unique(U)

    ## Step 1: estimate f(z,y), f(y) and f(z|y)

    Nyz,_,_ = np.histogram2d(Y,Z,bins=[len(yr),len(zr)])
    pyz_emp = Nyz/N 
    pz_emp = np.sum(pyz_emp, axis=0)
    py_emp = np.sum(pyz_emp, axis=1) 
    pz_y_emp = np.transpose(pyz_emp)/py_emp

    Nyu,_,_ = np.histogram2d(Y,U,bins=[len(yr),len(ur)])
    pyu_emp = Nyu/N 
    pu_emp = np.sum(pyu_emp, axis=0)
    py_emp = np.sum(pyu_emp, axis=1)
    pu_y_emp = np.transpose(pyu_emp)/py_emp 

    ## estimate the f(z,y,u) to get f(z/y,u)  

    mat = np.array([Z,Y,U]).transpose(1,0)  
    H, [by, bu, bz]= np.histogramdd(mat,bins=[len(yr),len(ur),len(zr)])
    iz, iy, iu = np.where(H)
    pzyu_emp = H/N
    pu_emp = np.sum(np.sum(pzyu_emp, axis=0),axis=0)
    pz_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=1)
    py_emp = np.sum(np.sum(pzyu_emp, axis=2),axis=0)
    pyu_emp = np.sum(pzyu_emp, axis=0)    
    pz_yu_emp = pzyu_emp/np.expand_dims(pyu_emp, axis=0)

    # pdb.set_trace()
    
    ## Step 2: for each y in range of values of Y variable  

    i = np.arange(0,N) # indices 
    k = 0
    w = np.zeros(N) # weights for the indices 
    i_new = [] 
    Y_new = []

    for m in range(len(yr)):

        j = np.where(Y==yr[m])[0]
        w = (pz_yu_emp[Z,m,U]/(pz_yu_emp[Z,Y,U])/N) ## conditional distribution is done by taking samples from p(x,y,z) 
                                                                ## and normalising by p(y,u)
                                                                # 
        w = w/sum(w) ##  to renormalise 0.99999999976 
   
        # Step 3: Resample Indices according to weight w 
        i_new = i_new + list(rand.choice(N,size=j.shape[0],replace=True,p=w))
        Y_new += [m]*j.shape[0]

    i_new.sort()

    idx = np.array(idx)    
    idx_new = idx[i_new]

    # pdb.set_trace()
    # confounded data 
    filename_conf = filename[idx]
    Y_conf = Y; Z_conf = Z; U_conf = U
    labels_conf = np.array([filename_conf, Y_conf, U_conf, Z_conf]).transpose(1,0)
    # new_vals_conf = np.array([Y_conf, U_conf, Z_conf], dtype=int).transpose(1,0)

    # unconfounded data 
    filename_deconf = filename[idx_new]
    Y_deconf = np.array(Y_new); Z_deconf = Z[i_new]; U_deconf = U[i_new]
    labels_deconf = np.array([filename_deconf, Y_deconf, U_deconf, Z_deconf]).transpose(1,0)
    # new_vals_deconf = np.array([Y_deconf, U_deconf, Z_deconf], dtype=int).transpose(1,0)

    # filenames = list(index_n_labels.keys())
    # labels_conf = dict(zip(filename_conf, new_vals_conf))
    # labels_deconf = dict(zip(filename_deconf, new_vals_deconf))
    
    return labels_conf, labels_deconf


# if __name__ == "__main__":
#     # index_n_labels = split_n_label(split = 'train', pheno = "Atelectasis")
    
#     # lab_conf, lab_deconf = cb_backdoor(index_n_labels,p=0.1,qyu=0.95,N=14*5000)
#     splits = read_splits()
#     index_n_labels = split_n_label(splits, mask = 'train')

#     lab_conf, lab_deconf = cb_label_flip(index_n_labels,p=0.5,qzu0=0.8,qzu1=0.95,qyu=0.95,N=5000)

#     # lab_conf, lab_deconf = cb_frontdoor_type2(index_n_labels,p=0.7,qzu0=0.8,qzu1=0.9,qyu=0.95,N=5000)

#     # lab_conf, lab_deconf = cb_frontdoor(index_n_labels,p=0.5,qyu=0.95,N=5000)
#     # lab_conf, lab_deconf = cb_front_n_back(index_n_labels,p=0.1,qyu=0.95,qzu0=0.8,qzu1=0.9,N=14*5000)
#     pdb.set_trace()

    

