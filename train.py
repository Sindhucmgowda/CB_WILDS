import numpy as np
import argparse
import sys  
import pickle 
import time 
import os 
import logging 
from timeit import default_timer as timer 
import random 
import pdb 
import os.path as osp 
from pathlib import Path
import math
import json
import Constants

# from data.data_camelyon import Camelyon17Dataset
from data.data_cam import Camelyon, split_train_test, split_n_label, transform  
from data import data_cxr
from utils.logging import setup_logs
from src.training import train_step, snapshot, train_helper
from src.validation import validation, validation_helper 
from src.prediction import prediction_analysis, prediction_analysis_helper
from utils.early_stopping import EarlyStopping 
from utils.checkpointing import save_checkpoint, has_checkpoint, load_checkpoint
from utils.infinite_loader import StatefulSampler, InfiniteDataLoader

# from bootstrap.bootstrap_add import cb_backdoor, cb_frontdoor, cb_front_n_back, cb_par_front_n_back, cb_label_flip
from bootstrap.bootstrap_wilds import cb_backdoor, cb_frontdoor, cb_front_n_back, cb_label_flip
# from model.ful_model import Conv_confemb

## Torch
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
# from torch.utils import data
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.models as mod

run_name = "cb" + time.strftime("-%Y-%m-%d_%H_%M_%S")
print(run_name)

def cb(conf_type, index_n_labels, p, qyu, N, qzy = None, qzu0 = None, qzu1 = None):
    if conf_type == 'back':  
        labels_conf, labels_deconf = cb_backdoor(index_n_labels,p=p,
                                        qyu=qyu,
                                        N=N)        
    elif conf_type == 'front':
        assert(qzy is not None)
        labels_conf, labels_deconf = cb_frontdoor(index_n_labels,p=p,
                                        qyu=qyu,qzy= qzy,
                                        N=N)
    elif conf_type == 'back_front':
        assert(qzy is not None)
        labels_conf, labels_deconf = cb_front_n_back(index_n_labels,p=p,
                                        qyu=qyu,qzy= qzy,
                                        N=N)
    elif conf_type == 'par_back_front':
        assert(qzy is not None)
        labels_conf, labels_deconf = cb_par_front_n_back(index_n_labels,p=p,
                                        qyu=qyu,qzy= qzy,
                                        N=N)
    elif conf_type == 'label_flip':
        assert(qzu0 is not None and qzu1 is not None)
        labels_conf, labels_deconf = cb_label_flip(index_n_labels,p=p,
                                        qyu=qyu, qzu0=qzu0, qzu1=qzu1,
                                        N=N)
        
    return (labels_conf, labels_deconf)    
        

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Causal Bootstrapping')
    parser.add_argument('--type','-t', type = str, choices = ['back', 'front', 'back_front', 'label_flip'], required = True)
    parser.add_argument('--samples','-N', type = int, default=8000,required = False)
    parser.add_argument('--no-cuda','-g', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--output_dir','-l', type=str, required=True)
    
    parser.add_argument('--log-interval','-i', type=int, required=False, default=1)
    parser.add_argument('--epochs','-e', type=int, required=False, default=15)
    
    parser.add_argument('--data_type', choices = ['Conf', 'Deconf', 'DA'], required = True)

    # parser.add_argument('--conf-type','-ct',type=str, required=True, default='rot')
    # parser.add_argument('--conf-val','-cv', type=float, required=False, default=0.5)
    
    parser.add_argument('--corr-coff','-q', type=float, required=False, default=0.95)    
    parser.add_argument('--qzy',type=float, required=False, default=0.95)    # unused for backdoor
    parser.add_argument('--qzu0',type=float, required=False, default=0.80) 
    parser.add_argument('--qzu1',type=float, required=False, default=0.95)

    parser.add_argument('--data','-d', type=str, choices = ['camelyon', 'CXR'])
    parser.add_argument('--domains','-do', nargs = '+', type = int, default=[2,3], required=False)
    parser.add_argument('--batch-size','-b', type=int, default=32, required=False)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--es_patience', type=int, default=10) # *val_freq steps
    parser.add_argument('--val_freq', type=int, default=100)
    parser.add_argument('--use_pretrained', action = 'store_true')

    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'), comment=run_name)

    # to genertate train/val/test split - once generated and stored
    # split_train_test(train_ratio=0.8, root_dir='/scratch/gobi2/sindhu/datasets/WILDS')

    os.makedirs(args.output_dir, exist_ok=True)
    res_pth = os.path.join(args.output_dir, 'results') 
    os.makedirs(res_pth, exist_ok=True)
        
    with open(Path(res_pth)/'args.json', 'w') as outfile:
        json.dump(vars(args), outfile)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu") 
    print(device)
    
    global_timer = timer() # global timer
    logger = setup_logs(args.output_dir, run_name) # setup logs       
    batch_size = args.batch_size

    # Training samples (confounding and deconfounding)
    if args.data == 'camelyon':
        index_n_labels = split_n_label(split = 'train', domains = args.domains, data = args.data)
    elif args.data == 'CXR':
        df = data_cxr.get_dfs(envs = ['MIMIC', 'CXP'], split = 'train')
        index_n_labels = data_cxr.prepare_df_for_cb(df)
    
    # slightly wasting compute for DA since trained models will be the same for all corr_coff
    qyu_train = 0.5 if args.type == 'DA' else args.corr_coff 
    
    labels_conf, labels_deconf = cb(args.type, index_n_labels, p = 0.5, qyu = qyu_train, 
                                    N = Constants.train_N[args.data], qzy = args.qzy, 
                                   qzu0 = args.qzu0, qzu1 = args.qzu1)
    
    # Validation samples (confounding and deconfounding1)
    if args.data == 'camelyon':
        index_n_labels_v = split_n_label(split = 'valid', domains = args.domains, data = args.data)  
    elif args.data == 'CXR':
        df_v = data_cxr.get_dfs(envs = ['MIMIC', 'CXP'], split = 'val')
        index_n_labels_v = data_cxr.prepare_df_for_cb(df_v)
         
    labels_conf_v, labels_deconf_v = cb(args.type, index_n_labels_v, p = 0.5, qyu = qyu_train, 
                                    N = args.samples, qzy = args.qzy, 
                                   qzu0 = args.qzu0, qzu1 = args.qzu1)
    
    logger.info(f"sam: {args.samples}, epoch: {args.epochs}")
    
    train_type = args.data_type
        
    # Defining the Convolutional model 
    model_conv = mod.densenet121(pretrained=args.use_pretrained) 
    num_ftrs = model_conv.classifier.in_features
    model_conv.classifier = nn.Linear(num_ftrs, 2)
    model_conv = model_conv.to(device)

    # optimizer 
    optimizer = optim.Adam(model_conv.parameters(), lr = args.lr)         
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience=5, factor=0.1)

    ## load data of required kind using DataLoader and creating Dataclasses
    if args.data == 'camelyon':
        if train_type in ['Conf', 'DA']: 
            train_data = Camelyon(labels = labels_conf)
            valid_data = Camelyon(labels = labels_conf_v)
        elif train_type == 'Deconf': 
            train_data = Camelyon(labels = labels_deconf)
            valid_data = Camelyon(labels =  labels_deconf_v)      
    elif args.data == 'CXR':
        if train_type in ['Conf', 'DA']: 
            train_data = data_cxr.dataset_from_cb_output(df, labels_conf, split = 'train')
            valid_data = data_cxr.dataset_from_cb_output(df_v, labels_conf_v, split = 'val')
        elif train_type == 'Deconf': 
            train_data = data_cxr.dataset_from_cb_output(df, labels_deconf, split = 'train')
            valid_data = data_cxr.dataset_from_cb_output(df_v, labels_deconf_v, split = 'val')
        
    train_loader = InfiniteDataLoader(train_data, batch_size=batch_size, num_workers = 1)
    validation_loader = DataLoader(valid_data, batch_size=batch_size*2, shuffle=True) 
    
    es = EarlyStopping(patience = args.es_patience)             
    n_steps = args.epochs * (len(train_data) // batch_size)  
        
    if has_checkpoint():
        state = load_checkpoint()
        model_conv.load_state_dict(state['model_dict'])
        optimizer.load_state_dict(state['optimizer_dict'])
        exp_lr_scheduler.load_state_dict(state['scheduler_dict'])
        train_loader.sampler.load_state_dict(state['sampler_dict'])
        start_step = state['start_step']
        es = state['es']
        torch.random.set_rng_state(state['rng'])
        print("Loaded checkpoint at step %s" % start_step)
    else:
        start_step = 0          
        
    tr_losses, tr_accs = [], []
    n_steps = 30
    for step in range(start_step, n_steps):    
        if es.early_stop:
            break               
        data, target, _ = next(iter(train_loader))
       
        start_timer = timer()

        # Train and validate
        step_loss, step_acc = train_step(data, target, args, model_conv, writer, 
                                     device, optimizer, batch_size) 
        
        end_timer = timer()
        tr_losses.append(step_loss)
        tr_accs.append(step_acc)        
            
        if step % args.log_interval == 0:
            logger.info('Train Step: {} \tStep time: {:.4f}\tAccuracy: {:.4f}\tLoss: {:.6f}'.format(
                step, end_timer - start_timer, step_acc, step_loss))
            
        if step % args.val_freq == 0:
            val_acc, val_loss, val_AUC, val_conf_mat, val_F1_score = validation(args, 
                model_conv, device, validation_loader, batch_size)

            writer.add_scalar(f'Loss/train/step/{train_type}', np.mean(tr_losses), step)
            writer.add_scalar(f'Loss/valid/step/{train_type}', val_loss, step)
            writer.add_scalar(f'AUC/valid/step/{train_type}', val_AUC, step)
            writer.add_scalar(f'Accuracy/train/step/{train_type}', np.mean(tr_accs), step)
            
            tr_losses, tr_accs = [], []

            exp_lr_scheduler.step(val_AUC)

            es(-val_AUC, step , model_conv.state_dict(), Path(res_pth)/'model.pt')   
            
            save_checkpoint(model_conv, optimizer, exp_lr_scheduler,
                            train_loader.sampler.state_dict(train_loader._infinite_iterator), 
                            step+1, es, torch.random.get_rng_state())
        
    ## end 
    end_global_timer = timer()
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))
    logger.info(f'Training ends: {train_type}')

#     key_results = ["Conf Train: Conf Test","Conf Train: Unconf Test","Deconf Train: Conf Test","Deconf Train: Unconf Test"]
#     final_acc = dict.fromkeys(key_results,None)
#     metrics = dict.fromkeys(key_results, None)
    metrics = {}
    
    
    if args.data == 'camelyon':
        index_n_labels_t = split_n_label(split = 'test', domains = args.domains, data = args.data)
        index_n_labels_t_real = split_n_label(split = 'test', domains = [4], data = args.data)
    elif args.data == 'CXR':
        df = data_cxr.get_dfs(envs = ['MIMIC', 'CXP'], split = 'test')
        index_n_labels_t = data_cxr.prepare_df_for_cb(df)
        
        df_real = data_cxr.get_dfs(envs = ['NIH'], split = 'test')
        
    labels_t_real = index_n_labels_t_real.to_numpy()
        
    del(optimizer)

    keylist_test = ['Unconf', 'Conf', 'Reverse', 'Real']
    for test_type in keylist_test: 
        if test_type == 'Real':
            if args.data == 'camelyon':
                test_data = Camelyon(labels = labels_t_real)
            elif args.data == 'CXR':   
                test_data = data_cxr.dataset_from_cb_output(df_real, labels_t_real, split = 'test')
        else:
            if test_type == 'Conf':
                qyu = args.corr_coff
            elif test_type == 'Unconf':
                qyu = 0.5
            elif test_type == 'Reverse':
                qyu = 1- args.corr_coff

            labels_t, _ = cb(args.type, index_n_labels_t, p = 0.5, qyu = qyu, 
                                        N = args.samples, qzy = args.qzy, 
                                       qzu0 = args.qzu0, qzu1 = args.qzu1)    

            if args.data == 'camelyon':
                test_data = Camelyon(labels = labels_t)
            elif args.data == 'CXR':   
                test_data = data_cxr.dataset_from_cb_output(df, labels_t, split = 'test')
        
        test_loader = DataLoader(test_data, batch_size=batch_size*2, shuffle=True)

        logger.info(f'===> loading best model {train_type} for prediction')
        model_conv.load_state_dict(torch.load(Path(res_pth)/'model.pt'))

        logger.info(f'===> testing best {train_type} model on {test_type} for prediction')

        # test_acc,test_loss = prediction(args, model_conv, device, test_loader, batch_size)       
        test_acc, test_loss, AUC, conf_mat, F1_score, target_list, pred_list = prediction_analysis(args, model_conv, device, test_loader, batch_size)   
        
        metrics[f'{test_type} target'] = target_list
        metrics[f'{test_type} pred'] = pred_list
        metrics[f'{test_type} acc'] = test_acc
        metrics[f'{test_type} auc'] = AUC
        metrics[f'{test_type} conf_mat'] = conf_mat
        
    writer.flush()
    writer.close()
    logger.info("################## Success #########################")
    logger.info(f'Final AUC scores: {AUC}')

    with open(os.path.join(res_pth, 'final_results.p'), 'wb') as res: 
        pickle.dump(metrics, res)
        
    with open(os.path.join(res_pth, 'done'), 'w') as f:
        f.write('done')    