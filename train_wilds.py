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

# from data.data_camelyon import Camelyon17Dataset
from data.data_cam import Camelyon, split_train_test, split_n_label, transform   
from utils.logging import setup_logs
from src.training import train, snapshot, train_helper
from src.validation import validation, validation_helper 
from src.prediction import prediction_analysis, prediction_analysis_helper
from utils.early_stopping import EarlyStopping 

# from bootstrap.bootstrap_add import cb_backdoor, cb_frontdoor, cb_front_n_back, cb_par_front_n_back, cb_label_flip
from bootstrap.bootstrap_wilds import cb_backdoor
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

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Causal Bootstrapping')
    parser.add_argument('--type','-t', type = str, default='back',required = False)
    parser.add_argument('--samples','-N', type = int, default=4000,required = False)
    parser.add_argument('--no-cuda','-g', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--output_dir','-l', type=str, required=True)
    
    parser.add_argument('--log-interval','-i', type=int, required=False, default=1)
    parser.add_argument('--epochs','-e', type=int, required=False, default=15)
    
    parser.add_argument('--data_type', choices = ['Conf', 'Deconf'], required = True)

    # parser.add_argument('--conf-type','-ct',type=str, required=True, default='rot')
    # parser.add_argument('--conf-val','-cv', type=float, required=False, default=0.5)
    
    parser.add_argument('--qzy',type=float, required=False, default=0.95)
    parser.add_argument('--corr-coff','-q', type=float, required=False, default=0.95) # unused for backdoor    
    parser.add_argument('--qzu0',type=float, required=False, default=0.80) 
    parser.add_argument('--qzu1',type=float, required=False, default=0.95)

    parser.add_argument('--data','-d', type=str, default= 'camelyon', required=False)
    parser.add_argument('--domains','-do', nargs = '+', type = int, default=[2,3], required=False)
    parser.add_argument('--batch-size','-b', type=int, default=64, required=False)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--es_patience', type=int, default=5) # *val_freq steps
    parser.add_argument('--val_freq', type=int, default=200)
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
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu") 
    print(device)
    
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    global_timer = timer() # global timer
    logger = setup_logs(args.output_dir, run_name) # setup logs
    
    ## for data augmentation (DA) scenario (only train in the confounded case)    
    keylist_test = ['Unconf', 'Conf']
    batch_size = args.batch_size

    index_n_labels = split_n_label(split = 'train', domains = args.domains, data = 'camelyon')

    # Training samples (confounding and deconfounding)
    if args.type == 'back':  
        labels_conf, labels_deconf = cb_backdoor(index_n_labels,p=0.5,
                                        qyu=args.corr_coff,
                                        N=15*4700)        
    elif args.type == 'front':
        labels_conf, labels_deconf = cb_frontdoor(index_n_labels,p=0.5,
                                        qyu=args.corr_coff,qzy= args.qzy,
                                        N=15*4700)
    elif args.type == 'back_front':
        labels_conf, labels_deconf = cb_front_n_back(index_n_labels,p=0.5,
                                        qyu=args.corr_coff,qzy= args.qzy,
                                        N=15*4700)
    elif args.type == 'par_back_front':
        labels_conf, labels_deconf = cb_par_front_n_back(index_n_labels,p=0.5,
                                        qyu=args.corr_coff,qzy= args.qzy,
                                        N=15*4700)
    elif args.type == 'label_flip':
        labels_conf, labels_deconf = cb_label_flip(index_n_labels,p=0.5,
                                        qyu=args.corr_coff,qzu0= args.qzu0,qzu1=args.qzu1,
                                        N=15*4700)

    index_n_labels_v = split_n_label(split = 'valid', domains = args.domains, data = 'camelyon')
 
    # ## Validation samples (confounding and deconfounding1)
    if args.type == 'back':  
        labels_conf_v, labels_deconf_v = cb_backdoor(index_n_labels_v,p=0.5,
                                        qyu=args.corr_coff,
                                        N=2*args.samples)
    elif args.type == 'front':  
        labels_conf_v, labels_deconf_v = cb_frontdoor(index_n_labels_v,p=0.5,
                                        qyu=args.corr_coff,qzy= args.qzy,
                                        N=2*args.samples)
    elif args.type == 'back_front':  
        labels_conf_v, labels_deconf_v = cb_front_n_back(index_n_labels_v,p=0.5,
                                        qyu=args.corr_coff,qzy= args.qzy,
                                        N=2*args.samples)

    elif args.type == 'par_back_front':  
        labels_conf_v, labels_deconf_v = cb_par_front_n_back(index_n_labels_v,p=0.5,
                                        qyu=args.corr_coff,qzy= args.qzy,
                                        N=2*args.samples)
                    
    elif args.type == 'label_flip':
        labels_conf_v, labels_deconf_v = cb_label_flip(index_n_labels_v,p=0.5,
                                        qyu=args.corr_coff,qzu0= args.qzu0,qzu1=args.qzu1,
                                        N=2*args.samples)
    # pdb.set_trace()
    logger.info(f"sam: {args.samples}, epoch: {args.epochs}")
    
    train_type = args.data_type
        
    # Defining the Convolutional model 
    model_conv = mod.densenet121(pretrained=args.use_pretrained) 
    num_ftrs = model_conv.classifier.in_features
    model_conv.classifier = nn.Linear(num_ftrs, 2)
    model_conv = model_conv.to(device)

    # optimizer 
    optimizer = optim.Adam(model_conv.parameters(), lr = args.lr)         
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    ## load data of required kind using DataLoader and creating Dataclasses
    if train_type == 'Conf': 
        labels = labels_conf; labels_v = labels_conf_v
        train_data = Camelyon(labels = labels)
        valid_data = Camelyon(labels = labels_v)

    elif train_type == 'Deconf': 
        labels = labels_deconf; labels_v = labels_deconf_v
        train_data = Camelyon(labels = labels)
        valid_data = Camelyon(labels = labels_v)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(valid_data, batch_size=batch_size*2, shuffle=True) 
    
    es = EarlyStopping(patience = args.es_patience)         
    n_chunks_per_epoch = int(math.ceil(len(train_loader)/args.val_freq))
    n_chunks = args.epochs * n_chunks_per_epoch 
    counter = 0
    
    for epoch in range(1, args.epochs + 1):
        if es.early_stop:
            break
        iter_dl = iter(train_loader)    
        for chunk in range(n_chunks_per_epoch):   
            counter += 1
            if es.early_stop:
                break

            chunk_timer = timer()

            # Train and validate
            tr_loss, tr_acc = train(args, model_conv, writer, 
                device, iter_dl, optimizer, counter , batch_size, n_steps = args.val_freq)

            val_acc, val_loss, val_AUC, val_conf_mat, val_F1_score = validation(args, 
                model_conv, device, validation_loader, batch_size)

            end_chunk_timer = timer()

            writer.add_scalar(f'Loss/train/chunk/{train_type}', tr_loss, counter)
            writer.add_scalar(f'Loss/valid/chunk/{train_type}', val_loss, counter)
            writer.add_scalar(f'AUC/valid/chunk/{train_type}', val_AUC, counter)
            writer.add_scalar(f'Accuracy/train/chunk/{train_type}', tr_acc, counter)

            exp_lr_scheduler.step()

            for param_group in optimizer.param_groups:
                logger.info(f"Current learning rate is: {param_group['lr']}")
                writer.add_scalar('LearningRate/chunk',param_group['lr'])

            logger.info("#### End chunk {}/{}, elapsed time: {}".format(counter, n_chunks, end_chunk_timer - chunk_timer))

            es(-val_AUC, counter , model_conv.state_dict(), Path(res_pth)/'model.pt')   
        
    ## end 
    end_global_timer = timer()
    logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))
    logger.info(f'Training ends: {train_type}')

#     key_results = ["Conf Train: Conf Test","Conf Train: Unconf Test","Deconf Train: Conf Test","Deconf Train: Unconf Test"]
#     final_acc = dict.fromkeys(key_results,None)
#     metrics = dict.fromkeys(key_results, None)
    final_acc, metrics = {}, {}
    
    index_n_labels_t = split_n_label(split = 'test', domains = args.domains, data = 'camelyon')

    del(optimizer)

    for test_type in keylist_test: 
        if test_type == 'Conf':             
            if args.type == 'back':  
                labels_conf_t, _ = cb_backdoor(index_n_labels_t,p=0.5,
                                                qyu=args.corr_coff,N=2*args.samples)
            elif args.type == 'front':
                labels_conf_t, _ = cb_frontdoor(index_n_labels_t,p=0.5,
                                                qyu=args.corr_coff,qzy=args.qzy,N=2*args.samples)
            elif args.type == 'back_front':
                labels_conf_t, _ = cb_front_n_back(index_n_labels_t,p=0.5,
                                                qyu=args.corr_coff,qzy=args.qzy,N=2*args.samples)
            elif args.type == 'par_back_front':
                labels_conf_t, _ = cb_par_front_n_back(index_n_labels_t,p=0.5,
                                                qyu=args.corr_coff,qzy=args.qzy,N=2*args.samples)
            elif args.type == 'label_flip':
                labels_conf_t, _ = cb_label_flip(index_n_labels_t,p=0.5,
                                        qyu=args.corr_coff,qzu0= args.qzu0,qzu1=args.qzu1,
                                        N=args.samples)
    
            test_data = Camelyon(labels = labels_conf_t)
            test_loader = DataLoader(test_data, batch_size=batch_size*2, shuffle=True) # set shuffle to True

        elif test_type == 'Unconf':  
            # pdb.set_trace()
            if args.type == 'back': 
                labels_unconf_t, _ = cb_backdoor(index_n_labels_t,p=0.5,
                                                qyu=0.5,N=2*args.samples)
            elif args.type == 'front':
                labels_unconf_t, _ = cb_frontdoor(index_n_labels_t,p=0.5,
                                                qyu=0.5,qzy=args.qzy,N=2*args.samples)
            elif args.type == 'back_front':
                labels_unconf_t, _ = cb_front_n_back(index_n_labels_t,p=0.5,
                                                qyu=0.5,qzy=args.qzy,N=2*args.samples)
            elif args.type == 'par_back_front':
                labels_unconf_t, _ = cb_par_front_n_back(index_n_labels_t,p=0.5,
                                                qyu=0.5,qzy=args.qzy,N=2*args.samples)
            elif args.type == 'label_flip':
                labels_unconf_t, _ = cb_label_flip(index_n_labels_t,p=0.5,
                                        qyu=0.5,qzu0= args.qzu0,qzu1=args.qzu1,
                                        N=2*args.samples)

            test_data = Camelyon(labels = labels_unconf_t)
            test_loader = DataLoader(test_data, batch_size=batch_size*2, shuffle=True) # set shuffle to True            

        logger.info(f'===> loading best model {train_type} for prediction')
        model_conv.load_state_dict(torch.load(Path(res_pth)/'model.pt'))

        logger.info(f'===> testing best {train_type} model on {test_type} for prediction')

        # test_acc,test_loss = prediction(args, model_conv, device, test_loader, batch_size)       
        test_acc, test_loss, AUC, conf_mat, F1_score = prediction_analysis(args, model_conv, device, test_loader, batch_size)   

        final_acc[f'{train_type} Train: {test_type} Test'] = [test_acc, AUC]
        metrics[f'{train_type} Train: {test_type} Test'] = [conf_mat, F1_score]

        # final_acc[f'{train_type} Train: {test_type} Test'] = [test_acc]
        # metrics[f'{train_type} Train: {test_type} Test'] = [conf_mat, F1_score]

    writer.flush()
    writer.close()
    logger.info("################## Success #########################")
    logger.info(f'Final Accuracies and AUC scores: {final_acc}')
    # logger.info(f'Final Metrics (Confusion Metrics and F1 score): {metrics}')

    with open(os.path.join(res_pth, 'final_results.p'), 'wb') as res: 
        pickle.dump([final_acc, metrics], res)
        
    with open(os.path.join(res_pth, 'done'), 'w') as f:
        f.write('done')    