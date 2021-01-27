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

# from data.data_camelyon import Camelyon17Dataset
from data.data_cam import Camelyon, split_train_test, split_n_label, transform   
from utils.logging import setup_logs
from src.training import train, snapshot, train_helper
from src.validation import validation, validation_helper 
from src.prediction import prediction_analysis, prediction_analysis_helper

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
    parser.add_argument('--logging-dir','-l', type=str, required=True)
    
    parser.add_argument('--log-interval','-i', type=int, required=False, default=1)
    parser.add_argument('--epochs','-e', type=int, required=False, default=15)

    # parser.add_argument('--conf-type','-ct',type=str, required=True, default='rot')
    # parser.add_argument('--conf-val','-cv', type=float, required=False, default=0.5)
    
    parser.add_argument('--corr-coff','-q', type=float, required=False, default=0.95)
    parser.add_argument('--qzy',type=float, required=False, default=0.95)
    parser.add_argument('--qzu0',type=float, required=False, default=0.80)
    parser.add_argument('--qzu1',type=float, required=False, default=0.95)

    parser.add_argument('--data','-d', type=str, default= 'camelyon', required=False)
    parser.add_argument('--domains','-do', type=list, default=[2,3], required=False)
    parser.add_argument('--batch-size','-b', type=int, default=64, required=False)

    args = parser.parse_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    writer = SummaryWriter(log_dir=os.path.join(args.logging_dir, 'tensorboard'), comment=run_name)

    # to genertate train/val/test split - once generated and stored
    # split_train_test(train_ratio=0.8, root_dir='/scratch/gobi2/sindhu/datasets/WILDS')

    os.makedirs(args.logging_dir, exist_ok=True)
    res_pth = os.path.join(args.logging_dir, 'results') 
    os.makedirs(res_pth, exist_ok=True)
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu") 
    print(device)
    
    # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    global_timer = timer() # global timer
    logger = setup_logs(args.logging_dir, run_name) # setup logs
    
    ## for data augmentation (DA) scenario (only train in the confounded case)    
    keylist_train = ['Conf', 'Deconf']; keylist_test = ['Unconf', 'Conf']
    batch_size = args.batch_size

    index_n_labels = split_n_label(split = 'train', domains = args.domains, data = 'camelyon', root_dir='/scratch/gobi2/sindhu/datasets/WILDS')

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

    index_n_labels_v = split_n_label(split = 'valid', domains = args.domains, data = 'camelyon', root_dir='/scratch/gobi2/sindhu/datasets/WILDS')
 
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
    
    for train_type in keylist_train:

        logger.info(f"training model {train_type} train data")
        
        # Defining the Convolutional model 
        model_conv = mod.densenet121(pretrained=False) # (checking if using pretrained weights is causing the problem) 
        num_ftrs = model_conv.classifier.in_features
        model_conv.classifier = nn.Linear(num_ftrs, 2)
        model_conv = model_conv.to(device)

        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        # model_conv = Conv_confemb(args.data)
        # model_conv = model_conv.to(device)

        # optimizer 
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_conv.parameters()), lr = 0.001)         
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        ## load data of required kind using DataLoader and creating Dataclasses
        if train_type == 'Conf': 
            labels = labels_conf; labels_v = labels_conf_v
            train_data = Camelyon(labels = labels, transform = transform)
            valid_data = Camelyon(labels = labels_v, transform = transform)

        elif train_type == 'Deconf': 
            labels = labels_deconf; labels_v = labels_deconf_v
            train_data = Camelyon(labels = labels, transform = transform)
            valid_data = Camelyon(labels = labels_v, transform = transform)

        # pdb.set_trace()
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) # set shuffle to True
        validation_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True) # set shuffle to True

        best_AUC = 0; best_loss = np.inf; best_epoch = -1 

        # pdb.set_trace()
        for epoch in range(1, args.epochs + 1):
            # pdb.set_trace()
            epoch_timer = timer()

            # Train and validate
            tr_loss, tr_acc = train(args, model_conv, writer, 
                device, train_loader, optimizer, epoch, batch_size)
            
            val_acc, val_loss, val_AUC, val_conf_mat, val_F1_score = validation(args, 
                model_conv, device, validation_loader, batch_size)
            
            # Save
            if val_AUC > best_AUC: 
                best_AUC = max(val_AUC, best_AUC)
                snapshot(args.logging_dir, res_pth, (run_name + train_type) , {
                    'epoch': epoch + 1,
                    'validation_acc': val_AUC, 
                    'state_dict': model_conv.state_dict(),
                    'validation_loss': val_loss,
                    'optimizer': optimizer.state_dict(),
                })
                best_epoch = epoch + 1
            
            elif epoch - best_epoch > 2:
                best_epoch = epoch + 1
            
            end_epoch_timer = timer()

            writer.add_scalar(f'Loss/train/epoch/{train_type}', tr_loss, epoch)
            writer.add_scalar(f'Loss/valid/epoch/{train_type}', val_loss, epoch)
            writer.add_scalar(f'AUC/valid/epoch/{train_type}', val_AUC, epoch)
            writer.add_scalar(f'Accuracy/train/epoch/{train_type}', tr_acc, epoch)
            # writer.add_scalar(f'Accuracy/valid/epoch/{train_type}', val_acc, epoch)

            exp_lr_scheduler.step()

            for param_group in optimizer.param_groups:
                logger.info(f"Current learning rate is: {param_group['lr']}")
                writer.add_scalar('LearningRate/epoch',param_group['lr'])
            
            logger.info("#### End epoch {}/{}, elapsed time: {}".format(epoch, args.epochs, end_epoch_timer - epoch_timer))

        ## end 
        end_global_timer = timer()
        logger.info("Total elapsed time: %s" % (end_global_timer - global_timer))
        logger.info(f'Training ends: {train_type}')

    key_results = ["Conf Train: Conf Test","Conf Train: Unconf Test","Deconf Train: Conf Test","Deconf Train: Unconf Test"]
    final_acc = dict.fromkeys(key_results,None)
    metrics = dict.fromkeys(key_results, None)
    
    index_n_labels_t = split_n_label(split = 'test', domains = args.domains, data = 'camelyon', root_dir='/scratch/gobi2/sindhu/datasets/WILDS')

    del(optimizer)
    ## This way the same test set is being used by both conf and deconf data for both train and test 

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
    
            test_data = Camelyon(labels = labels_conf_t, transform = transform)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True) # set shuffle to True

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

            test_data = Camelyon(labels = labels_unconf_t, transform = transform)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True) # set shuffle to True            

        for train_type in keylist_train:

            logger.info(f'===> loading best model {train_type} for prediction')
            checkpoint = torch.load(os.path.join(args.logging_dir, run_name + train_type + '-model_best.pth'))
            model_conv.load_state_dict(checkpoint['state_dict'])

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