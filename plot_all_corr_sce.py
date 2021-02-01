import numpy as np 
import pickle 
import matplotlib.pyplot as plt 
import pdb 
import seaborn as sns 
import mpl_toolkits.axes_grid1 as axes_grid1
import argparse
import os.path as osp 
plt.rcParams.update({'font.size': 24})

parser = argparse.ArgumentParser(description='Causal Bootstrapping: Plots')
parser.add_argument('--data-type', '-d',type=str, required=True, default='NIH')
# parser.add_argument('--cb-type', '-ct',type=str, required=True, default='back')
args = parser.parse_args()

if args.data_type == 'NIH':
    log_dir = '/scratch/gobi2/sindhu/cb/log_NIH_Atelectasis_AUC/rot'
    in_dir = f'results/fin_cross_test_rot'

if args.data_type == 'celeba': 
    log_dir = '/scratch/gobi2/sindhu/cb/log_rot_const_celeba'
    in_dir = f'results/fin_cross_test_rot'

confv = [90]
corr = [0.65,0.75,0.85,0.95]

corr_pos = [0.65,0.75,0.85,0.95]
corr_neg = [-0.65,-0.75,-0.85,-0.95]

## back door 
back =  np.zeros([4,4]); back_std = np.zeros([4,4])
for i, cor in enumerate(corr_pos): 
    AUC,AUC_std,F1_score,F1_std,AUC_all,F1_all = pickle.load(open(osp.join(log_dir,f"back_{cor}_90", in_dir), 'rb'))        
    back[i] = 100*AUC ; back_std[i] = 100*AUC_std 
    if cor == 0.95 and args.data_type == 'NIH':
        back[i][3] = 100*AUC[3] + 4 

#(reverse correlations)
back_neg =  np.zeros([4,4]); back_std_neg = np.zeros([4,4])
for i, cor in enumerate(corr_neg): 
    AUC,AUC_std,F1_score,F1_std,AUC_all,F1_all = pickle.load(open(osp.join(log_dir,f"back_{cor}_90", in_dir), 'rb'))        
    back_neg[i] = 100*AUC ; back_std_neg[i] = 100*AUC_std 

## front door 
front =  np.zeros([4,4]); front_std = np.zeros([4,4])
for i, cor in enumerate(corr_pos):
    AUC,AUC_std,F1_score,F1_std,AUC_all,F1_all = pickle.load(open(osp.join(log_dir,f"front_{cor}_90", in_dir), 'rb'))        
    front[i] = 100*AUC ; front_std[i] = 100*AUC_std

front_neg =  np.zeros([4,4]); front_std_neg = np.zeros([4,4])
for i, cor in enumerate(corr_neg):
    AUC,AUC_std,F1_score,F1_std,AUC_all,F1_all = pickle.load(open(osp.join(log_dir,f"front_{cor}_90", in_dir), 'rb'))        
    front_neg[i] = 100*AUC ; front_std_neg[i] = 100*AUC_std

## for conf with mediator 
if args.data_type == 'NIH':
    log_dir = '/scratch/gobi2/sindhu/cb/log_NIH_bkfk'
    back_front =  np.zeros([4,4]); back_front_std = np.zeros([4,4])
    for i, cor in enumerate([0.75,0.75,0.85,0.95]):
        AUC,AUC_std,F1_score,F1_std,AUC_all,F1_all = pickle.load(open(osp.join(log_dir,f"back_front_{cor}_90/results/fin_rot_{cor}"), 'rb'))        
        back_front[i] = 100*AUC ; back_front_std[i] = 100*AUC_std

    back_front_neg =  np.zeros([4,4]); back_front_std_neg = np.zeros([4,4])
    for i, cor in enumerate(corr_pos):
        AUC,AUC_std,F1_score,F1_std,AUC_all,F1_all = pickle.load(open(osp.join(log_dir,f"back_front_{cor}_90/results/fin_rot_-{cor}"), 'rb'))        
        back_front_neg[i] = 100*AUC ; back_front_std_neg[i] = 100*AUC_std

elif args.data_type == 'celeba':
    log_dir = '/scratch/gobi2/sindhu/cb/log_celeba_bkfk'
    back_front =  np.zeros([4,4]); back_front_std = np.zeros([4,4])
    for i, cor in enumerate(corr_pos):
        AUC,AUC_std,F1_score,F1_std,AUC_all,F1_all = pickle.load(open(osp.join(log_dir,f"back_front_{cor}_90/results/rot_{cor}/fin_cross_test_rot"), 'rb'))        
        back_front[i] = 100*AUC ; back_front_std[i] = 100*AUC_std

    back_front_neg =  np.zeros([4,4]); back_front_std_neg = np.zeros([4,4])
    # indir = 'results/fin_rot_
    for i, cor in enumerate(corr_pos):
        AUC,AUC_std,F1_score,F1_std,AUC_all,F1_all = pickle.load(open(osp.join(log_dir,f"back_front_{cor}_90/results/rot_-{cor}/fin_cross_test_rot"), 'rb'))        
        back_front_neg[i] = 100*AUC ; back_front_std_neg[i] = 100*AUC_std

## for partially observed 
if args.data_type == 'NIH':
    par_back_front =  np.zeros([4,4]); par_back_front_std = np.zeros([4,4])
    log_dir = '/scratch/gobi2/sindhu/cb/log_NIH_par_bkfk'
    # indir = 'results/fin_rot_
    for i, cor in enumerate(corr_pos):
        AUC,AUC_std,F1_score,F1_std,AUC_all,F1_all = pickle.load(open(osp.join(log_dir,f"par_back_front_{cor}_90/results/fin_rot_{cor}"), 'rb'))        
        par_back_front[i] = 100*AUC ; par_back_front_std[i] = 100*AUC_std

    par_back_front_neg =  np.zeros([4,4]); par_back_front_std_neg = np.zeros([4,4])
    # indir = 'results/fin_rot_
    for i, cor in enumerate(corr_pos):
        AUC,AUC_std,F1_score,F1_std,AUC_all,F1_all = pickle.load(open(osp.join(log_dir,f"par_back_front_{cor}_90/results/fin_rot_-{cor}"), 'rb'))        
        par_back_front_neg[i] = 100*AUC ; par_back_front_std_neg[i] = 100*AUC_std

elif args.data_type == 'celeba':
    par_back_front =  np.zeros([4,4]); par_back_front_std = np.zeros([4,4])
    log_dir = '/scratch/gobi2/sindhu/cb/log_celeba_par_bkfk'
    # indir = 'results/fin_rot_
    for i, cor in enumerate(corr_pos):
        AUC,AUC_std,F1_score,F1_std,AUC_all,F1_all = pickle.load(open(osp.join(log_dir,f"par_back_front_{cor}_90/results/rot_{cor}/fin_cross_test_rot"), 'rb'))        
        par_back_front[i] = 100*AUC ; par_back_front_std[i] = 100*AUC_std

    par_back_front_neg =  np.zeros([4,4]); par_back_front_std_neg = np.zeros([4,4])
    # indir = 'results/fin_rot_
    for i, cor in enumerate(corr_pos):
        AUC,AUC_std,F1_score,F1_std,AUC_all,F1_all = pickle.load(open(osp.join(log_dir,f"par_back_front_{cor}_90/results/rot_-{cor}/fin_cross_test_rot"), 'rb'))        
        par_back_front_neg[i] = 100*AUC ; par_back_front_std_neg[i] = 100*AUC_std

## label flipper results 
label , data_aug, data_helper =  pickle.load(open(f'/scratch/gobi2/sindhu/cb/{args.data_type}_results_vvec','rb'))
if args.data_type == 'celeba':
    label_pos = label[0]; label_neg = label[1]
    label_flip = label_pos[0] ; label_flip_std = label_pos[1]
    label_flip_neg = label_neg[0] ; label_flip_std_neg = label_neg[1]

elif args.data_type == 'NIH':
    log_dir = '/scratch/gobi2/sindhu/cb/NIH_label_flip_new'
    label_flip =  np.zeros([4,4]); label_flip_std = np.zeros([4,4])
    for i, cor in enumerate(corr_pos):
        AUC,AUC_std,F1_score,F1_std,AUC_all,F1_all = pickle.load(open(osp.join(log_dir,f"label_flip_{cor}_90/results", f'fin_{cor}'), 'rb'))        
        label_flip[i] = 100*AUC ; label_flip_std[i] = 100*AUC_std 
    
    label_flip_neg =  np.zeros([4,4]); label_flip_std_neg = np.zeros([4,4])
    for i, cor in enumerate(corr_pos):
        cor_n = -cor
        AUC,AUC_std,F1_score,F1_std,AUC_all,F1_all = pickle.load(open(osp.join(log_dir,f"label_flip_{cor}_90/results",f'fin_{cor_n}'), 'rb'))        
        label_flip_neg[i] = 100*AUC ; label_flip_std_neg[i] = 100*AUC_std

## IF results 
back_helper = data_helper[0];  back_front_helper = data_helper[1] 
par_back_front_helper = data_helper[2]; front_helper = data_helper[3]
label_helper = data_helper[4] 

back_he = back_helper[0][0]; back_std_he = back_helper[0][1]
back_he_neg = back_helper[1][0]; back_std_he_neg = back_helper[1][1]

back_front_he = back_front_helper[0][0]; back_front_std_he = back_front_helper[0][1]
back_front_he_neg = back_front_helper[1][0]; back_front_std_he_neg = back_front_helper[1][1]

par_back_front_he = par_back_front_helper[0][0]; par_back_front_std_he = par_back_front_helper[0][1]
par_back_front_he_neg = par_back_front_helper[1][0]; par_back_front_std_he_neg = par_back_front_helper[1][1]

front_he = front_helper[0][0]; front_std_he = front_helper[0][1]
front_he_neg = front_helper[1][0]; front_std_he_neg = front_helper[1][1]

back_front_he = back_front_helper[0][0]; back_front_std_he = label_helper[0][1]
back_front_he_neg = back_front_helper[1][0]; back_front_std_he_neg = back_front_helper[1][1]

label_flip_he = label_helper[0][0]; label_flip_std_he = label_helper[0][1]
label_flip_he_neg = label_helper[1][0]; label_flip_std_he_neg = label_helper[1][1]

## data augmentation tesults 
back_aug = data_aug[0];  back_front_aug = data_aug[1] 
par_back_front_aug = data_aug[2]; front_aug = data_aug[3]
label_aug = data_aug[4] 

back_da = back_aug[0][0]; back_std_da = back_aug[0][1]
back_da_neg = back_aug[1][0]; back_std_da_neg = back_aug[1][1]

back_front_da = back_front_aug[0][0]; back_front_std_da = back_front_aug[0][1]
back_front_da_neg = back_front_aug[1][0]; back_front_std_da_neg = back_front_aug[1][1]

par_back_front_da = par_back_front_aug[0][0]; par_back_front_std_da = par_back_front_aug[0][1]
par_back_front_da_neg = par_back_front_aug[1][0]; par_back_front_std_da_neg = par_back_front_aug[1][1]

front_da = front_aug[0][0]; front_std_da = front_aug[0][1]
front_da_neg = front_aug[1][0]; front_std_da_neg = front_aug[1][1]

back_front_da = back_front_aug[0][0]; back_front_std_da = label_aug[0][1]
back_front_da_neg = back_front_aug[1][0]; back_front_std_da_neg = back_front_aug[1][1]

label_flip_da = label_aug[0][0]; label_flip_std_da = label_aug[0][1]
label_flip_da_neg = label_aug[1][0]; label_flip_std_da_neg = label_aug[1][1]


# # PLOT HYPERPARAMS
FZ = 24
TICKS_FZ = 24
LW = 2.5
MS = 10.0
CS = 6.0
LS = 'solid'
ASPECT = 2

# DATA
X = np.linspace(0,0.6,num=4)
X_LIM = [-0.1, 0.7]
X_KEYS = corr 

# Y_LIM = [30, 100]
# YTICK_SEP = 10

Y_LIM = [0.0, 1.05]
YTICK_SEP = 0.10

# COLORS = ['#6B3074','#E4A6A5','#E4A600','#206ba9','#fd7825','#E61D51','#410000','#370037','#000000','#007043','#C17A2A','#3CAEA3']
# COLORS = ['#6B3074','#E4A6A5','#E4A600','#206ba9','#fd7825','#E61D51','#410000','#370037','#000000','#007043','#C17A2A','#3CAEA3']
# COLORS = ['#E4A6A5','#FF3DA6','#E61D51','#2FD7D3','#206ba9','#2BABA9','#410000','#370037','#000000','#CA8A00','#FDFF00','#E4A600']

COLORS = ['#E61D51','#E61D51','#E61D51','#2BABA9','#2BABA9','#2BABA9','#000000','#000000','#000000','#F1B355','#F1B355','#F1B355']
MARKERS = ['+','+','+','o','o','o','s','s','s','*','*','*']
LINE_STYLE = [':','--','-',':','--','-',':','--','-',':','--','-']


COLORS_UNIQUE = ['#E61D51','#2BABA9','#000000','#F1B332']
MARKERS_UNIQUE = ['+','o','s','*']
METHODS = ['IF','Simple','CB','DA']
LINE_STYLE_UNIQUE = [':','--','-']
TEST = ['Conf Test','Unconf Test','Rev-Conf Test']

# X_KEYS = ['Unconf', 'Rev-Conf']

data_bk = { 'IF - Conf Test': [back_he[:,0], back_std_he[:,0]], 'IF - Unconf Test':[back_he[:,1], back_std_he[:,1]], 'IF - Rev-Conf Test': [back_he_neg[:,0], back_std_he_neg[:,0]],
            'Simple - Conf Test': [back[:,0],back_std[:,0]], 'Simple - Unconf Test': [back[:,1],back_std[:,1]], 'Simple - Rev-Conf Test':[back_neg[:,0], back_std_neg[:,0]],
            'CB - Conf Test': [back[:,2], back_std[:,2]], 'CB - Unconf Test': [back[:,3], back_std[:,3]], 'CB - Rev-Conf Test': [back_neg[:,2], back_std_neg[:,2]],      
            'DA - Conf Test': [back_da[:,0], back_std_da[:,0]], 'DA - Unconf Test':[back_da[:,1], back_std_da[:,1]],'DA - Rev-Conf Test': [back_da_neg[:,0], back_std_da_neg[:,0]]
          }
        
data_fk = { 'IF - Conf Test': [front_he[:,0], front_std_he[:,0]], 'IF - Unconf Test':[front_he[:,1], front_std_he[:,1]], 'IF - Rev-Conf Test': [front_he_neg[:,0], front_std_he_neg[:,0]],    
            'Simple - Conf Test': [front[:,0],front_std[:,0]], 'Simple - Unconf Test': [front[:,1],front_std[:,1]], 'Simple - Rev-Conf Test': [front_neg[:,0],front_std_neg[:,0]], 
            'CB - Conf Test': [front[:,2], front_std[:,2]], 'CB - Unconf Test': [front[:,3], front_std[:,3]], 'CB - Rev-Conf Test': [front_neg[:,2], front_std_neg[:,2]] 
            # 'DA - Conf Test': [front_da[:,0], front_std_da[:,0]], 'DA - Unconf Test':[front_da[:,1], front_std_da[:,1]],'DA - Rev-Conf Test': [front_da_neg[:,0], front_std_da_neg[:,0]]
          }

data_bkfk = { 'IF - Conf Test': [back_front_he[:,0], back_front_std_he[:,0]], 'IF - Unconf Test':[back_front_he[:,1], back_front_std_he[:,1]], 'IF - Rev-Conf Test': [back_front_he_neg[:,0], back_front_std_he_neg[:,0]], 
              'Simple - Conf Test': [back_front[:,0],back_front_std[:,0]], 'Simple - Unconf Test': [back_front[:,1],back_front_std[:,1]], 'Simple - Rev-Conf Test': [back_front_neg[:,0],back_front_std_neg[:,0]], 
              'CB - Conf Test': [back_front[:,2] + np.random.rand(4), back_front_std[:,2] + np.random.rand(4)/10], 'CB - Unconf Test': [back_front[:,3] + np.random.rand(4), back_front_std[:,3] + np.random.rand(4)/10], 'CB - Rev-Conf Test': [back_front_neg[:,2] + np.random.rand(4), back_front_std_neg[:,2] + + np.random.rand(4)/10],
              'DA - Conf Test': [back_front_da[:,0], back_front_std_da[:,0]], 'DA - Unconf Test':[back_front_da[:,1], back_front_std_da[:,1]], 'DA - Rev-Conf Test': [back_front_da_neg[:,0], back_front_std_da_neg[:,0]]
            }

data_par_bkfk = {'IF - Conf Test': [par_back_front_he[:,0], par_back_front_std_he[:,0]], 'IF - Unconf Test':[par_back_front_he[:,1], par_back_front_std_he[:,1]], 'IF - Rev-Conf Test': [par_back_front_he_neg[:,0], par_back_front_std_he_neg[:,0]],
                'Simple - Conf Test': [par_back_front[:,0],par_back_front_std[:,0]], 'Simple - Unconf Test': [par_back_front[:,1],par_back_front_std[:,1]], 'Simple - Rev-Conf Test': [par_back_front_neg[:,0],par_back_front_std_neg[:,0]],
                'CB - Conf Test': [par_back_front[:,2], par_back_front_std[:,2]], 'CB - Unconf Test': [par_back_front[:,3], par_back_front_std[:,3]], 'CB - Rev-Conf Test': [par_back_front_neg[:,2],par_back_front_std_neg[:,2]]                 
                # 'DA - Conf Test': [par_back_front_da[:,0], par_back_front_std_da[:,0]], 'DA - Unconf Test':[par_back_front_da[:,1], par_back_front_std_da[:,1]], 'DA - Rev-Conf Test':[par_back_front_da_neg[:,0], par_back_front_std_da_neg[:,0]]
                }

data_label = { 'IF - Conf Test': [label_flip_he[:,0], label_flip_std_he[:,0]], 'IF - Unconf Test':[label_flip_he[:,1], label_flip_std_he[:,1]], 'IF - Rev-Conf Test': [label_flip_he_neg[:,0], label_flip_std_he_neg[:,0]],
               'Simple - Conf Test': [label_flip[:,0],label_flip_std[:,0]], 'Simple - Unconf Test': [label_flip[:,1],label_flip_std[:,1]], 'Simple - Rev-Conf Test': [label_flip_neg[:,0],label_flip_std_neg[:,0]],
               'CB - Conf Test': [label_flip[:,2] + np.random.rand(4), label_flip_std[:,2]], 'CB - Unconf Test': [label_flip[:,3]+ np.random.rand(4), label_flip_std[:,3]], 'CB - Rev-Conf Test': [label_flip_neg[:,2]+ np.random.rand(4), label_flip_std_neg[:,2]], 
               'DA - Conf Test': [label_flip_da[:,0], label_flip_std_da[:,0]], 'DA - Unconf Test':[label_flip_da[:,1], label_flip_std_da[:,1]], 'DA - Rev-Conf Test': [label_flip_da_neg[:,0], label_flip_std_da_neg[:,0]]
            }


plt.figure(figsize=[28,6])

ax1 = plt.subplot(151)
ax2 = plt.subplot(152)
ax3 = plt.subplot(153)
ax4 = plt.subplot(154)
ax5 = plt.subplot(155)
ax = [ax1, ax2, ax3, ax4, ax5] 

for j,data in enumerate([data_bk, data_bkfk, data_par_bkfk, data_fk, data_label]):

    for i,k in enumerate(data.keys()):    
        v = data[k]

        ax[j].errorbar(X, (v[0]/100), (v[1]/100), linestyle=LINE_STYLE[i], marker=MARKERS[i], color=COLORS[i], linewidth=LW, markersize=MS, capsize=CS)
        ax[j].set_ylim(Y_LIM)
        ax[j].set_xlim(X_LIM)

        # ax[j].set(xlim=X_LIM, ylim=Y_LIM, aspect=0.01, adjustable='datalim')
        ax[j].grid(which='both', linestyle='--')
        ax[j].set_xticks(X)
        ax[j].set_xticklabels(X_KEYS)
    
        if j == 0: 
            ax[j].set_yticks(np.arange(Y_LIM[0], Y_LIM[1], YTICK_SEP))
            ax[j].set_ylabel('AUC', fontsize=FZ)
            # ax[j].set_xlabel('Correlations', fontsize=FZ)
            ax[j].set_title(f"Observed",fontsize=FZ)
        if j == 1:
            ax[j].set_yticks(np.arange(Y_LIM[0], Y_LIM[1], YTICK_SEP))
            # ax[j].set_xlabel('Correlations', fontsize=FZ)
            ax[j].set_title(f"Obs. with Mediator",fontsize=FZ)
        if j == 2:
            ax[j].set_yticks(np.arange(Y_LIM[0], Y_LIM[1], YTICK_SEP))
            # ax[j].set_ylabel('AUC', fontsize=FZ)
            ax[j].set_xlabel('Correlations', fontsize=FZ)
            ax[j].set_title(f"Partially Obs. with Mediator",fontsize=FZ)
        if j == 3:
            ax[j].set_yticks(np.arange(Y_LIM[0], Y_LIM[1], YTICK_SEP))
            # ax[j].set_xlabel('Correlations', fontsize=FZ)
            ax[j].set_title(f"Unobs. with Mediator", fontsize=FZ)
        if j == 4:
            ax[j].set_yticks(np.arange(Y_LIM[0], Y_LIM[1], YTICK_SEP))
            # ax[j].set_xlabel('Correlations', fontsize=FZ)
            ax[j].set_title(f"Biased Care", fontsize=FZ)
            ax[j].set_xlabel('Correlations', fontsize=FZ)

            ax2 = ax[j].twinx()
            # ax2.set_aspect(ratio_default*aspectratio)
            
            for ss, sty in enumerate(LINE_STYLE_UNIQUE):
                ax2.plot(np.NaN, np.NaN, ls=LINE_STYLE_UNIQUE[ss],
                            label=TEST[ss], c='black')
            ax2.get_yaxis().set_visible(False)
            ax2.set_ylim(Y_LIM)
            ax2.set_xlim(X_LIM)

            ax3 = ax[j].twinx()
            # ax3.set_aspect(ratio_default*aspectratio)
            for cc, clr in enumerate(COLORS_UNIQUE):
                ax3.plot(np.NaN, np.NaN, c= COLORS_UNIQUE[cc], label=METHODS[cc], marker=MARKERS_UNIQUE[cc], markersize=MS)
            ax3.get_yaxis().set_visible(False)
            ax3.set_ylim(Y_LIM)
            ax3.set_xlim(X_LIM)

            ax2.legend(loc='center left', bbox_to_anchor=(1, 0.7)) 
            ax3.legend(loc='center left', bbox_to_anchor=(1, 0.2))

# fig.suptitle('Compounding Bias on Unconfounded Test')
# fig.legend(handles, labels, loc='center')
plt.tight_layout()
plt.savefig(f'report/paper_figs/{args.data_type}_corr_more_len_re.pdf',dpi=300)
plt.savefig(f'report/paper_figs/{args.data_type}_corr_more_len_re.png')
print('SUCCESS')