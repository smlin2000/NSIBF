#Code runs saved, trained NSIBF model on PUMP dataset
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append(r'C:/Users/rossm/Documents/GitHub/test_nsibf')
from framework.models import NSIBF
from framework.preprocessing.data_loader import load_pump_data
from framework.HPOptimizer.Hyperparameter import UniformIntegerHyperparameter,ConstHyperparameter,\
    UniformFloatHyperparameter
from framework.HPOptimizer import HPOptimizers
from framework.preprocessing import normalize_and_encode_signals
from framework.utils.metrics import bf_search
from framework.utils import negative_sampler
import sys
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)



train_df,val_df,test_df,signals = load_pump_data()

seqL = 5
kf = NSIBF(signals, window_length=seqL, input_range=seqL*3)


train_df = normalize_and_encode_signals(train_df,signals,scaler='min_max') 
train_x,train_u,train_y,_ = kf.extract_data(train_df)
x_train = [train_x,train_u]
y_train = [train_x,train_y]
pos = len(train_x)*3//4
valtest_x = train_x[pos:,:]
valtest_u = train_u[pos:,:]
valtest_y = train_y[pos:,:]


#set retrain to False to reproduce the results in the paper
retrain_model = False
if retrain_model:
    x,u,y = [],[],[]
    for i in range(20):
        r = 0.05*i
        negative_df = negative_sampler.apply_negative_samples(val_df, signals, sample_ratio=r, sample_delta=0.05)
        negative_df = normalize_and_encode_signals(negative_df,signals,scaler='min_max')
        neg_x,neg_u,_,neg_labels = kf.extract_data(negative_df,purpose='AD',freq=seqL,label='class_label')
        neg_labels = neg_labels.sum(axis=1)
        neg_labels[neg_labels<seqL]=0
        neg_labels[neg_labels==seqL]=1
        x.append(neg_x)
        u.append(neg_u)
        y.append(neg_labels)
    x = np.concatenate(x)
    u = np.concatenate(u)
    y = np.concatenate(y)
    print(list(y).count(1),len(y))
    x_neg = [x,u]
    y_neg = y
    
    hp_list = []
    hp_list.append(UniformIntegerHyperparameter('z_dim',1,120)) 
    hp_list.append(UniformIntegerHyperparameter('hnet_hidden_layers',1,3))  
    hp_list.append(UniformIntegerHyperparameter('fnet_hidden_layers',1,3))
    hp_list.append(UniformIntegerHyperparameter('fnet_hidden_dim',32,256))
    hp_list.append(UniformIntegerHyperparameter('uencoding_layers',1,3))
    hp_list.append(UniformIntegerHyperparameter('uencoding_dim',32,256))
    hp_list.append(UniformFloatHyperparameter('l2',0,0.05))
    hp_list.append(ConstHyperparameter('save_best_only',True))
    hp_list.append(ConstHyperparameter('validation_split',0.1))
    hp_list.append(ConstHyperparameter('batch_size',256*8))
    hp_list.append(ConstHyperparameter('epochs',100))
    hp_list.append(ConstHyperparameter('verbose',2))
    
    
    optor = HPOptimizers.RandomizedGS(kf, hp_list,x_train, y_train,x_neg,y_neg)
    kf,optHPCfg,bestScore = optor.run(n_searches=10,verbose=1)
    # kf.save_model('../results/PUMP')
    print('optHPCfg',optHPCfg)
    print('bestScore',bestScore)
else: ## load pretrained model
    kf = kf.load_model(r'C:/Users/rossm/Documents/GitHub/test_nsibf/results/PUMP')
kf.estimate_noise(valtest_x,valtest_u,valtest_y)

val_df = normalize_and_encode_signals(val_df,signals,scaler='min_max') 
val_x,val_u,val_y,_ = kf.extract_data(val_df)

T = np.linspace(1,len(val_x),len(val_x))

plt.plot(T[0:1000],val_x[0:1000],linestyle='-',label='Observed measurements')
plt.show()

test_df = normalize_and_encode_signals(test_df,signals,scaler='min_max')
test_x,test_u,_,labels = kf.extract_data(test_df,purpose='AD',freq=seqL,label='label')
print(len(test_x))
#28678 length

x_mu,x_cov = kf.filter(test_x, test_u,reset_hidden_states=True)

true_x = []

for i in range(len(x_mu)):
    for j in range(seqL):
        true_x.append(test_x[i+1,j])

T = np.linspace(1,len(true_x),len(true_x))

plt.plot(T[0:1000],true_x[0:1000],linestyle='-',label='Observed measurements')
plt.show()


labels = labels.sum(axis=1)
labels[labels>0]=1

## estimate noise matrices
kf.estimate_noise(val_x,val_u,val_y)
#z_scores = kf.score_samples(test_x, test_u,reset_hidden_states=True)
## np.savetxt('../results/PUMP/NSIBF_scores',z_scores)
## z_scores = np.loadtxt('../results/PUMP/NSIBF_scores')
#recon_scores,pred_scores = kf.score_samples_via_residual_error(test_x,test_u)
#print()
#  
#z_scores = np.nan_to_num(z_scores)
#t, th = bf_search(z_scores, labels[1:],start=0,end=np.percentile(z_scores,99.9),step_num=10000,display_freq=50,verbose=False)
#print('NSIBF')
#print('best-f1', t[0])
#print('precision', t[1])
#print('recall', t[2])
#print('accuracy',(t[3]+t[4])/(t[3]+t[4]+t[5]+t[6]))
#print('TP', t[3])
#print('TN', t[4])
#print('FP', t[5])
#print('FN', t[6])
#print()
#
#t, th = bf_search(recon_scores[1:], labels[1:],start=0,end=np.percentile(recon_scores,99.9),step_num=10000,display_freq=50,verbose=False)
#print('NSIBF-RECON')
#print('best-f1', t[0])
#print('precision', t[1])
#print('recall', t[2])
#print('accuracy',(t[3]+t[4])/(t[3]+t[4]+t[5]+t[6]))
#print('TP', t[3])
#print('TN', t[4])
#print('FP', t[5])
#print('FN', t[6])
#print()
#
#t, th = bf_search(pred_scores, labels[1:],start=0,end=np.percentile(pred_scores,99.9),step_num=10000,display_freq=50,verbose=False)
#print('NSIBF-PRED')
#print('best-f1', t[0])
#print('precision', t[1])
#print('recall', t[2])
#print('accuracy',(t[3]+t[4])/(t[3]+t[4]+t[5]+t[6]))
#print('TP', t[3])
#print('TN', t[4])
#print('FP', t[5])
#print('FN', t[6])

   


        