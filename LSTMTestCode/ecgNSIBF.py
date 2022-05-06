#Code runs saved, trained NSIBF model on WADI dataset

import numpy as np
import sys
import pandas as pd
from testDataLoader import *
sys.path.append(r'C:/Users/rossm/Documents/GitHub/NSIBF')
from framework.models import NSIBF
from framework.preprocessing.data_loader import load_wadi_data
from framework.HPOptimizer.Hyperparameter import UniformIntegerHyperparameter,ConstHyperparameter,\
    UniformFloatHyperparameter
from framework.HPOptimizer import HPOptimizers
from framework.preprocessing import normalize_and_encode_signals
from framework.utils.metrics import bf_search
from framework.utils import negative_sampler
from framework.preprocessing.signals import ContinousSignal,DiscreteSignal,SignalSource

import logging
import tensorflow as tf
logging.getLogger('tensorflow').setLevel(logging.ERROR)

dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)

columnTitles = []
counter = 0

#generate names for the columns in ecg dataset
for column_name, column_series in dataframe.iloc[:,:-1].iteritems():
    counter += 1
    columnTitles.append("Sensor" + str(counter))
columnTitles.append("Label")

dataframe.columns = columnTitles


#df_to_csv("ecgNamed", dataframe)
#

#seqL = 12
#kf = NSIBF(signals, window_length=seqL, input_range=seqL*3)


#train_df,val_df,test_df,signals = load_wadi_data()




train_dataOri, val_dataOri, test_dataOri, train_labels, val_labels, test_labels = load_ecg_dataset()

#print(train_dataOri)

train_dataOri = tf.cast(train_dataOri, tf.float64)
val_dataOri = tf.cast(val_dataOri, tf.float64)
test_dataOri = tf.cast(test_dataOri, tf.float64)

min_val = tf.reduce_min(train_dataOri)
max_val = tf.reduce_max(train_dataOri)

train_data = (train_dataOri - min_val) / (max_val - min_val)
val_data = (val_dataOri - min_val) / (max_val - min_val)
test_data = (test_dataOri - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float64)
val_data = tf.cast(val_data, tf.float64)
test_data = tf.cast(test_data, tf.float64)

#drop labels column
columnTitles = columnTitles[:-1]

train_dfOri = pd.DataFrame(train_dataOri.numpy(), columns=columnTitles)
val_dfOri = pd.DataFrame(val_dataOri.numpy(), columns=columnTitles)
test_dfOri = pd.DataFrame(test_dataOri.numpy(), columns=columnTitles)

train_df = pd.DataFrame(train_data.numpy(), columns=columnTitles)
val_df = pd.DataFrame(val_data.numpy(), columns=columnTitles)
test_df = pd.DataFrame(test_data.numpy(), columns=columnTitles)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

signals = []
for name in train_df:
    signals.append( ContinousSignal(name, SignalSource.sensor, isInput=True, isOutput=True, 
                                    min_value=train_df[name].min(), max_value=train_df[name].max(),
                                    mean_value=train_df[name].mean(), std_value=train_df[name].std()) )

#print(train_df.head(10))
#print(val_df.head(10))
#print(test_df.head(10))
#
#print(signals[0].name)
#print(signals[0].min_value)

test_df['label'] = test_labels
print(test_df.head(10))

seqL = 12
kf = NSIBF(signals, window_length=seqL, input_range=seqL*3)

train_x, train_u, train_y, _ = kf.extract_data(train_df)
x_train = [train_x,train_u]
y_train = [train_x,train_y]

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
    hp_list.append(UniformIntegerHyperparameter('z_dim',1,200)) 
    hp_list.append(UniformIntegerHyperparameter('hnet_hidden_layers',1,3))
    hp_list.append(UniformIntegerHyperparameter('fnet_hidden_layers',1,3))
    hp_list.append(UniformIntegerHyperparameter('fnet_hidden_dim',32,256))
    hp_list.append(UniformIntegerHyperparameter('uencoding_layers',1,3))
    hp_list.append(UniformIntegerHyperparameter('uencoding_dim',32,256))
    hp_list.append(UniformFloatHyperparameter('l2',0,0.05))
    hp_list.append(ConstHyperparameter('epochs',50))
    hp_list.append(ConstHyperparameter('save_best_only',True))
    hp_list.append(ConstHyperparameter('validation_split',0.1))
    hp_list.append(ConstHyperparameter('batch_size',256*16))
    hp_list.append(ConstHyperparameter('verbose',2))

    optor = HPOptimizers.RandomizedGS(kf, hp_list,x_train, y_train,x_neg,y_neg)
    kf,optHPCfg,bestScore = optor.run(n_searches=10,verbose=1)
    kf.save_model(r'C:/Users/rossm/Documents/GitHub/test_nsibf/testModelecgNSIBF')
    print('optHPCfg',optHPCfg)
    print('bestScore',bestScore)
else:
    kf = kf.load_model(r'C:/Users/rossm/Documents/GitHub/test_nsibf/testModelecgNSIBF')

val_df = normalize_and_encode_signals(val_df,signals,scaler='min_max') 
val_x,val_u,val_y,_ = kf.extract_data(val_df)

test_df = normalize_and_encode_signals(test_df,signals,scaler='min_max')
test_x,test_u,_,labels = kf.extract_data(test_df,purpose='AD',freq=seqL,label='label')
labels = labels.sum(axis=1)
labels[labels>0]=1

kf.estimate_noise(val_x,val_u,val_y)

z_scores = kf.score_samples(test_x, test_u,reset_hidden_states=True)
# np.savetxt('../results/WADI/NSIBF_sores',z_scores)
# z_scores = np.loadtxt('../results/WADI/NSIBF_sores')
recon_scores,pred_scores = kf.score_samples_via_residual_error(test_x,test_u)
print()
  
z_scores = np.nan_to_num(z_scores)
t, th = bf_search(z_scores, labels[1:],start=0,end=np.percentile(z_scores,99.9),step_num=10000,display_freq=50,verbose=False)
print('NSIBF')
print('best-f1', t[0])
print('precision', t[1])
print('recall', t[2])
print('accuracy',(t[3]+t[4])/(t[3]+t[4]+t[5]+t[6]))
print('TP', t[3])
print('TN', t[4])
print('FP', t[5])
print('FN', t[6])
print()

t, th = bf_search(recon_scores[1:], labels[1:],start=0,end=np.percentile(recon_scores,99.9),step_num=10000,display_freq=50,verbose=False)
print('NSIBF-RECON')
print('best-f1', t[0])
print('precision', t[1])
print('recall', t[2])
print('accuracy',(t[3]+t[4])/(t[3]+t[4]+t[5]+t[6]))
print('TP', t[3])
print('TN', t[4])
print('FP', t[5])
print('FN', t[6])
print()

t, th = bf_search(pred_scores, labels[1:],start=0,end=np.percentile(pred_scores,99.9),step_num=10000,display_freq=50,verbose=False)
print('NSIBF-PRED')
print('best-f1', t[0])
print('precision', t[1])
print('recall', t[2])
print('accuracy',(t[3]+t[4])/(t[3]+t[4]+t[5]+t[6]))
print('TP', t[3])
print('TN', t[4])
print('FP', t[5])
print('FN', t[6])

#df_to_csv("testTraindf", train_df)



#train_labels = train_labels.astype(bool)
#val_labels = val_labels.astype(bool)
#test_labels = test_labels.astype(bool)
#
#
#normal_train_data = train_data[train_labels]
#validation_data = val_data[val_labels]
#normal_test_data = test_data[test_labels]
#
#print(normal_test_data)
