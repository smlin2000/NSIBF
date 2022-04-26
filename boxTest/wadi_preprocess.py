# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:55:12 2022

@author: uanjum
"""
import pandas as pd
import numpy as np
import random
import math
import zipfile

from enum import Enum

class SignalSource(Enum):
    sensor = 101
    controller = 102
    other = 103


class BaseSignal(object):
    '''
    The base signal class
    
    :param name: the name of the signal
    :param source: the source of the signal, can be SignalSource.sensor, SignalSource.controller or SignalSource.other
    :param isInput: whether it is an input signal
    :param isOutput: whether it is an output signal
    :param measure_point: the measurement point of a sensor, two sensors can share one measurement point for redundancy. If set to None, then the measurement point will be set to the name of the signal
        (default is None)
    '''


    def __init__(self, name, source, isInput, isOutput, measure_point=None):
        '''
        Constructor
        '''
        self.name = name
        self.source = source
        self.isInput = isInput
        self.isOutput = isOutput
        if measure_point is None:
            self.measure_point = name
        else:
            self.measure_point = measure_point
            
class ContinousSignal(BaseSignal):
    '''
    The class for signals which take continuous values
    
    :param name: the name of the signal
    :param source: the source of the signal, can be SignalSource.sensor, SignalSource.controller or SignalSource.other
    :param isInput: whether it is an input signal
    :param isOutput: whether it is an output signal
    :param min_value: minimal value for the signal
        (default is None)
    :param max_value: the maximal value for the signal
        (default is None)
    :param mean_value: mean for the signal value distribution
        (default is None)
    :param std_value: std for the signal value distribution
        (default is None)
    '''


    def __init__(self, name, source, isInput, isOutput, min_value=None, max_value=None, mean_value=None, std_value=None):
        '''
        Constructor
        '''
        super().__init__(name, source, isInput, isOutput)
        self.min_value = min_value
        self.max_value = max_value
        self.mean_value = mean_value
        self.std_value = std_value


class DiscreteSignal(BaseSignal):
    '''
    The class for signals which take discrete values
    
    :param name: the name of the signal
    :param source: the source of the signal, can be SignalSource.sensor, SignalSource.controller or SignalSource.other
    :param isInput: whether it is an input signal
    :param isOutput: whether it is an output signal
    :param values: the list of possible values for the signal
    '''


    def __init__(self, name, source, isInput, isOutput, values):
        '''
        Constructor
        '''
        super().__init__(name, source, isInput, isOutput)
        self.values = values
    
    def get_onehot_feature_names(self):
        """
        Get the one-hot encoding feature names for the possible values of the signal
        
        :return name_list: the list of one-hot encoding feature names
        """
        name_list = []
        for value in self.values:
            name_list.append(self.name+'='+str(value))
        return name_list
    
    def get_feature_name(self,value):
        """
        Get the one-hot encoding feature name for a possible value of the signal
        
        :param: value: a possible value of the signal
        :return name: the one-hot encoding feature name of the given value
        """
        return self.name+'='+str(value)

wadi0 = pd.read_csv('C:/Users/uanjum/Box/Datasets/Anomaly Detection/Final data/WADI_attackdataLABLE.csv') # read WADI full dataset
wadi0 = wadi0.dropna(axis=1)
    
z_tr = zipfile.ZipFile('C:/Users/uanjum/Box/Coding/NSIBF-main/datasets/WADI/WADI_train.zip', "r")
f_tr = z_tr.open(z_tr.namelist()[0])
train_df=pd.read_csv(f_tr)
f_tr.close()
z_tr.close()

z_tr = zipfile.ZipFile('C:/Users/uanjum/Box/Coding/NSIBF-main/datasets/WADI/WADI_test.zip', "r")
f_tr = z_tr.open(z_tr.namelist()[0])
test_df=pd.read_csv(f_tr)
f_tr.close()
z_tr.close()

train_df=train_df.fillna(method='ffill')
test_df.loc[test_df['label']>=1,'label']=1
test_df=test_df.fillna(method='ffill')

sensors = ['1_AIT_001_PV', '1_AIT_002_PV', '1_AIT_003_PV', '1_AIT_004_PV', 
           '1_AIT_005_PV', '1_FIT_001_PV', '1_LT_001_PV', '2_DPIT_001_PV', 
           '2_FIC_101_CO', '2_FIC_101_PV', '2_FIC_101_SP', '2_FIC_201_CO', 
           '2_FIC_201_PV', '2_FIC_201_SP', '2_FIC_301_CO', '2_FIC_301_PV', 
           '2_FIC_301_SP', '2_FIC_401_CO', '2_FIC_401_PV', '2_FIC_401_SP', 
           '2_FIC_501_CO', '2_FIC_501_PV', '2_FIC_501_SP', '2_FIC_601_CO', 
           '2_FIC_601_PV', '2_FIC_601_SP', '2_FIT_001_PV', '2_FIT_002_PV', 
           '2_FIT_003_PV', '2_FQ_101_PV', '2_FQ_201_PV', '2_FQ_301_PV', '2_FQ_401_PV', 
           '2_FQ_501_PV', '2_FQ_601_PV', '2_LT_001_PV', '2_LT_002_PV', '2_MCV_101_CO', 
           '2_MCV_201_CO', '2_MCV_301_CO', '2_MCV_401_CO', '2_MCV_501_CO', '2_MCV_601_CO', 
           '2_P_003_SPEED', '2_P_004_SPEED', '2_PIC_003_CO', '2_PIC_003_PV', '2_PIT_001_PV', 
           '2_PIT_002_PV', '2_PIT_003_PV', '2A_AIT_001_PV', '2A_AIT_002_PV', '2A_AIT_003_PV', 
           '2A_AIT_004_PV', '2B_AIT_001_PV', '2B_AIT_002_PV', '2B_AIT_003_PV', '2B_AIT_004_PV', 
           '3_AIT_001_PV', '3_AIT_002_PV', '3_AIT_003_PV', '3_AIT_004_PV', '3_AIT_005_PV', 
           '3_FIT_001_PV', '3_LT_001_PV', 'LEAK_DIFF_PRESSURE', 'TOTAL_CONS_REQUIRED_FLOW']


actuators = ['1_MV_001_STATUS', '1_MV_004_STATUS', '1_P_001_STATUS', '1_P_003_STATUS', 
             '1_P_005_STATUS', '2_LS_101_AH', '2_LS_101_AL', '2_LS_201_AH', '2_LS_201_AL', 
             '2_LS_301_AH', '2_LS_301_AL', '2_LS_401_AH', '2_LS_401_AL', '2_LS_501_AH', 
             '2_LS_501_AL', '2_LS_601_AH', '2_LS_601_AL', '2_MV_003_STATUS', '2_MV_006_STATUS',
             '2_MV_101_STATUS', '2_MV_201_STATUS', '2_MV_301_STATUS', '2_MV_401_STATUS', 
             '2_MV_501_STATUS', '2_MV_601_STATUS', '2_P_003_STATUS']

signals = []
for name in sensors:
    signals.append( ContinousSignal(name, SignalSource.sensor, isInput=True, isOutput=True, 
                                    min_value=train_df[name].min(), max_value=train_df[name].max(),
                                    mean_value=train_df[name].mean(), std_value=train_df[name].std()) )
for name in actuators:
    signals.append( DiscreteSignal(name, SignalSource.controller, isInput=True, isOutput=False, 
                                        values=train_df[name].unique()) )


pos = len(train_df)*3//4
val_df = train_df.loc[pos:,:]
val_df = val_df.reset_index(drop=True)

train_df0 = train_df.loc[:pos,:]
train_df1 = train_df0.reset_index(drop=True)

seqL = 12
wl = seqL
input_range = seqL*3       
targets = []
covariates = []
covariates0 = []
for signal in signals:
    if signal.isInput==True:
        if isinstance(signal, ContinousSignal):
            covariates.append(signal.name)
        if isinstance(signal, DiscreteSignal):
            covariates.extend(signal.get_onehot_feature_names())
            covariates0.append(signal.get_onehot_feature_names())
    if signal.isOutput==True:
        targets.append(signal.name)

df = train_df1
onehot_entries = {}
for signal in signals:
    if isinstance(signal, DiscreteSignal):
        onehot_entries[signal.name] = signal.get_onehot_feature_names()
        for value in signal.values:
            new_entry = signal.get_feature_name(value)
            df[new_entry] = 0
            df.loc[df[signal.name]==value,new_entry] = 1
    if isinstance(signal, ContinousSignal):
        df[signal.name] = df[signal.name].astype(float)
        if signal.max_value != signal.min_value:
            df[signal.name]=df[signal.name].apply(lambda x:float(x-signal.min_value)/float(signal.max_value-signal.min_value))
        

df = train_df.copy()
x_feats,u_feats,y_feats,z_feats = [],[],[],[]











