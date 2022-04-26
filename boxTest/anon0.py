# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 14:07:10 2021

@author: uanjum
"""

import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib.dates import DayLocator, HourLocator, DateFormatter, drange
#import datetime
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

swat0 = pd.read_csv('C:/Users/rossm/Documents/GitHub/test_nsibf/Final data/SWaT_Dataset_Attack_v0.csv') # read SWaT full dataset
wadi0 = pd.read_csv('C:/Users/rossm/Documents/GitHub/test_nsibf/Final data/WADI_attackdataLABLE.csv') # read WADI full dataset
wadi0 = wadi0.dropna(axis=1)

swat0['Timestamp'] =  pd.to_datetime(swat0[' Timestamp'], infer_datetime_format=True)
wadi0['Timestamp'] =  pd.to_datetime(wadi0['Timestamp'], format='%m/%d/%Y %M:%S.0')

# Test Plots
plt.plot(swat0['Timestamp'],swat0['AIT202'],color='green')
plt.plot(wadi0['Timestamp'],wadi0['1_FIT_001_PV'],color='red')
plt.tight_layout()
plt.show()
# ------------------------------------------------------------ #
# ------------------------------------------------------------ #
rng = np.random.RandomState(42)
clf = IsolationForest(max_samples=100, random_state=rng)
# ------------------------------------------------------------ #
# ------------------------------------------------------------ #
# ------------------------------------------------------------ #
# -SWAT------------------------------------------------------- #
# Split data into normal and attack data sets
# Extract training (normal)  
swat0_reg0 = swat0[swat0['Normal/Attack'] == "Normal"] # No attack
swat0_reg0 = swat0_reg0.drop([' Timestamp', 'Normal/Attack', 'Timestamp'], axis=1)
swat0_reg1 = swat0_reg0.to_numpy()
swat0_reg_lab0 = np.repeat(1, repeats=len(swat0_reg1))
# ------------------------------------------------------------ #
# Extract testing (attack)
swat0_att0 = swat0[swat0['Normal/Attack'] == "Attack"] # No attack
swat0_att0 = swat0_att0.drop([' Timestamp', 'Normal/Attack', 'Timestamp'], axis=1)
swat0_att1 = swat0_att0.to_numpy()
swat0_att_lab0 = np.repeat(-1, repeats=len(swat0_att1))
# ------------------------------------------------------------ #
clf.fit(swat0_reg1)
swat0_reg = clf.predict(swat0_reg1)
swat0_att = clf.predict(swat0_att1)
# ------------------------------------------------------------ #
swat0_reg_precision = precision_score(swat0_reg_lab0, swat0_reg)
swat0_reg_recall = recall_score(swat0_reg_lab0, swat0_reg)
swat0_reg_f1 = f1_score(swat0_reg_lab0, swat0_reg)
# ------------------------------------------------------------ #
swat0_att_precision = precision_score(swat0_att_lab0, swat0_att, average='weighted', labels=np.unique(swat0_att))
swat0_att_recall = recall_score(swat0_att_lab0, swat0_att, average='weighted', labels=np.unique(swat0_att))
swat0_att_f1 = f1_score(swat0_att_lab0, swat0_att, average='weighted', labels=np.unique(swat0_att))
# ------------------------------------------------------------ #
# ------------------------------------------------------------ #
# Combine Labels from nomal and attack data sets
swat0_pred = np.concatenate([swat0_reg, swat0_att])
swat0_act = np.concatenate([swat0_reg_lab0, swat0_att_lab0])
# ------------------------------------------------------------ #
swat0_precision = precision_score(swat0_pred, swat0_act)
swat0_recall = recall_score(swat0_pred, swat0_act)
swat0_f1 = f1_score(swat0_pred, swat0_act)
# ------------------------------------------------------------ #
conf_matrix = confusion_matrix(y_true=swat0_pred, y_pred=swat0_act)
# Print the confusion matrix using Matplotlib
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
# ------------------------------------------------------------ #
# ------------------------------------------------------------ #
# Alternative Method3: Train-Test (75-25%) Split dataset
swat0_1 = swat0.drop([' Timestamp', 'Normal/Attack', 'Timestamp'], axis=1)
swat0_2 = swat0_1.to_numpy()
swat0_lab = swat0['Normal/Attack'].to_numpy()
swat0_lab[swat0_lab == 'Normal'] = 1
swat0_lab[swat0_lab == 'Attack'] = -1
swat0_train, swat0_test, swat0_lab_train, swat0_lab_test = train_test_split(swat0_2, swat0_lab, test_size=0.25)

clf.fit(swat0_train)
swat0_pred0 = clf.predict(swat0_train)
swat0_pred1 = clf.predict(swat0_test)
# ------------------------------------------------------------ #
swat0_pred0_precision = precision_score(swat0_pred0, np.int64(swat0_lab_train))
swat0_pred0_recall = recall_score(swat0_pred0, np.int64(swat0_lab_train))
swat0_pred0_f1 = f1_score(swat0_pred0, np.int64(swat0_lab_train))

swat0_pred1_precision = precision_score(swat0_pred1, np.int64(swat0_lab_test))
swat0_pred1_recall = recall_score(swat0_pred1, np.int64(swat0_lab_test))
swat0_pred1_f1 = f1_score(swat0_pred1, np.int64(swat0_lab_test))
# ------------------------------------------------------------ #
# ------------------------------------------------------------ #
# ------------------------------------------------------------ #
# -WADI------------------------------------------------------- #
# Split data into normal and attack data sets
# Extract training (normal)  
wadi0_reg0 = wadi0[wadi0['Attack LABLE (1:No Attack, -1:Attack)'] == 1] # No attack
wadi0_reg0 = wadi0_reg0.drop(['Row ', 'Timestamp', 'Attack LABLE (1:No Attack, -1:Attack)', 'Date ', 'Time'], axis=1)
wadi0_reg1 = wadi0_reg0.to_numpy()
wadi0_reg_lab0 = np.repeat(1, repeats=len(wadi0_reg1))
# ------------------------------------------------------------ #
# Extract testing (attack)
wadi0_att0 = wadi0[wadi0['Attack LABLE (1:No Attack, -1:Attack)'] == -1] # No attack
wadi0_att0 = wadi0_att0.drop(['Row ', 'Timestamp', 'Attack LABLE (1:No Attack, -1:Attack)', 'Date ', 'Time'], axis=1)
wadi0_att1 = wadi0_att0.to_numpy()
wadi0_att_lab0 = np.repeat(-1, repeats=len(wadi0_att1))
# ------------------------------------------------------------ #
clf.fit(wadi0_reg1)
wadi0_reg = clf.predict(wadi0_reg1)
wadi0_att = clf.predict(wadi0_att1)
# ------------------------------------------------------------ #
wadi0_reg_precision = precision_score(wadi0_reg_lab0, wadi0_reg, average='weighted', labels=np.unique(wadi0_reg))
wadi0_reg_recall = recall_score(wadi0_reg_lab0, wadi0_reg)
wadi0_reg_f1 = f1_score(wadi0_reg_lab0, wadi0_reg)
# ------------------------------------------------------------ #
wadi0_att_precision = precision_score(wadi0_att_lab0, wadi0_att, average='weighted', labels=np.unique(wadi0_att))
wadi0_att_recall = recall_score(wadi0_att_lab0, wadi0_att, average='weighted', labels=np.unique(wadi0_att))
wadi0_att_f1 = f1_score(wadi0_att_lab0, wadi0_att, average='weighted', labels=np.unique(wadi0_att))
# ------------------------------------------------------------ #
# ------------------------------------------------------------ #
# Combine Labels from nomal and attack data sets
wadi0_pred = np.concatenate([wadi0_reg, wadi0_att])
wadi0_act = np.concatenate([wadi0_reg_lab0, wadi0_att_lab0])
# ------------------------------------------------------------ #
wadi0_precision = precision_score(wadi0_pred, wadi0_act)
wadi0_recall = recall_score(wadi0_pred, wadi0_act)
wadi0_f1 = f1_score(wadi0_pred, wadi0_act)
# ------------------------------------------------------------ #
conf_matrix = confusion_matrix(y_true=wadi0_pred, y_pred=wadi0_act)
# Print the confusion matrix using Matplotlib
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()
# ------------------------------------------------------------ #
# ------------------------------------------------------------ #
# Alternative Method3: Train-Test (75-25%) Split dataset
wadi0_1 = wadi0.drop(['Row ', 'Timestamp', 'Attack LABLE (1:No Attack, -1:Attack)', 'Date ', 'Time'], axis=1)
wadi0_2 = wadi0_1.to_numpy()
wadi0_lab = wadi0['Attack LABLE (1:No Attack, -1:Attack)'].to_numpy()
wadi0_train, wadi0_test, wadi0_lab_train, wadi0_lab_test = train_test_split(wadi0_2, wadi0_lab, test_size=0.25)

clf.fit(wadi0_train)
wadi0_pred0 = clf.predict(wadi0_train)
wadi0_pred1 = clf.predict(wadi0_test)
# ------------------------------------------------------------ #
wadi0_pred0_precision = precision_score(wadi0_pred0, wadi0_lab_train)
wadi0_pred0_recall = recall_score(wadi0_pred0, wadi0_lab_train)
wadi0_pred0_f1 = f1_score(wadi0_pred0, wadi0_lab_train)

wadi0_pred1_precision = precision_score(wadi0_pred1, wadi0_lab_test)
wadi0_pred1_recall = recall_score(wadi0_pred1, wadi0_lab_test)
wadi0_pred1_f1 = f1_score(wadi0_pred1, wadi0_lab_test)
# ------------------------------------------------------------ #







