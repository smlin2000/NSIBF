from cmath import nan
import sys
sys.path.append(r'C:/Users/rossm/Documents/GitHub/test_nsibf')
import zipfile
import pandas as pd
import numpy as np

def load_dataset(path):
    try:
        z_file = zipfile.ZipFile(path, "r")
    except TypeError:
        print("Invalid path")
        exit(-1)

    file = z_file.open(z_file.namelist()[0])
    df = pd.read_csv(file)
    file.close()
    z_file.close()

    #print(df.head(10))

    return df

def drop_columns(test_df, train_df):

    test_df['Time'] = pd.to_datetime(test_df['Time'], infer_datetime_format=True)

    test_df_columns = list(test_df)

    train_df_columns = list(train_df)

    test_actuators = []

    train_actuators = []

    index = 0

    for i in test_df_columns:
        if test_df[i][385] == 1 or test_df[i][385] == 0 or test_df[i][385] == 2 or pd.isnull(test_df[i][385]):
            test_actuators.append(test_df.columns[index])
        index += 1
    
    index = 0

    #for i in train_df_columns:
    #    if train_df[i][1387] == 1 or train_df[i][1387] == 0 or train_df[i][1387] == 2 or str(train_df[i].max()).isdigit() or pd.isnull(train_df[i][1387]):
    #        train_actuators.append(train_df.columns[index])
    #    index += 1

    for i in train_df_columns:
        if str(train_df[i].max()).isdigit() or str(train_df[i].min()).isdigit() or pd.isnull(train_df[i][0]):
            train_actuators.append(train_df.columns[index])
        index += 1
    
    #print(test_actuators)

    #print(train_actuators)

    #test_columns_actuators = ['Time', 'label', '2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS', '1_MV_001_STATUS', '1_MV_004_STATUS', '1_P_001_STATUS', '1_P_003_STATUS', 
    #             '1_P_005_STATUS', '2_LS_101_AH', '2_LS_101_AL', '2_LS_201_AH', '2_LS_201_AL', 
    #             '2_LS_301_AH', '2_LS_301_AL', '2_LS_401_AH', '2_LS_401_AL', '2_LS_501_AH', 
    #             '2_LS_501_AL', '2_LS_601_AH', '2_LS_601_AL', '2_MV_003_STATUS', '2_MV_006_STATUS',
    #             '2_MV_101_STATUS', '2_MV_201_STATUS', '2_MV_301_STATUS', '2_MV_401_STATUS', 
    #             '2_MV_501_STATUS', '2_MV_601_STATUS', '2_P_003_STATUS']

    #train_columns_actuators = ['2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS', '1_MV_001_STATUS', '1_MV_004_STATUS', '1_P_001_STATUS', '1_P_003_STATUS', 
    #             '1_P_005_STATUS', '2_LS_101_AH', '2_LS_101_AL', '2_LS_201_AH', '2_LS_201_AL', 
    #             '2_LS_301_AH', '2_LS_301_AL', '2_LS_401_AH', '2_LS_401_AL', '2_LS_501_AH', 
    #             '2_LS_501_AL', '2_LS_601_AH', '2_LS_601_AL', '2_MV_003_STATUS', '2_MV_006_STATUS',
    #             '2_MV_101_STATUS', '2_MV_201_STATUS', '2_MV_301_STATUS', '2_MV_401_STATUS', 
    #             '2_MV_501_STATUS', '2_MV_601_STATUS', '2_P_003_STATUS']

    test_df = test_df.drop('Time', axis=1)
    test_df = test_df.drop(test_actuators, axis=1)
    train_df = train_df.drop(train_actuators, axis=1)

    return test_df, train_df

#def normalize_data(test_df, train_df):
