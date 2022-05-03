from cmath import nan
import sys
sys.path.append(r'C:/Users/rossm/Documents/GitHub/test_nsibf')
import zipfile
import pandas as pd
import numpy as np

def load_dataset(path, isTrain = False):
    try:
        z_file = zipfile.ZipFile(path, "r")
    except TypeError:
        print("Invalid path")
        exit(-1)

    file = z_file.open(z_file.namelist()[0])
    df = pd.read_csv(file)
    file.close()
    z_file.close()

    if isTrain:
        print("Train df")
        train_df = df.copy()
        train_df=train_df.fillna(method='ffill')
        #print(len(train_df))
        # len(train_df)*3%4
        pos = len(train_df)*3//4
        # pos = 181440
        val_df = train_df.loc[pos:,:]
        val_df = val_df.reset_index(drop=True)
        train_df = train_df.loc[:pos,:]
        train_df = train_df.reset_index(drop=True)
        return train_df, val_df
    else:
        print("Test df")
        test_df = df.copy()
        test_df.loc[test_df['label']>=1,'label']=1
        test_df=test_df.fillna(method='ffill')
        return test_df

def drop_columns(train_df, val_df, test_df):

    test_df['Time'] = pd.to_datetime(test_df['Time'], infer_datetime_format=True)

    test_df_columns = list(test_df)

    train_df_columns = list(train_df)

    test_actuators = []

    train_actuators = []

    index = 0

    #for i in test_df_columns:
    #    if test_df[i][385] == 1 or test_df[i][385] == 0 or test_df[i][385] == 2 or pd.isnull(test_df[i][385]):
    #        test_actuators.append(test_df.columns[index])
    #    index += 1

    for i in test_df_columns:
        if str(test_df[i].max()).isdigit() or str(test_df[i].min()).isdigit() or pd.isnull(test_df[i][0]):
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

    #test_columns_actuators = ['Time', 'label', '1_LS_001_AL', '1_LS_002_AL','2_LS_001_AL', '2_LS_002_AL', 
    #             '2_P_001_STATUS', '2_P_002_STATUS', '1_MV_001_STATUS', '1_MV_002_STATUS', '1_MV_003_STATUS',
    #             '1_MV_004_STATUS', '1_P_001_STATUS','1_P_002_STATUS', '1_P_003_STATUS', '1_P_004_STATUS',
    #             '1_P_005_STATUS', '1_P_006_STATUS', '2_LS_101_AH', '2_LS_101_AL', '2_LS_201_AH', '2_LS_201_AL', 
    #             '2_LS_301_AH', '2_LS_301_AL', '2_LS_401_AH', '2_LS_401_AL', '2_LS_501_AH', '2_MCV_007_CO',
    #             '2_LS_501_AL', '2_LS_601_AH', '2_LS_601_AL', '2_MV_001_STATUS', '2_MV_002_STATUS','2_MV_003_STATUS', '2_MV_004_STATUS',
    #             '2_MV_005_STATUS', '2_MV_006_STATUS', '2_MV_009_STATUS',
    #             '2_MV_101_STATUS', '2_MV_201_STATUS', '2_MV_301_STATUS', '2_MV_401_STATUS', 
    #             '2_MV_501_STATUS', '2_MV_601_STATUS', '2_P_003_STATUS', '2_P_004_STATUS', '2_PIC_003_SP',
    #             '2_SV_101_STATUS', '2_SV_201_STATUS', '2_SV_301_STATUS', '2_SV_401_STATUS', '2_SV_501_STATUS',
    #             '2_SV_601_STATUS', '3_AIT_001_PV', '3_LS_001_AL', '3_MV_001_STATUS', '3_MV_002_STATUS', '3_MV_003_STATUS',
    #             '3_P_001_STATUS', '3_P_002_STATUS', '3_P_003_STATUS', '3_P_004_STATUS', 'PLANT_START_STOP_LOG', '1_MV_001_STATUS=1',
    #             '1_MV_001_STATUS=0', '1_MV_001_STATUS=2', '1_MV_004_STATUS=1', '1_MV_004_STATUS=0', '1_MV_004_STATUS=2',
    #             '1_P_001_STATUS=1', '1_P_001_STATUS=2', '1_P_003_STATUS=1', '1_P_003_STATUS=2', '1_P_005_STATUS=2',
    #             '1_P_005_STATUS=1', '2_LS_101_AH=0', '2_LS_101_AH=1', '2_LS_101_AL=0', '2_LS_101_AL=1', '2_LS_201_AH=0',
    #             '2_LS_201_AH=1', '2_LS_201_AL=0', '2_LS_201_AL=1', '2_LS_301_AH=0', '2_LS_301_AH=1', '2_LS_301_AL=0',
    #             '2_LS_301_AL=1', '2_LS_401_AH=0', '2_LS_401_AH=1', '2_LS_401_AL=0', '2_LS_401_AL=1', '2_LS_501_AH=0',
    #             '2_LS_501_AH=1', '2_LS_501_AL=0', '2_LS_501_AL=1', '2_LS_601_AH=0', '2_LS_601_AH=1', '2_LS_601_AL=0',
    #             '2_LS_601_AL=1', '2_MV_003_STATUS=2', '2_MV_003_STATUS=0', '2_MV_003_STATUS=1', '2_MV_006_STATUS=2', '2_MV_006_STATUS=0',
    #             '2_MV_006_STATUS=1', '2_MV_101_STATUS=1', '2_MV_101_STATUS=0', '2_MV_101_STATUS=2', '2_MV_201_STATUS=1', '2_MV_201_STATUS=0',
    #             '2_MV_201_STATUS=2', '2_MV_301_STATUS=1', '2_MV_301_STATUS=0', '2_MV_301_STATUS=2', '2_MV_401_STATUS=1', '2_MV_401_STATUS=2',
    #             '2_MV_401_STATUS=0', '2_MV_501_STATUS=1', '2_MV_501_STATUS=0', '2_MV_501_STATUS=2', '2_MV_601_STATUS=1', '2_MV_601_STATUS=0',
    #             '2_MV_601_STATUS=2', '2_P_003_STATUS=2', '2_P_003_STATUS=1']
#
    #train_columns_actuators = ['2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS', '1_MV_001_STATUS', '1_MV_004_STATUS', '1_P_001_STATUS', '1_P_003_STATUS', 
    #             '1_P_005_STATUS', '2_LS_101_AH', '2_LS_101_AL', '2_LS_201_AH', '2_LS_201_AL', 
    #             '2_LS_301_AH', '2_LS_301_AL', '2_LS_401_AH', '2_LS_401_AL', '2_LS_501_AH', 
    #             '2_LS_501_AL', '2_LS_601_AH', '2_LS_601_AL', '2_MV_003_STATUS', '2_MV_006_STATUS',
    #             '2_MV_101_STATUS', '2_MV_201_STATUS', '2_MV_301_STATUS', '2_MV_401_STATUS', 
    #             '2_MV_501_STATUS', '2_MV_601_STATUS', '2_P_003_STATUS']

    #train_columns_actuators = ['1_LS_001_AL', '1_LS_002_AL','2_LS_001_AL', '2_LS_002_AL', 
    #             '2_P_001_STATUS', '2_P_002_STATUS', '1_MV_001_STATUS', '1_MV_002_STATUS', '1_MV_003_STATUS',
    #             '1_MV_004_STATUS', '1_P_001_STATUS','1_P_002_STATUS', '1_P_003_STATUS', '1_P_004_STATUS',
    #             '1_P_005_STATUS', '1_P_006_STATUS', '2_LS_101_AH', '2_LS_101_AL', '2_LS_201_AH', '2_LS_201_AL', 
    #             '2_LS_301_AH', '2_LS_301_AL', '2_LS_401_AH', '2_LS_401_AL', '2_LS_501_AH', '2_MCV_007_CO',
    #             '2_LS_501_AL', '2_LS_601_AH', '2_LS_601_AL', '2_MV_001_STATUS', '2_MV_002_STATUS','2_MV_003_STATUS', '2_MV_004_STATUS',
    #             '2_MV_005_STATUS', '2_MV_006_STATUS', '2_MV_009_STATUS',
    #             '2_MV_101_STATUS', '2_MV_201_STATUS', '2_MV_301_STATUS', '2_MV_401_STATUS', 
    #             '2_MV_501_STATUS', '2_MV_601_STATUS', '2_P_003_STATUS', '2_P_004_STATUS', '2_PIC_003_SP',
    #             '2_SV_101_STATUS', '2_SV_201_STATUS', '2_SV_301_STATUS', '2_SV_401_STATUS', '2_SV_501_STATUS',
    #             '2_SV_601_STATUS', '3_AIT_001_PV', '3_LS_001_AL', '3_MV_001_STATUS', '3_MV_002_STATUS', '3_MV_003_STATUS',
    #             '3_P_001_STATUS', '3_P_002_STATUS', '3_P_003_STATUS', '3_P_004_STATUS', 'PLANT_START_STOP_LOG', '1_MV_001_STATUS=1',
    #             '1_MV_001_STATUS=0', '1_MV_001_STATUS=2', '1_MV_004_STATUS=1', '1_MV_004_STATUS=0', '1_MV_004_STATUS=2',
    #             '1_P_001_STATUS=1', '1_P_001_STATUS=2', '1_P_003_STATUS=1', '1_P_003_STATUS=2', '1_P_005_STATUS=2',
    #             '1_P_005_STATUS=1', '2_LS_101_AH=0', '2_LS_101_AH=1', '2_LS_101_AL=0', '2_LS_101_AL=1', '2_LS_201_AH=0',
    #             '2_LS_201_AH=1', '2_LS_201_AL=0', '2_LS_201_AL=1', '2_LS_301_AH=0', '2_LS_301_AH=1', '2_LS_301_AL=0',
    #             '2_LS_301_AL=1', '2_LS_401_AH=0', '2_LS_401_AH=1', '2_LS_401_AL=0', '2_LS_401_AL=1', '2_LS_501_AH=0',
    #             '2_LS_501_AH=1', '2_LS_501_AL=0', '2_LS_501_AL=1', '2_LS_601_AH=0', '2_LS_601_AH=1', '2_LS_601_AL=0',
    #             '2_LS_601_AL=1', '2_MV_003_STATUS=2', '2_MV_003_STATUS=0', '2_MV_003_STATUS=1', '2_MV_006_STATUS=2', '2_MV_006_STATUS=0',
    #             '2_MV_006_STATUS=1', '2_MV_101_STATUS=1', '2_MV_101_STATUS=0', '2_MV_101_STATUS=2', '2_MV_201_STATUS=1', '2_MV_201_STATUS=0',
    #             '2_MV_201_STATUS=2', '2_MV_301_STATUS=1', '2_MV_301_STATUS=0', '2_MV_301_STATUS=2', '2_MV_401_STATUS=1', '2_MV_401_STATUS=2',
    #             '2_MV_401_STATUS=0', '2_MV_501_STATUS=1', '2_MV_501_STATUS=0', '2_MV_501_STATUS=2', '2_MV_601_STATUS=1', '2_MV_601_STATUS=0',
    #             '2_MV_601_STATUS=2', '2_P_003_STATUS=2', '2_P_003_STATUS=1']

    test_df = test_df.drop('Time', axis=1)
    train_df = train_df.drop(train_actuators, axis=1)
    val_df = val_df.drop(train_actuators, axis=1)
    test_df = test_df.drop(test_actuators, axis=1)

    return train_df, val_df, test_df 

def df_to_csv(train_df, val_df, test_df):
    #exports all 3 dataframes as csv files in zip files for viewing dataframes
    compression_opts_train = dict(method='zip', archive_name ='trainout.csv')  

    compression_opts_val = dict(method = 'zip', archive_name = 'valout.csv')

    compression_opts_test = dict(method = 'zip', archive_name = 'testout.csv')

    train_df.to_csv('trainout.zip', index=False, compression=compression_opts_train)

    val_df.to_csv('valout.zip', index=False, compression=compression_opts_val)  

    test_df.to_csv('testout.zip', index=False, compression=compression_opts_test)

    return


def normalize_data(df_original):
    #data normalization using max-min method
    df = df_original.copy()
    for column in df:
        df[column] = df[column].astype(float)
        #print(df[column])
        minval = float(df[column].min())
        maxval = float(df[column].max())
        if maxval != minval:
            df[column]=df[column].apply(lambda x:float(x-minval)/float(maxval-minval))
    
    return df
