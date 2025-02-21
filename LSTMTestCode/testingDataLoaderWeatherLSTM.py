import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import IPython
import IPython.display
import numpy as np
import tensorflow as tf
sys.path.append(r'C:/Users/rossm/Documents/GitHub/test_nsibf')
from testDataLoader import load_dataset, drop_columns

from framework.models import NSIBF
from framework.preprocessing.data_loader import load_wadi_data
from framework.preprocessing import normalize_and_encode_signals
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

test_dataset_path = r'/Users/rossm/Documents/GitHub/NSIBF/datasets/WADI/WADI_test.zip'
train_dataset_path = r'/Users/rossm/Documents/GitHub/NSIBF/datasets/WADI/WADI_train.zip'

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

train_df,val_df,test_df,signals = load_wadi_data()

#compression_opts = dict(method='zip',
#
#                        archive_name='out.csv')  
#
#train_df.to_csv('trainout.zip', index=False,
#
#          compression=compression_opts)  
#val_df.to_csv('valout.zip', index=False,
#
#          compression=compression_opts)  
#test_df.to_csv('testout.zip', index=False,
#
#          compression=compression_opts)  

#test_columns_actuators = ['Time', 'label', '1_LS_001_AL', '1_LS_002_AL','2_LS_001_AL', '2_LS_002_AL', 
#                 '2_P_001_STATUS', '2_P_002_STATUS', '1_MV_001_STATUS', '1_MV_002_STATUS', '1_MV_003_STATUS',
#                 '1_MV_004_STATUS', '1_P_001_STATUS','1_P_002_STATUS', '1_P_003_STATUS', '1_P_004_STATUS',
#                 '1_P_005_STATUS', '1_P_006_STATUS', '2_LS_101_AH', '2_LS_101_AL', '2_LS_201_AH', '2_LS_201_AL', 
#                 '2_LS_301_AH', '2_LS_301_AL', '2_LS_401_AH', '2_LS_401_AL', '2_LS_501_AH', '2_MCV_007_CO',
#                 '2_LS_501_AL', '2_LS_601_AH', '2_LS_601_AL', '2_MV_001_STATUS', '2_MV_002_STATUS','2_MV_003_STATUS', '2_MV_004_STATUS',
#                 '2_MV_005_STATUS', '2_MV_006_STATUS', '2_MV_009_STATUS',
#                 '2_MV_101_STATUS', '2_MV_201_STATUS', '2_MV_301_STATUS', '2_MV_401_STATUS', 
#                 '2_MV_501_STATUS', '2_MV_601_STATUS', '2_P_003_STATUS', '2_P_004_STATUS', '2_PIC_003_SP',
#                 '2_SV_101_STATUS', '2_SV_201_STATUS', '2_SV_301_STATUS', '2_SV_401_STATUS', '2_SV_501_STATUS',
#                 '2_SV_601_STATUS', '3_AIT_001_PV', '3_LS_001_AL', '3_MV_001_STATUS', '3_MV_002_STATUS', '3_MV_003_STATUS',
#                 '3_P_001_STATUS', '3_P_002_STATUS', '3_P_003_STATUS', '3_P_004_STATUS', 'PLANT_START_STOP_LOG', '1_MV_001_STATUS=1',
#                 '1_MV_001_STATUS=0', '1_MV_001_STATUS=2', '1_MV_004_STATUS=1', '1_MV_004_STATUS=0', '1_MV_004_STATUS=2',
#                 '1_P_001_STATUS=1', '1_P_001_STATUS=2', '1_P_003_STATUS=1', '1_P_003_STATUS=2', '1_P_005_STATUS=2',
#                 '1_P_005_STATUS=1', '2_LS_101_AH=0', '2_LS_101_AH=1', '2_LS_101_AL=0', '2_LS_101_AL=1', '2_LS_201_AH=0',
#                 '2_LS_201_AH=1', '2_LS_201_AL=0', '2_LS_201_AL=1', '2_LS_301_AH=0', '2_LS_301_AH=1', '2_LS_301_AL=0',
#                 '2_LS_301_AL=1', '2_LS_401_AH=0', '2_LS_401_AH=1', '2_LS_401_AL=0', '2_LS_401_AL=1', '2_LS_501_AH=0',
#                 '2_LS_501_AH=1', '2_LS_501_AL=0', '2_LS_501_AL=1', '2_LS_601_AH=0', '2_LS_601_AH=1', '2_LS_601_AL=0',
#                 '2_LS_601_AL=1', '2_MV_003_STATUS=2', '2_MV_003_STATUS=0', '2_MV_003_STATUS=1', '2_MV_006_STATUS=2', '2_MV_006_STATUS=0',
#                 '2_MV_006_STATUS=1', '2_MV_101_STATUS=1', '2_MV_101_STATUS=0', '2_MV_101_STATUS=2', '2_MV_201_STATUS=1', '2_MV_201_STATUS=0',
#                 '2_MV_201_STATUS=2', '2_MV_301_STATUS=1', '2_MV_301_STATUS=0', '2_MV_301_STATUS=2', '2_MV_401_STATUS=1', '2_MV_401_STATUS=2',
#                 '2_MV_401_STATUS=0', '2_MV_501_STATUS=1', '2_MV_501_STATUS=0', '2_MV_501_STATUS=2', '2_MV_601_STATUS=1', '2_MV_601_STATUS=0',
#                 '2_MV_601_STATUS=2', '2_P_003_STATUS=2', '2_P_003_STATUS=1']

#train_columns_actuators = ['2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS', '1_MV_001_STATUS', '1_MV_004_STATUS', '1_P_001_STATUS', '1_P_003_STATUS', 
#                 '1_P_005_STATUS', '2_LS_101_AH', '2_LS_101_AL', '2_LS_201_AH', '2_LS_201_AL', 
#                 '2_LS_301_AH', '2_LS_301_AL', '2_LS_401_AH', '2_LS_401_AL', '2_LS_501_AH', 
#                 '2_LS_501_AL', '2_LS_601_AH', '2_LS_601_AL', '2_MV_003_STATUS', '2_MV_006_STATUS',
#                 '2_MV_101_STATUS', '2_MV_201_STATUS', '2_MV_301_STATUS', '2_MV_401_STATUS', 
#                 '2_MV_501_STATUS', '2_MV_601_STATUS', '2_P_003_STATUS']

#train_columns_actuators = ['1_LS_001_AL', '1_LS_002_AL','2_LS_001_AL', '2_LS_002_AL', 
#                 '2_P_001_STATUS', '2_P_002_STATUS', '1_MV_001_STATUS', '1_MV_002_STATUS', '1_MV_003_STATUS',
#                 '1_MV_004_STATUS', '1_P_001_STATUS','1_P_002_STATUS', '1_P_003_STATUS', '1_P_004_STATUS',
#                 '1_P_005_STATUS', '1_P_006_STATUS', '2_LS_101_AH', '2_LS_101_AL', '2_LS_201_AH', '2_LS_201_AL', 
#                 '2_LS_301_AH', '2_LS_301_AL', '2_LS_401_AH', '2_LS_401_AL', '2_LS_501_AH', '2_MCV_007_CO',
#                 '2_LS_501_AL', '2_LS_601_AH', '2_LS_601_AL', '2_MV_001_STATUS', '2_MV_002_STATUS','2_MV_003_STATUS', '2_MV_004_STATUS',
#                 '2_MV_005_STATUS', '2_MV_006_STATUS', '2_MV_009_STATUS',
#                 '2_MV_101_STATUS', '2_MV_201_STATUS', '2_MV_301_STATUS', '2_MV_401_STATUS', 
#                 '2_MV_501_STATUS', '2_MV_601_STATUS', '2_P_003_STATUS', '2_P_004_STATUS', '2_PIC_003_SP',
#                 '2_SV_101_STATUS', '2_SV_201_STATUS', '2_SV_301_STATUS', '2_SV_401_STATUS', '2_SV_501_STATUS',
#                 '2_SV_601_STATUS', '3_AIT_001_PV', '3_LS_001_AL', '3_MV_001_STATUS', '3_MV_002_STATUS', '3_MV_003_STATUS',
#                 '3_P_001_STATUS', '3_P_002_STATUS', '3_P_003_STATUS', '3_P_004_STATUS', 'PLANT_START_STOP_LOG', '1_MV_001_STATUS=1',
#                 '1_MV_001_STATUS=0', '1_MV_001_STATUS=2', '1_MV_004_STATUS=1', '1_MV_004_STATUS=0', '1_MV_004_STATUS=2',
#                 '1_P_001_STATUS=1', '1_P_001_STATUS=2', '1_P_003_STATUS=1', '1_P_003_STATUS=2', '1_P_005_STATUS=2',
#                 '1_P_005_STATUS=1', '2_LS_101_AH=0', '2_LS_101_AH=1', '2_LS_101_AL=0', '2_LS_101_AL=1', '2_LS_201_AH=0',
#                 '2_LS_201_AH=1', '2_LS_201_AL=0', '2_LS_201_AL=1', '2_LS_301_AH=0', '2_LS_301_AH=1', '2_LS_301_AL=0',
#                 '2_LS_301_AL=1', '2_LS_401_AH=0', '2_LS_401_AH=1', '2_LS_401_AL=0', '2_LS_401_AL=1', '2_LS_501_AH=0',
#                 '2_LS_501_AH=1', '2_LS_501_AL=0', '2_LS_501_AL=1', '2_LS_601_AH=0', '2_LS_601_AH=1', '2_LS_601_AL=0',
#                 '2_LS_601_AL=1', '2_MV_003_STATUS=2', '2_MV_003_STATUS=0', '2_MV_003_STATUS=1', '2_MV_006_STATUS=2', '2_MV_006_STATUS=0',
#                 '2_MV_006_STATUS=1', '2_MV_101_STATUS=1', '2_MV_101_STATUS=0', '2_MV_101_STATUS=2', '2_MV_201_STATUS=1', '2_MV_201_STATUS=0',
#                 '2_MV_201_STATUS=2', '2_MV_301_STATUS=1', '2_MV_301_STATUS=0', '2_MV_301_STATUS=2', '2_MV_401_STATUS=1', '2_MV_401_STATUS=2',
#                 '2_MV_401_STATUS=0', '2_MV_501_STATUS=1', '2_MV_501_STATUS=0', '2_MV_501_STATUS=2', '2_MV_601_STATUS=1', '2_MV_601_STATUS=0',
#                 '2_MV_601_STATUS=2', '2_P_003_STATUS=2', '2_P_003_STATUS=1']

#test_df = test_df.drop('Time', axis=1)
#test_df = test_df.drop(test_columns_actuators, axis=1)
#val_df = val_df.drop(train_columns_actuators,axis=1)
#train_df = train_df.drop(train_columns_actuators, axis=1)

num_features = train_df.shape[1]
print(test_df['3_AIT_002_PV'].max())
print(test_df['3_AIT_002_PV'].min())
print(test_df['3_AIT_002_PV'].std())
train_df = normalize_and_encode_signals(train_df,signals,scaler='min_max')
val_df = normalize_and_encode_signals(val_df,signals,scaler='min_max') 
test_df = normalize_and_encode_signals(test_df,signals,scaler='min_max')



#UP_test_df = load_dataset(test_dataset_path)
#UP_train_df = load_dataset(train_dataset_path)

#test_df, train_df = drop_columns(UP_test_df, UP_train_df)

#dataframes = [test_df, train_df]

#df = pd.concat(dataframes)

#df = df.loc[:, :'2A_AIT_004_PV']
#
#test_df = test_df.loc[:, :'2A_AIT_004_PV']
#
#train_df = train_df.loc[:, :'2A_AIT_004_PV']

#n = len(df)
#train_df = df[0:int(n*0.7)]
#val_df = df[int(n*0.7):int(n*0.9)]
#test_df = df[int(n*0.9):]
#
#print(train_df, val_df, test_df)
#


#num_features_train = train_df.shape[1]
#
#num_features_test = test_df.shape[1]


#train_mean = train_df.mean()
#train_std = train_df.std()
#
#train_df = (train_df - train_mean) / train_std
#val_df = (val_df - train_mean) / train_std
#test_df = (test_df - train_mean) / train_std

#df_std = (df - train_mean) / train_std
#df_std = df_std.melt(var_name='Column', value_name='Normalized')
#plt.figure(figsize=(12, 6))
#ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
#_ = ax.set_xticklabels(df.keys(), rotation=90)

#plt.show()

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window

def plot(self, model=None, plot_col='1_AIT_001_PV', max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(max_n, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)

    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index

    if label_col_index is None:
      continue

    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)

    if n == 0:
      plt.legend()

  plt.xlabel('Time [s]')

WindowGenerator.plot = plot

def make_dataset(self, data):
  data = np.array(data, dtype=np.float32)
  ds = tf.keras.utils.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)

  return ds

WindowGenerator.make_dataset = make_dataset

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.test = test
WindowGenerator.example = example


MAX_EPOCHS = 5

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      callbacks=[early_stopping])
  return history

OUT_STEPS = 24
multi_window = WindowGenerator(input_width=24,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)

multi_window.plot()

multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units].
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps*features].
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features].
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_lstm_model, multi_window)

IPython.display.clear_output()

multi_val_performance = {}
multi_performance = {}


multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.train)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)
multi_window.plot(multi_lstm_model)
plt.show()