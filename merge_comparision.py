#here we use the mlp_merge demo

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.layers import Convolution2D, MaxPooling2D,Flatten,TimeDistributed
from keras.utils.visualize_util import plot
import pandas as pd
from keras.layers import Merge
import h5py
from keras.models import model_from_json


# Convolution
nb_row=3
nb_col=1
nb_filter = 64
pool_size = 1

# LSTM
lstm_output_size = 64

# Training
batch_size = 32
nb_epoch = 20

n_frames=4
n_hours=24
n_cols=1


#load training raw data
rawdata_train = pd.read_csv("EURUSD60_train.csv",encoding='gbk')
rawdata_train1=rawdata_train[[2]] # get one column
rawdata_train2=rawdata_train[[3]] # get one column



data=rawdata_train1.as_matrix()
data2=rawdata_train2.as_matrix()

temp_dataX= []
temp_dataX2= []
temp_dataY= []



temp_dataX=data[0:(data.shape[0]-n_hours)]
temp_dataX2=data2[0:(data.shape[0]-n_hours)]
temp_dataY=data[n_frames*n_hours:data.shape[0]]
temp_dataX=np.reshape(temp_dataX,-1)
temp_dataX2=np.reshape(temp_dataX2,-1)
temp_dataY=np.reshape(temp_dataY,-1)



dataX =[]
dataX2 =[]
dataY= []
# here is for the case that input graph and output graph
for i in range(temp_dataX.shape[0]-n_frames*n_hours*n_cols+n_hours*n_cols):
    if i%(n_hours*n_cols) == 0:
        dataX.append(temp_dataX[i:i+n_frames*n_hours*n_cols])
        dataX2.append(temp_dataX2[i:i + n_frames * n_hours * n_cols])
        dataY.append(temp_dataY[i])






dataX=np.reshape(dataX,(np.array(dataX).shape[0],-1))
dataX2=np.reshape(dataX2,(np.array(dataX2).shape[0],-1))
dataY=np.reshape(dataY,(np.array(dataY).shape[0],-1))



#build model
left_branch = Sequential()
left_branch.add(Dense(32, input_dim=n_frames*n_hours*n_cols))

right_branch = Sequential()
right_branch.add(Dense(32, input_dim=n_frames*n_hours*n_cols))

merged = Merge([left_branch, right_branch], mode='concat')

final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(1, activation='softmax'))

final_model.compile(optimizer='rmsprop', loss='mean_squared_error')

print('Train...')
final_model.fit([dataX, dataX2], dataY, nb_epoch=nb_epoch,verbose=0)  # we pass one data array per model input

plot(final_model,to_file='/home/slave1/PycharmProjects/conv_lstm/branch model for merge/model_merge.png',show_shapes=True)


json_string = final_model.to_json()
open('/home/slave1/PycharmProjects/conv_lstm/branch model for merge/model_merge_architecture.json','w').write(json_string)
final_model.save_weights('/home/slave1/PycharmProjects/conv_lstm/branch model for merge/model_merge_weights.h5')


