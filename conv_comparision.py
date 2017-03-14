#4*24*4 predict 24*1

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.layers import Convolution2D, MaxPooling2D,Flatten,TimeDistributed
from keras.utils.visualize_util import plot
import pandas as pd
import h5py
from keras.models import model_from_json


# Convolution
nb_row=3
nb_col=3
nb_filter = 64
pool_size = 2

# LSTM
lstm_output_size = 64

# Training
batch_size = 32
nb_epoch = 100

n_frames=4
n_hours=4
n_cols=4

#Normalization
def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min);
    return x;


#load training raw data
rawdata_train = pd.read_csv("4h per day(1h).csv",encoding='gbk')
rawdata_train=rawdata_train[[2,3,4,5]] # open,max,min,close


data=rawdata_train.as_matrix()[0:160]

temp_dataX= []
temp_dataY= []


temp_dataX=data[0:(data.shape[0]-n_hours)]
temp_dataY=data[n_frames*n_hours:data.shape[0]]

#get one column
temp_dataY=np.transpose(np.transpose(temp_dataY)[0:1])

temp_dataX=np.reshape(temp_dataX,-1)
temp_dataY=np.reshape(temp_dataY,-1)



dataX =[]
dataY= []


# here is for the case that input graph and output one col
for i in range(temp_dataX.shape[0]-n_frames*n_hours*n_cols+n_hours*n_cols):
    if i%(n_hours*n_cols) == 0:
        dataX.append(temp_dataX[i:i+n_frames*n_hours*n_cols])
for i in range(temp_dataY.shape[0]):
    if i%(n_hours*1)==0:
        dataY.append(temp_dataY[i])

#Normalization
# maxV=np.max(dataY)
# minV=np.min(dataY)
# for i in range(np.array(dataY).shape[0]):
#     for j in range(np.array(dataY).shape[1]):
#         dataY[i][j]=MaxMinNormalization(dataY[i][j],maxV,minV)



dataX=np.reshape(dataX,(np.array(dataX).shape[0],n_frames*n_hours,n_cols,1))
dataY=np.reshape(dataY,(np.array(dataY).shape[0],-1))

print (dataX.shape)
print (dataY.shape)


#build the model
model = Sequential()
model.add(Convolution2D(nb_filter, nb_row, nb_col, border_mode='same',input_shape=(np.array(dataX).shape[1],np.array(dataX).shape[2],np.array(dataX).shape[3])))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), border_mode='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(np.array(dataY).shape[1]))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])


plot(model,to_file='model_4244_conv.png',show_shapes=True)

print('Train...')
model.fit(dataX, dataY, batch_size=batch_size, nb_epoch=nb_epoch,verbose=0)



json_string = model.to_json()
open('model_4244_conv_architecture.json','w').write(json_string)
model.save_weights('model_4244_conv_weights.h5')