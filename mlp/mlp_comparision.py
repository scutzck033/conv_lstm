#4*24*1 predict 24*1
#using benchmark MLP

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


# Training
batch_size = 32
nb_epoch = 10000

n_frames=4
n_hours=4
n_cols=1

#Normalization
def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min);
    return x;

#load training raw data
rawdata_train = pd.read_csv("../data/4h per day(1h)_normalized.csv",encoding='gbk')
rawdata_train=rawdata_train[[2]] # open,close,highest,lowest


data=rawdata_train.as_matrix()[0:160]

temp_dataX= []
temp_dataY= []



temp_dataX=data[0:(data.shape[0]-n_hours)]
temp_dataY=data[n_frames*n_hours:data.shape[0]]
temp_dataX=np.reshape(temp_dataX,-1)
temp_dataY=np.reshape(temp_dataY,-1)



dataX =[]
dataY= []
# here is for the case that input graph and output graph
for i in range(temp_dataX.shape[0]-n_frames*n_hours*n_cols+n_hours*n_cols):
    if i%(n_hours*n_cols) == 0:
        dataX.append(temp_dataX[i:i+n_frames*n_hours*n_cols])
        dataY.append(temp_dataY[i])






dataX=np.reshape(dataX,(np.array(dataX).shape[0],-1))
dataY=np.reshape(dataY,(np.array(dataY).shape[0],-1))

#Normalization
# maxV=np.max(dataY)
# minV=np.min(dataY)
# for i in range(np.array(dataY).shape[0]):
#     for j in range(np.array(dataY).shape[1]):
#         dataY[i][j]=MaxMinNormalization(dataY[i][j],maxV,minV)

print (dataY)
#build the model
model = Sequential()
model.add(Dense(output_dim=64, input_dim=n_frames*n_hours*n_cols))
model.add(Activation("relu"))
model.add(Dense(output_dim=1))
# model.add(Activation("sigmoid"))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

plot(model,to_file='model_mlp_4241.png',show_shapes=True)

print('Train...')
model.fit(dataX, dataY, batch_size=batch_size, nb_epoch=nb_epoch,verbose=0)



json_string = model.to_json()
open('model_mlp_4241_architecture.json','w').write(json_string)
model.save_weights('model_mlp_4241_weights.h5')