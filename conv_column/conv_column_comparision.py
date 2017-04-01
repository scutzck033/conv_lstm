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
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt
from keras import optimizers
from keras import regularizers
from keras.constraints import maxnorm

#When using the GridSearchCV
# solve the problem: TypeError: get_params() got an unexpected keyword argument 'deep'
from keras.wrappers.scikit_learn import BaseWrapper
import copy

def custom_get_params(self, **params):
    res = copy.deepcopy(self.sk_params)
    res.update({'build_fn': self.build_fn})
    return res

BaseWrapper.get_params = custom_get_params


# Convolution
nb_row=3
nb_col=3
nb_filter = 64
pool_size = 2

# LSTM
lstm_output_size = 64

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
rawdata_train = pd.read_csv("../data/ShangZheng1H_NoNomrlized.csv",encoding='gbk')
rawdata_train=rawdata_train[[5]] # close


data=rawdata_train.as_matrix()[0:244]

temp_dataX= []
temp_dataY= []


temp_dataX=data[0:(data.shape[0]-n_hours)]
temp_dataY=data[n_frames*n_hours:data.shape[0]]


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
# def create_model(dropout_rate=0.0, weight_constraint=0):
model = Sequential()
model.add(Convolution2D(nb_filter, nb_row, nb_col, border_mode='same',input_shape=(np.array(dataX).shape[1],np.array(dataX).shape[2],np.array(dataX).shape[3]),W_constraint=maxnorm(2),init='normal'))
model.add(Dropout(0.9))
model.add(MaxPooling2D(pool_size=(pool_size, pool_size), border_mode='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(np.array(dataY).shape[1]))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    # return model


# plot(model,to_file='model_4241_conv.png',show_shapes=True)

# print('Train...')
model.fit(dataX, dataY, batch_size=batch_size, nb_epoch=nb_epoch,verbose=2)
#
#
#
json_string = model.to_json()
open('model_4241_conv_architecture.json','w').write(json_string)
model.save_weights('model_4241_conv_weights.h5')

# fix random seed for reproducibility
# seed = 7
# np.random.seed(seed)

# get the model
# model = KerasRegressor(build_fn=create_model,verbose=2,batch_size=batch_size,nb_epoch=nb_epoch)


# define the grid search parameters
# weight_constraint = [1, 2, 3, 4, 5]
# dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
# grid_result = grid.fit(dataX, dataY)
#
#
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# params = grid_result.cv_results_['params']
# for mean,param in zip(means,params):
#     print("%f with: %r" % (mean,param))

prediction = model.predict(dataX, verbose=0)
print('Predict...')

temp = 0.0
# print(prediction)

print('label...')
# print(dataY)

for i in range(prediction.shape[0]):
     for j in range(prediction.shape[1]):
         temp = temp + abs(prediction[i][j] - dataY[i][j]) / dataY[i][j]
error = temp / (prediction.shape[0] * prediction.shape[1])
print("Model_conv_column error: %.2f%%" % (error * 100))

x = np.linspace(0, 1, 100)
x = [n for n in range(0, prediction.shape[0])]
plt.plot(x, prediction, label="$ConvColumnError:$" + '%.2f' % (error * 100) + '%', color="red")
plt.plot(x, dataY, color="blue", label="$label$")
plt.legend()

plt.xlabel("Time(day)")
plt.ylabel("Value")
plt.title("ShangZhengIndex_NoNomorlized")
# plt.show()

plt.savefig("ShangZhengIndex_NoNomorlized_Trained.png")