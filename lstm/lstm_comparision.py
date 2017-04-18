#coding:UTF-8
# LSTM for international airline passengers problem with time step regression framing
import numpy
# import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../utils')
from DataUtil import DataUtil

# # convert an array of values into a dataset matrix
# def create_dataset(dataset, look_back=1):
# 	dataX, dataY = [], []
# 	for i in range(len(dataset)-look_back-1):
# 		a = dataset[i:(i+look_back), 0]
# 		dataX.append(a)
# 		dataY.append(dataset[i + look_back, 0])
# 	return numpy.array(dataX), numpy.array(dataY)
# # fix random seed for reproducibility
# numpy.random.seed(7)
# # load the dataset
# dataframe = pandas.read_csv('../data/上证指数/3年数据_Volume列归一化.csv', usecols=[2], engine='python', skipfooter=3)
# dataset = dataframe.values
# dataset = dataset.astype('float32')
# # normalize the dataset
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
# # split into train and test sets
# train_size = int(len(dataset) * 0.67)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# # reshape into X=t and Y=t+1
# look_back = 4
# trainX, dataY = create_dataset(train, look_back)
# testX, testY = create_dataset(test, look_back)
# # reshape input to be [samples, time steps, features]
# trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# # create and fit the LSTM network
# model = Sequential()
# model.add(LSTM(4, input_dim=1))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(trainX, dataY, nb_epoch=20, batch_size=1, verbose=2)
# # make predictions
# prediction = model.predict(trainX)
# testPredict = model.predict(testX)
# invert predictions
# trainPredict = scaler.inverse_transform(trainPredict)
# trainY = scaler.inverse_transform([trainY])
# testPredict = scaler.inverse_transform(testPredict)
# testY = scaler.inverse_transform([testY])
# # calculate root mean squared error
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
# # shift train predictions for plotting
# trainPredictPlot = numpy.empty_like(dataset)
# trainPredictPlot[:, :] = numpy.nan
# trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:, :] = numpy.nan
# testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# # plot baseline and predictions
# # plt.plot(scaler.inverse_transform(dataset))
# # plt.plot(trainPredictPlot)
# # plt.plot(testPredictPlot)
# # plt.show()
# print ('Predict...')
# print (testPredictPlot)
# print ('Label...')
# print (testY)

#coding:utf-8
#4*24*4 predict 24*1


import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.layers import Convolution2D, MaxPooling2D,Flatten,TimeDistributed
from keras.layers import Merge
from keras.utils.visualize_util import plot
import pandas as pd
import h5py
from keras.models import model_from_json
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt
from keras import optimizers
from keras import regularizers
from keras.constraints import maxnorm
import sys
sys.path.append('../utils')
from DataUtil import DataUtil

#When using the GridSearchCV
# solve the problem: TypeError: get_params() got an unexpected keyword argument 'deep'
from keras.wrappers.scikit_learn import BaseWrapper
import copy

def custom_get_params(self, **params):
    res = copy.deepcopy(self.sk_params)
    res.update({'build_fn': self.build_fn})
    return res

BaseWrapper.get_params = custom_get_params

n_frames=4
n_hours=4
n_cols=1

# LSTM
lstm_output_size = 4

# Training
batch_size = 32
nb_epoch = 1000


#load training raw data
rawdata_train = pd.read_csv("../data/上证指数/3年数据_Volume列无归一化.csv",encoding='gbk')

data,len=DataUtil.getData(rawdata_train.as_matrix(),startpoint=' 2015/03/19-10:30',endpoint=' 2016/12/30-15:00',n_hours=n_hours)
print (len)
data = data[:,4] #close


temp_dataX= []
temp_dataY= []


temp_dataX=data[0:(data.shape[0]-n_hours)]
temp_dataY=data[n_frames*n_hours:data.shape[0]]

temp_dataX=np.reshape(temp_dataX,-1)
temp_dataY=np.reshape(temp_dataY,-1)

# print (volume.shape)

dataX =[]
dataY= []

# here is for the case that input graph and output one col
for i in range(temp_dataX.shape[0]-n_frames*n_hours*n_cols+n_hours*n_cols):
    if i%(n_hours*n_cols) == 0:
        dataX.append(temp_dataX[i:i+n_frames*n_hours*n_cols])
for i in range(temp_dataY.shape[0]):
    if i%(n_hours*1)==0:
        dataY.append(temp_dataY[i + n_hours -1])

# print (np.array(dataX).shape)
# print (np.array(dataY).shape)


# reshape input to be [samples, time steps, features]
dataX = numpy.reshape(dataX, (np.array(dataX).shape[0], np.array(dataX).shape[1], 1))
dataY=np.reshape(dataY,(np.array(dataY).shape[0],-1))

# print (np.array(dataX).shape)
# create and fit the LSTM network
model = Sequential()
#a hidden layer with 'lstm_output_size' LSTM blocks or neurons
# input_shape =(input_length,input_dim)
model.add(LSTM(lstm_output_size, input_shape=(n_hours*n_frames, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(dataX, dataY, nb_epoch=nb_epoch, batch_size=1, verbose=2)


# plot
plot(model,to_file='model_lstm_oneFeature.png',show_shapes=True)



#
json_string = model.to_json()
open('model_lstm_oneFeature_architecture.json','w').write(json_string)
model.save_weights('model_lstm_oneFeature_weights.h5')
# # # # #
prediction = model.predict(dataX, verbose=0)
# print('Predict...')
#
temp = 0.0
#     # print(prediction)
#
# print('label...')
#     # print(dataY)
#
for i in range(prediction.shape[0]):
     for j in range(prediction.shape[1]):
         temp = temp + abs(prediction[i][j] - dataY[i][j]) / dataY[i][j]
error = temp / (prediction.shape[0] * prediction.shape[1])
print("Model_conv_graph error: %.2f%%" % (error * 100))

match_percent = DataUtil.tendencyMatch(dataY,prediction)
print ("match_percent is: %.2f%%" %(match_percent * 100))
# # #
# # #
# # #
x = np.linspace(0, 1, 25)
x = [n for n in range(0, prediction.shape[0])]
plt.plot(x, prediction, label="$LstmError:$"+'%.2f' %(error*100)+'%', color="red")
plt.plot(x, dataY, color="blue", label="$label$")
plt.legend()

plt.xlabel("Time(day)")
plt.ylabel("Value")
plt.title("lstm")
# plt.show()
# # # #
plt.savefig("model_lstm_oneFeature.png")
# plt.close("all")

# # summarize history for loss
# plt.plot(history.history['loss'][200:nb_epoch])
# plt.plot(history.history['val_loss'][200:nb_epoch])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'validation'], loc='upper left')
# plt.show()
# plt.savefig("ShangZhengIndex_DependentNomorlized_TrainAndValidation_loss")


