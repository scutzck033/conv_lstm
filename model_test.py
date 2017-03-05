#model test

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import pandas as pd
import h5py
from keras.models import model_from_json

def model_mlp_4241_test(rawdata):
    rawdata_test = rawdata[[2]]
    dataTest = rawdata_test.as_matrix()[160:244]

    temp_dataX_Test = []
    temp_dataY_Test = []

    n_frames = 4
    n_hours = 4
    n_cols = 1

    temp_dataX_Test = dataTest[0:(dataTest.shape[0] - n_hours)]
    temp_dataY_Test = dataTest[n_frames * n_hours:dataTest.shape[0]]

    temp_dataX_Test = np.reshape(temp_dataX_Test, -1)
    temp_dataY_Test = np.reshape(temp_dataY_Test, -1)

    dataX_Test = []
    dataY_Test = []

    for i in range(temp_dataX_Test.shape[0] - n_frames * n_hours * n_cols + n_hours * n_cols):
        if i % (n_hours * n_cols) == 0:
            dataX_Test.append(temp_dataX_Test[i:i + n_frames * n_hours * n_cols])
            dataY_Test.append(temp_dataY_Test[i])

    dataX_Test = np.reshape(dataX_Test, (np.array(dataX_Test).shape[0], -1))
    dataY_Test = np.reshape(dataY_Test, (np.array(dataY_Test).shape[0], -1))

    model = model_from_json(open('model_mlp_4241_architecture.json').read())
    model.load_weights('model_mlp_4241_weights.h5')

    print('Score...')
    prediction = model.predict(dataX_Test, verbose=0)
    print('Predict...')

    temp = 0.0


    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            temp = temp + abs(prediction[i][j] - dataY_Test[i][j]) / dataY_Test[i][j]
    error = temp / (prediction.shape[0] * prediction.shape[1])
    print("Model_mlp_4241 error: %.2f%%" % (error * 100))


def model_merge_test(rawdata):
    rawdata_test = rawdata[[2]]
    rawdata_test2=rawdata[[3]]
    dataTest = rawdata_test.as_matrix()[0:888]
    dataTest2 = rawdata_test2.as_matrix()[0:888]
    temp_dataX_Test = []
    temp_dataY_Test = []

    n_frames = 4
    n_hours = 24
    n_cols = 1

    temp_dataX_Test = dataTest[0:(dataTest.shape[0] - n_hours)]
    temp_dataX_Test2 = dataTest2[0:(dataTest.shape[0] - n_hours)]
    temp_dataY_Test = dataTest[n_frames * n_hours:dataTest.shape[0]]

    temp_dataX_Test = np.reshape(temp_dataX_Test, -1)
    temp_dataX_Test2 = np.reshape(temp_dataX_Test2, -1)
    temp_dataY_Test = np.reshape(temp_dataY_Test, -1)

    dataX_Test = []
    dataX_Test2 = []
    dataY_Test = []

    for i in range(temp_dataX_Test.shape[0] - n_frames * n_hours * n_cols + n_hours * n_cols):
        if i % (n_hours * n_cols) == 0:
            dataX_Test.append(temp_dataX_Test[i:i + n_frames * n_hours * n_cols])
            dataX_Test2.append(temp_dataX_Test2[i:i + n_frames * n_hours * n_cols])
            dataY_Test.append(temp_dataY_Test[i])

    dataX_Test = np.reshape(dataX_Test, (np.array(dataX_Test).shape[0], -1))
    dataX_Test2 = np.reshape(dataX_Test2, (np.array(dataX_Test2).shape[0], -1))
    dataY_Test = np.reshape(dataY_Test, (np.array(dataY_Test).shape[0], -1))

    model = model_from_json(open('/home/slave1/PycharmProjects/conv_lstm/branch model for merge/model_merge_architecture.json').read())
    model.load_weights('/home/slave1/PycharmProjects/conv_lstm/branch model for merge/model_merge_weights.h5')

    print('Score...')
    prediction = model.predict([dataX_Test,dataX_Test2], verbose=0)
    print('Predict...')

    temp = 0.0

    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            temp = temp + abs(prediction[i][j] - dataY_Test[i][j]) / dataY_Test[i][j]
    error = temp / (prediction.shape[0] * prediction.shape[1])
    print("Model_merge error: %.2f%%" % (error * 100))


def model_4241_test(rawdata):
    rawdata_test = rawdata[[2]]#get one column
    dataTest = rawdata_test.as_matrix()[0:888]
    temp_dataX_Test = []
    temp_dataY_Test = []

    n_frames = 4
    n_hours = 24
    n_cols = 1

    temp_dataX_Test=dataTest[0:(dataTest.shape[0]-n_hours)]
    temp_dataY_Test=dataTest[n_hours:dataTest.shape[0]]



    temp_dataX_Test=np.reshape(temp_dataX_Test,-1)
    temp_dataY_Test=np.reshape(temp_dataY_Test,-1)

    dataX_Test =[]
    dataY_Test= []

    for i in range(temp_dataX_Test.shape[0]-n_frames*n_hours*n_cols+n_hours*n_cols):
        if i%(n_hours*n_cols) == 0:
            dataX_Test.append(temp_dataX_Test[i:i+n_frames*n_hours*n_cols])
    for i in range(temp_dataY_Test.shape[0] - n_frames * n_hours * 1 + n_hours * 1):
        if i % (n_hours * 1) == 0:
            temp = []
            for j in range(n_frames):
                temp.append(temp_dataY_Test[i + j * n_hours])
            dataY_Test.append(temp)

    dataX_Test=np.reshape(dataX_Test,(np.array(dataX_Test).shape[0],n_frames,n_hours,n_cols,1))
    dataY_Test=np.reshape(dataY_Test,(np.array(dataY_Test).shape[0],n_frames,-1))

    model = model_from_json(open('model_4241_architecture.json').read())
    model.load_weights('model_4241_weights.h5')

    print ('Score...')
    prediction=model.predict(dataX_Test,verbose=0)
    print ('Predict...')

    temp = 0.0

    #conv_lstm
    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            for k in range(prediction.shape[2]):
                temp=temp+abs(prediction[i][j][k]-dataY_Test[i][j][k])/dataY_Test[i][j][k]
    error=temp/(prediction.shape[0]*prediction.shape[1]*prediction.shape[2])
    print("Model_4241 error: %.2f%%" % (error*100))


def model_4244_test(rawdata):
    rawdata_test = rawdata[[2,3,4,5]]
    dataTest = rawdata_test.as_matrix()[0:888]
    temp_dataX_Test = []
    temp_dataY_Test = []

    n_frames = 4
    n_hours = 24
    n_cols = 4

    temp_dataX_Test = dataTest[0:(dataTest.shape[0] - n_hours)]
    temp_dataY_Test = dataTest[n_hours:dataTest.shape[0]]
    temp_dataY_Test = np.transpose(np.transpose(temp_dataY_Test)[0:1])
    temp_dataX_Test = np.reshape(temp_dataX_Test, -1)
    temp_dataY_Test = np.reshape(temp_dataY_Test, -1)

    dataX_Test = []
    dataY_Test = []

    for i in range(temp_dataX_Test.shape[0]-n_frames*n_hours*n_cols+n_hours*n_cols):
        if i%(n_hours*n_cols) == 0:
            dataX_Test.append(temp_dataX_Test[i:i+n_frames*n_hours*n_cols])
    for i in range(temp_dataY_Test.shape[0]-n_frames*n_hours*1+n_hours*1):
        if i%(n_hours*1) == 0:
            temp = []
            for j in range(n_frames):
                temp.append(temp_dataY_Test[i + j * n_hours])
            dataY_Test.append(temp)

    dataX_Test = np.reshape(dataX_Test, (np.array(dataX_Test).shape[0], n_frames, n_hours, n_cols, 1))
    dataY_Test = np.reshape(dataY_Test, (np.array(dataY_Test).shape[0], n_frames, -1))

    model = model_from_json(open('model_4244_architecture.json').read())
    model.load_weights('model_4244_weights.h5')

    print('Score...')
    prediction = model.predict(dataX_Test, verbose=0)
    print('Predict...')

    temp = 0.0

    # conv_lstm
    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            for k in range(prediction.shape[2]):
                temp = temp + abs(prediction[i][j][k] - dataY_Test[i][j][k]) / dataY_Test[i][j][k]
    error = temp / (prediction.shape[0] * prediction.shape[1] * prediction.shape[2])
    print("Model_4244 error: %.2f%%" % (error * 100))


#load testing raw data
rawdata_test = pd.read_csv("4h per day(1h).csv",encoding='gbk')
# model_4244_test(rawdata_test)
# model_4241_test(rawdata_test)
model_mlp_4241_test(rawdata_test)
# model_merge_test(rawdata_test)










# rawdata_test=rawdata_test[[2]]
# rawdata_test=rawdata_test[[2,3,4,5]] # open,close,highest,lowest

# dataTest=rawdata_test.as_matrix()[0:888]
#
#
# temp_dataX_Test= []
# temp_dataY_Test= []
#
# n_frames=4
# n_hours=24
# n_cols=1
# # n_cols=4
#
# #conv_lstm
# # temp_dataX_Test=dataTest[0:(dataTest.shape[0]-n_hours)]
# # temp_dataY_Test=dataTest[n_hours:dataTest.shape[0]]
#
# #mlp
# temp_dataX_Test=dataTest[0:(dataTest.shape[0]-n_hours)]
# temp_dataY_Test=dataTest[n_frames*n_hours:dataTest.shape[0]]
#
# #4241
# # temp_dataY_Test=np.transpose(np.transpose(temp_dataY_Test)[0:1])
#
# temp_dataX_Test=np.reshape(temp_dataX_Test,-1)
# temp_dataY_Test=np.reshape(temp_dataY_Test,-1)
#
# dataX_Test =[]
# dataY_Test= []
#
# #conv_lstm
# # here is for the case that input graph and output graph --- 4241
# # for i in range(temp_dataX_Test.shape[0]-n_frames*n_hours*n_cols+n_hours*n_cols):
# #     if i%(n_hours*n_cols) == 0:
# #         dataX_Test.append(temp_dataX_Test[i:i+n_frames*n_hours*n_cols])
# #         dataY_Test.append(temp_dataY_Test[i:i + n_frames * n_hours * n_cols])
#
# # here is for the case that input graph and output one col --- 4244
# # for i in range(temp_dataX_Test.shape[0]-n_frames*n_hours*n_cols+n_hours*n_cols):
# #     if i%(n_hours*n_cols) == 0:
# #         dataX_Test.append(temp_dataX_Test[i:i+n_frames*n_hours*n_cols])
# # for i in range(temp_dataY_Test.shape[0]-n_frames*n_hours*1+n_hours*1):
# #     if i%(n_hours*1) == 0:
# #         dataY_Test.append(temp_dataY_Test[i:i+n_frames*n_hours*1])
#
# #mlp
# for i in range(temp_dataX_Test.shape[0]-n_frames*n_hours*n_cols+n_hours*n_cols):
#     if i%(n_hours*n_cols) == 0:
#         dataX_Test.append(temp_dataX_Test[i:i+n_frames*n_hours*n_cols])
#         dataY_Test.append(temp_dataY_Test[i:i +n_hours * n_cols])
#
# #conv_lstm
# # dataX_Test=np.reshape(dataX_Test,(np.array(dataX_Test).shape[0],n_frames,n_hours,n_cols,1))
# # dataY_Test=np.reshape(dataY_Test,(np.array(dataY_Test).shape[0],n_frames,-1))
#
# #mlp
# dataX_Test=np.reshape(dataX_Test,(np.array(dataX_Test).shape[0],-1))
# dataY_Test=np.reshape(dataY_Test,(np.array(dataY_Test).shape[0],-1))
#
# # model = model_from_json(open('model_4241_architecture.json').read())
# # model.load_weights('model_4241_weights.h5')
# # model = model_from_json(open('model_4244_architecture.json').read())
# # model.load_weights('model_4244_weights.h5')
# model = model_from_json(open('model_mlp_4241_architecture.json').read())
# model.load_weights('model_mlp_4241_weights.h5')
#
#
# print ('Score...')
# prediction=model.predict(dataX_Test,verbose=0)
# print ('Predict...')
#
# temp = 0.0
#
# #conv_lstm
# # for i in range(prediction.shape[0]):
# #     for j in range(prediction.shape[1]):
# #         for k in range(prediction.shape[2]):
# #             temp=temp+abs(prediction[i][j][k]-dataY_Test[i][j][k])/dataY_Test[i][j][k]
# #             print (temp)
# # error=temp/(prediction.shape[0]*prediction.shape[1]*prediction.shape[2])
# # print("Model_4241 error: %.2f%%" % (error*100))
# # print("Model_4244 error: %.2f%%" % (error*100))
#
# #mlp
# for i in range(prediction.shape[0]):
#     for j in range(prediction.shape[1]):
#         temp=temp+abs(prediction[i][j]-dataY_Test[i][j])/dataY_Test[i][j]
# print (prediction.shape)
# error=temp/(prediction.shape[0]*prediction.shape[1])
# print("Model_mlp_4241 error: %.2f%%" % (error*100))
