#coding:utf-8
#model test
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import pandas as pd
import h5py
from keras.models import model_from_json
import matplotlib.pyplot as plt
import dateutil, pylab, random
from pylab import *
from datetime import datetime, timedelta
import time
import datetime
import sys
sys.path.append('../utils')
from DataUtil import DataUtil

#Normalization
def MaxMinNormalization(x,Max,Min):
    x = (x - Min) / (Max - Min);
    return x;

def model_mlp_4241_test(rawdata):
    rawdata_test = rawdata[[2]]
    dataTest = rawdata_test.as_matrix()[160:244]#[0:888]

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

    # Normalization
    # maxV = np.max(dataY_Test)
    # minV = np.min(dataY_Test)
    # for i in range(np.array(dataY_Test).shape[0]):
    #     for j in range(np.array(dataY_Test).shape[1]):
    #         dataY_Test[i][j] = MaxMinNormalization(dataY_Test[i][j], maxV, minV)

    model = model_from_json(open('/home/darren/PycharmProjects/conv_lstm/mlp/model_mlp_4241_architecture.json').read())
    model.load_weights('/home/darren/PycharmProjects/conv_lstm/mlp/model_mlp_4241_weights.h5')

    print('Score...')
    prediction = model.predict(dataX_Test, verbose=0)
    print('Predict...')

    temp = 0.0
    print (prediction)

    print ('label...')
    print (dataY_Test)

    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            temp = temp + abs(prediction[i][j] - dataY_Test[i][j]) / dataY_Test[i][j]
    error = temp / (prediction.shape[0] * prediction.shape[1])
    print("Model_mlp_4241 error: %.2f%%" % (error * 100))

    x = np.linspace(0, 1, 100)
    x = [n for n in range(0, prediction.shape[0])]
    plt.plot(x, prediction, label="$prediction$", color="red")
    plt.plot(x, dataY_Test, color="blue", label="$label$")
    plt.legend()
    plt.xlabel("Time(day)")
    plt.ylabel("Value")
    plt.title("mlp_441")
    # plt.show()

    plt.savefig("mlp_441.png")

    plt.close('all')

def model_conv_4244_test(rawdata):
    # n_frames = 4
    # n_hours = 4
    # n_cols = 5
    n_frames = 4
    n_hours = 4
    n_cols = 4


    dataTest,len1 = DataUtil.getData(rawdata,startpoint=' 2017/01/03-10:30',endpoint=' 2017/03/24-15:00',n_hours=n_hours)
    #
    # # start point -- use moving_len to get the predicted starting date
    dateStr,len2 = DataUtil.getData(rawdata,startpoint=' 2017/01/03-10:30',endpoint=' 2017/03/24-15:00',n_hours=n_hours,moving_len=n_frames)
    dateStr = dateStr[:,0]
    volume = dataTest[:, 5]
    dataTest = dataTest[:, [1, 2, 3, 4]]
    # dataTest=dataTest[:,[1,2,3,4,5]]

    # dataTest = pd.read_csv("../data/pems_jun_2014_train.csv", encoding='gbk').as_matrix()[212:252]




    temp_dataX_Test = dataTest[0:(dataTest.shape[0] - n_hours)]
    temp_dataY_Test = dataTest[n_frames * n_hours:dataTest.shape[0]]
    volume = volume[0:(dataTest.shape[0] - n_hours)]


    # get one column
    temp_dataY_Test = temp_dataY_Test[:, 3]
    # dataY_Test = temp_dataY_Test[:, 103]


    temp_dataX_Test = np.reshape(temp_dataX_Test, -1)
    temp_dataY_Test = np.reshape(temp_dataY_Test, -1)
    volume = np.reshape(volume, -1)

    print (temp_dataX_Test.shape)
    print (temp_dataY_Test.shape)

    print (temp_dataY_Test.shape)
    dataX_Test = []
    dataY_Test = []
    dateTimeList = []
    train_volume = []

    for i in range(temp_dataX_Test.shape[0] - n_frames * n_hours * n_cols + n_hours * n_cols):
        if i % (n_hours * n_cols) == 0:
            dataX_Test.append(temp_dataX_Test[i:i + n_frames * n_hours * n_cols])
    for i in range(temp_dataY_Test.shape[0]):
        if i % (n_hours * 1) == 0:
            dataY_Test.append(temp_dataY_Test[i])
            dateTimeList.append(dateStr[i])
    for i in range(volume.shape[0] - n_frames * n_hours * 1 + n_hours * 1):
        if i % (n_hours * 1) == 0:
            train_volume.append(volume[i:i + n_frames * n_hours * 1])



    dataX_Test = np.reshape(dataX_Test, (np.array(dataX_Test).shape[0], n_frames * n_hours, n_cols, 1))
    dataY_Test = np.reshape(dataY_Test, (np.array(dataY_Test).shape[0], -1))
    train_volume = np.reshape(train_volume, (np.array(train_volume).shape[0], -1))

    # Normalization
    # maxV = np.max(dataY_Test)
    # minV = np.min(dataY_Test)
    # for i in range(np.array(dataY_Test).shape[0]):
    #     for j in range(np.array(dataY_Test).shape[1]):
    #         dataY_Test[i][j] = MaxMinNormalization(dataY_Test[i][j], maxV, minV)

    model = model_from_json(open('../conv_graph/model_4244_conv_architecture.json').read())
    model.load_weights('../conv_graph/model_4244_conv_weights.h5')



    print('Score...')
    prediction = model.predict([dataX_Test,train_volume], verbose=0)
    print('Predict...')


    temp = 0.0
    print(prediction)


    print('label...')
    print(dataY_Test.shape)

    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            temp = temp + abs(prediction[i][j] - dataY_Test[i][j]) / dataY_Test[i][j]
    error = temp / (prediction.shape[0] * prediction.shape[1])
    print("Model_conv_graph error: %.2f%%" % (error * 100))

    match_percent = DataUtil.tendencyMatch(dataY_Test, prediction)
    print("match_percent is: %.2f%%" % (match_percent * 100))

    dates = []
    # dates = np.linspace(0, 1, 100)
    for i in range(np.array(dateTimeList).shape[0]):
        temp = time.strptime(dateTimeList[i], " %Y/%m/%d-%H:%M")  # 字符串转换成time类型
        temp = datetime.datetime(temp[0], temp[1], temp[2],temp[3],temp[4])  # time类型转换成datetime类型
        dates.append(temp)



    pylab.plot_date(dates, prediction, linestyle='-',label ="$ConvGraphError:$"+'%.2f' %(error*100)+'%',color="green")
    pylab.plot_date(dates, dataY_Test, linestyle='-', label="$label$",color="blue")
    # plt.plot(dates, prediction, label ="$ConvGraphError:$"+'%.2f' %(error*100)+'%',color="green")
    # plt.plot(dates, dataY_Test, color="blue", label="$label$")
    plt.legend()

    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("match_percent is: %.2f%%" % (match_percent * 100))
    # plt.show()

    plt.savefig("./conv_model/Volume列作为merge特征/归一化.png")

    # plt.close('all')

    # x = np.linspace(0, 1, 50)
    # x = [n for n in range(0, prediction.shape[0])]
    # plt.plot(x, prediction, label="$ConvGraphError:$" + '%.2f' % (error * 100) + '%', color="green")
    # plt.plot(x, dataY_Test, color="blue", label="$label$")
    # plt.legend()
    #
    # plt.xlabel("Time(day)")
    # plt.ylabel("Value")
    # plt.title("TrafficFlow")
    # plt.show()
    # plt.close('all')


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

    model = model_from_json(open('/home/darren/PycharmProjects/conv_lstm/branch model for merge/model_merge_architecture.json').read())
    model.load_weights('/home/darren/PycharmProjects/conv_lstm/branch model for merge/model_merge_weights.h5')

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
    dataTest = rawdata_test.as_matrix()[160:244]#[0:888]
    temp_dataX_Test = []
    temp_dataY_Test = []

    n_frames = 4
    n_hours = 4
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
            dataY_Test.append(temp_dataY_Test[i:i + n_frames * n_hours * n_cols])
    # for i in range(temp_dataY_Test.shape[0] - n_frames * n_hours * 1 + n_hours * 1):
    #     if i % (n_hours * 1) == 0:
    #         temp = []
    #         for j in range(n_frames):
    #             temp.append(temp_dataY_Test[i + j * n_hours])
    #         dataY_Test.append(temp)

    # Normalization
    # maxV = np.max(dataY_Test)
    # minV = np.min(dataY_Test)
    # for i in range(np.array(dataY_Test).shape[0]):
    #     for j in range(np.array(dataY_Test).shape[1]):
    #         dataY_Test[i][j] = MaxMinNormalization(dataY_Test[i][j], maxV, minV)
    dataX_Test=np.reshape(dataX_Test,(np.array(dataX_Test).shape[0],n_frames,n_hours,n_cols,1))
    dataY_Test=np.reshape(dataY_Test,(np.array(dataY_Test).shape[0],n_frames,-1))

    model = model_from_json(open('model_4241_architecture.json').read())
    model.load_weights('model_4241_weights.h5')

    print ('Score...')
    prediction=model.predict(dataX_Test,verbose=0)
    print ('Predict...')
    print (prediction)
    print ('label...')
    print (dataY_Test)

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
    dataTest = rawdata_test.as_matrix()[160:208]#[0:888]
    temp_dataX_Test = []
    temp_dataY_Test = []

    n_frames = 4
    n_hours = 4
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

    # Normalization
    # maxV = np.max(dataY_Test)
    # minV = np.min(dataY_Test)
    # for i in range(np.array(dataY_Test).shape[0]):
    #     for j in range(np.array(dataY_Test).shape[1]):
    #         dataY_Test[i][j] = MaxMinNormalization(dataY_Test[i][j], maxV, minV)

    dataX_Test = np.reshape(dataX_Test, (np.array(dataX_Test).shape[0], n_frames, n_hours, n_cols, 1))
    dataY_Test = np.reshape(dataY_Test, (np.array(dataY_Test).shape[0], n_frames, -1))

    model = model_from_json(open('model_4244_architecture.json').read())
    model.load_weights('model_4244_weights.h5')

    print('Score...')
    prediction = model.predict(dataX_Test, verbose=0)
    print('Predict...')

    temp = 0.0
    print (prediction)
    print ("label...")
    print (dataY_Test)

    # conv_lstm
    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            for k in range(prediction.shape[2]):
                temp = temp + abs(prediction[i][j][k] - dataY_Test[i][j][k]) / dataY_Test[i][j][k]
    error = temp / (prediction.shape[0] * prediction.shape[1] * prediction.shape[2])
    print (error)
    print("Model_4244 error: %.2f%%" % (error * 100))

def model_conv_4241_test(rawdata):

    n_frames = 4
    n_hours = 4
    n_cols = 1

    dataTest, len1 = DataUtil.getData(rawdata, startpoint='2017/02/08', endpoint='2017/03/22', n_hours=n_hours)

    # start point -- use moving_len to get the predicted starting date
    dateStr, len2 = DataUtil.getData(rawdata, startpoint='2017/02/08', endpoint='2017/03/22', n_hours=n_hours,
                                     moving_len=n_frames)
    dateStr = dateStr[:, 0]
    dataTest = dataTest[:,5]


    # dataTest = dataTest[:,103]



    temp_dataX_Test = dataTest[0:(dataTest.shape[0] - n_hours)]
    temp_dataY_Test = dataTest[n_frames * n_hours:dataTest.shape[0]]


    temp_dataX_Test = np.reshape(temp_dataX_Test, -1)
    temp_dataY_Test = np.reshape(temp_dataY_Test, -1)

    dataX_Test = []
    dataY_Test = []
    dateTimeList = []

    for i in range(temp_dataX_Test.shape[0] - n_frames * n_hours * n_cols + n_hours * n_cols):
        if i % (n_hours * n_cols) == 0:
            dataX_Test.append(temp_dataX_Test[i:i + n_frames * n_hours * n_cols])
    for i in range(temp_dataY_Test.shape[0]):
        if i % (n_hours * 1) == 0:
            dataY_Test.append(temp_dataY_Test[i])
            dateTimeList.append(dateStr[i])

    dataX_Test = np.reshape(dataX_Test, (np.array(dataX_Test).shape[0], n_frames * n_hours, n_cols, 1))
    dataY_Test = np.reshape(dataY_Test, (np.array(dataY_Test).shape[0], -1))



    model = model_from_json(
        open('/home/darren/PycharmProjects/conv_lstm/conv_column/model_4241_conv_architecture.json').read())
    model.load_weights('/home/darren/PycharmProjects/conv_lstm/conv_column/model_4241_conv_weights.h5')

    print('Score...')
    prediction = model.predict(dataX_Test, verbose=0)
    print('Predict...')

    temp = 0.0
    print(prediction)

    print('label...')
    print(dataY_Test)

    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            temp = temp + abs(prediction[i][j] - dataY_Test[i][j]) / dataY_Test[i][j]
    error = temp / (prediction.shape[0] * prediction.shape[1])
    print("Model_conv_column error: %.2f%%" % (error * 100))

    dates = []
    # dates = np.linspace(0, 1, 100)
    for i in range(np.array(dateTimeList).shape[0]):
        temp = time.strptime(dateTimeList[i], "%Y/%m/%d")  # 字符串转换成time类型
        temp = datetime.datetime(temp[0], temp[1], temp[2])  # time类型转换成datetime类型
        dates.append(temp)

    pylab.plot_date(dates, prediction, linestyle='-', label="$ConvColumnError:$" + '%.2f' % (error * 100) + '%',
                    color="red")
    # pylab.plot_date(dates, dataY_Test, linestyle='-', label="$label$", color="blue")
    # x = np.linspace(0, 1, 100)
    # x = [n for n in range(0, prediction.shape[0])]
    # plt.plot(x, prediction, label="$ConvColumnError:$"+'%.2f' %(error*100)+'%', color="red")
    # plt.plot(x, dataY_Test, color="blue", label="$label$")
    plt.legend()

    plt.xlabel("Time(day)")
    plt.ylabel("Value")
    plt.title("ShangZhengIndex_DependentNomorlized")
    # plt.show()

    plt.savefig("ShangZhengIndex_DependentNomorlized.png")

    plt.close('all')
    # x = np.linspace(0, 1, 50)
    # x = [n for n in range(0, prediction.shape[0])]
    # plt.plot(x, prediction, label="$ConvColumnError:$" + '%.2f' % (error * 100) + '%', color="red")
    # # plt.plot(x, dataY_Test, color="blue", label="$label$")
    # plt.legend()

    # plt.xlabel("Time(day)")
    # plt.ylabel("Value")
    # plt.title("ShangZhengIndex_NoNomorlized")
    # plt.show()
    # plt.savefig("TrafficFlow.png")

def model_lstm_oneFeature_test(rawdata):

    n_frames = 4
    n_hours = 4
    n_cols = 1

    dataTest, len1 = DataUtil.getData(rawdata, startpoint=' 2017/01/03-10:30', endpoint=' 2017/03/24-15:00', n_hours=n_hours)

    # start point -- use moving_len to get the predicted starting date
    dateStr, len2 = DataUtil.getData(rawdata, startpoint=' 2017/01/03-10:30', endpoint=' 2017/03/24-15:00', n_hours=n_hours,
                                     moving_len=n_frames)
    dateStr = dateStr[:, 0]
    dataTest = dataTest[:,4]


    temp_dataX_Test = dataTest[0:(dataTest.shape[0] - n_hours)]
    temp_dataY_Test = dataTest[n_frames * n_hours:dataTest.shape[0]]


    temp_dataX_Test = np.reshape(temp_dataX_Test, -1)
    temp_dataY_Test = np.reshape(temp_dataY_Test, -1)

    dataX_Test = []
    dataY_Test = []
    dateTimeList = []

    for i in range(temp_dataX_Test.shape[0] - n_frames * n_hours * n_cols + n_hours * n_cols):
        if i % (n_hours * n_cols) == 0:
            dataX_Test.append(temp_dataX_Test[i:i + n_frames * n_hours * n_cols])
    for i in range(temp_dataY_Test.shape[0]):
        if i % (n_hours * 1) == 0:
            dataY_Test.append(temp_dataY_Test[i+ n_hours -1])
            dateTimeList.append(dateStr[i])

    dataX_Test = np.reshape(dataX_Test, (np.array(dataX_Test).shape[0], np.array(dataX_Test).shape[1], 1))
    dataY_Test = np.reshape(dataY_Test, (np.array(dataY_Test).shape[0], -1))



    model = model_from_json(
        open('../lstm/model_lstm_oneFeature_architecture.json').read())
    model.load_weights('../lstm/model_lstm_oneFeature_weights.h5')

    print('Score...')
    prediction = model.predict(dataX_Test, verbose=0)
    print('Predict...')

    temp = 0.0
    print(prediction)

    print('label...')
    print(dataY_Test)

    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            temp = temp + abs(prediction[i][j] - dataY_Test[i][j]) / dataY_Test[i][j]
    error = temp / (prediction.shape[0] * prediction.shape[1])
    print("Model_conv_column error: %.2f%%" % (error * 100))

    match_percent = DataUtil.tendencyMatch(dataY_Test, prediction)
    print("match_percent is: %.2f%%" % (match_percent * 100))

    dates = []
    # dates = np.linspace(0, 1, 100)
    for i in range(np.array(dateTimeList).shape[0]):
        temp = time.strptime(dateTimeList[i], " %Y/%m/%d-%H:%M")  # 字符串转换成time类型
        temp = datetime.datetime(temp[0], temp[1], temp[2],temp[3],temp[4])  # time类型转换成datetime类型
        dates.append(temp)

    pylab.plot_date(dates, prediction, linestyle='-', label="$lstmError:$" + '%.2f' % (error * 100) + '%',
                    color="red")
    pylab.plot_date(dates, dataY_Test, linestyle='-', label="$label$", color="blue")
    # x = np.linspace(0, 1, 100)
    # x = [n for n in range(0, prediction.shape[0])]
    # plt.plot(x, prediction, label="$ConvColumnError:$"+'%.2f' %(error*100)+'%', color="red")
    # plt.plot(x, dataY_Test, color="blue", label="$label$")
    plt.legend()

    plt.xlabel("Time(day)")
    plt.ylabel("Value")
    plt.title("match_percent is: %.2f%%" % (match_percent * 100))
    # plt.show()

    plt.savefig("./lstm_model/lstm_oneFeature.png")

    plt.close('all')
    # x = np.linspace(0, 1, 50)
    # x = [n for n in range(0, prediction.shape[0])]
    # plt.plot(x, prediction, label="$ConvColumnError:$" + '%.2f' % (error * 100) + '%', color="red")
    # # plt.plot(x, dataY_Test, color="blue", label="$label$")
    # plt.legend()

    # plt.xlabel("Time(day)")
    # plt.ylabel("Value")
    # plt.title("ShangZhengIndex_NoNomorlized")
    # plt.show()
    # plt.savefig("TrafficFlow.png")


def model_lstm_MultipleFeature_test(rawdata):

    n_frames = 4
    n_hours = 4
    n_cols = 5

    dataTest, len1 = DataUtil.getData(rawdata, startpoint=' 2017/01/03-10:30', endpoint=' 2017/03/24-15:00', n_hours=n_hours)

    # start point -- use moving_len to get the predicted starting date
    dateStr, len2 = DataUtil.getData(rawdata, startpoint=' 2017/01/03-10:30', endpoint=' 2017/03/24-15:00', n_hours=n_hours,
                                     moving_len=n_frames)
    dateStr = dateStr[:, 0]
    dataTest = dataTest[:,[1,2,3,4,5]]


    temp_dataX_Test = dataTest[0:(dataTest.shape[0] - n_hours)]
    temp_dataY_Test = dataTest[n_frames * n_hours:dataTest.shape[0]]

    # get close column
    temp_dataY_Test = temp_dataY_Test[:, 3]


    temp_dataX_Test = np.reshape(temp_dataX_Test, -1)
    temp_dataY_Test = np.reshape(temp_dataY_Test, -1)

    dataX_Test = []
    dataY_Test = []
    dateTimeList = []

    for i in range(temp_dataX_Test.shape[0] - n_frames * n_hours * n_cols + n_hours * n_cols):
        if i % (n_hours * n_cols) == 0:
            dataX_Test.append(temp_dataX_Test[i:i + n_frames * n_hours * n_cols])
    for i in range(temp_dataY_Test.shape[0]):
        if i % (n_hours * 1) == 0:
            dataY_Test.append(temp_dataY_Test[i+ n_hours -1])
            dateTimeList.append(dateStr[i])

    dataX_Test = np.reshape(dataX_Test, (np.array(dataX_Test).shape[0], n_hours*n_frames, n_cols))
    dataY_Test = np.reshape(dataY_Test, (np.array(dataY_Test).shape[0], -1))



    model = model_from_json(
        open('../lstm/model_lstm_MultipleFeature_architecture.json').read())
    model.load_weights('../lstm/model_lstm_MultipleFeature_weights.h5')

    print('Score...')
    prediction = model.predict(dataX_Test, verbose=0)
    print('Predict...')

    temp = 0.0
    print(prediction)

    print('label...')
    print(dataY_Test)

    for i in range(prediction.shape[0]):
        for j in range(prediction.shape[1]):
            temp = temp + abs(prediction[i][j] - dataY_Test[i][j]) / dataY_Test[i][j]
    error = temp / (prediction.shape[0] * prediction.shape[1])
    print("Model_conv_column error: %.2f%%" % (error * 100))

    match_percent = DataUtil.tendencyMatch(dataY_Test, prediction)
    print("match_percent is: %.2f%%" % (match_percent * 100))

    dates = []
    # dates = np.linspace(0, 1, 100)
    for i in range(np.array(dateTimeList).shape[0]):
        temp = time.strptime(dateTimeList[i], " %Y/%m/%d-%H:%M")  # 字符串转换成time类型
        temp = datetime.datetime(temp[0], temp[1], temp[2],temp[3],temp[4])  # time类型转换成datetime类型
        dates.append(temp)

    pylab.plot_date(dates, prediction, linestyle='-', label="$lstmError:$" + '%.2f' % (error * 100) + '%',
                    color="red")
    pylab.plot_date(dates, dataY_Test, linestyle='-', label="$label$", color="blue")
    # x = np.linspace(0, 1, 100)
    # x = [n for n in range(0, prediction.shape[0])]
    # plt.plot(x, prediction, label="$ConvColumnError:$"+'%.2f' %(error*100)+'%', color="red")
    # plt.plot(x, dataY_Test, color="blue", label="$label$")
    plt.legend()

    plt.xlabel("Time(day)")
    plt.ylabel("Value")
    plt.title("match_percent is: %.2f%%" % (match_percent * 100))
    # plt.show()

    plt.savefig("./lstm_model/lstm_MultipleFeature.png")

    plt.close('all')
    # x = np.linspace(0, 1, 50)
    # x = [n for n in range(0, prediction.shape[0])]
    # plt.plot(x, prediction, label="$ConvColumnError:$" + '%.2f' % (error * 100) + '%', color="red")
    # # plt.plot(x, dataY_Test, color="blue", label="$label$")
    # plt.legend()

    # plt.xlabel("Time(day)")
    # plt.ylabel("Value")
    # plt.title("ShangZhengIndex_NoNomorlized")
    # plt.show()
    # plt.savefig("TrafficFlow.png")

#load testing raw data
rawdata_test = pd.read_csv("../data/上证指数/3年数据_Volume列无归一化.csv",encoding='gbk').as_matrix()
# model_4244_test(rawdata_test)
# model_4241_test(rawdata_test)
# model_mlp_4241_test(rawdata_test)
# model_merge_test(rawdata_test)
# model_conv_4244_test(rawdata_test)
# model_conv_4241_test(rawdata_test)
model_lstm_oneFeature_test(rawdata_test)
model_lstm_MultipleFeature_test(rawdata_test)










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
