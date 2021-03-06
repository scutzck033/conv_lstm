#coding:utf-8
# # Use scikit-learn to grid search the batch size and epochs
# import numpy
# from sklearn.model_selection import GridSearchCV
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.wrappers.scikit_learn import KerasClassifier
#
#
# # solve the problem: TypeError: get_params() got an unexpected keyword argument 'deep'
# from keras.wrappers.scikit_learn import BaseWrapper
# import copy
#
# def custom_get_params(self, **params):
#     res = copy.deepcopy(self.sk_params)
#     res.update({'build_fn': self.build_fn})
#     return res
#
# BaseWrapper.get_params = custom_get_params
#
#
# # Function to create model, required for KerasClassifier
# def create_model():
# 	# create model
# 	model = Sequential()
# 	model.add(Dense(12, input_dim=8, activation='relu'))
# 	model.add(Dense(1, activation='sigmoid'))
# 	# Compile model
# 	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 	return model
# # fix random seed for reproducibility
# seed = 7
# numpy.random.seed(seed)
# # load dataset
# dataset = numpy.loadtxt("/home/darren/PycharmProjects/conv_lstm/data/pima-indians-diabetes.csv", delimiter=",")
# # split into input (X) and output (Y) variables
# X = dataset[:,0:8]
# Y = dataset[:,8]
# # create model
# model = KerasClassifier(build_fn=create_model, verbose=0)
# # define the grid search parameters
# batch_size = [10, 20, 40, 60, 80, 100]
# epochs = [10, 50, 100]
# param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
# grid_result = grid.fit(X, Y)
# # summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))


# import dateutil, pylab, random
# from pylab import *
# from datetime import datetime, timedelta
# import time
# import datetime
#
#
# today="2012/04/05"
# today=time.strptime(today,"%Y/%m/%d")                            #字符串转换成time类型
# print type(today) #查看date的类型<type 'time.struct_time'>
# today=datetime.datetime(today[0],today[1],today[2])               #time类型转换成datetime类型
# print type(today) #查看date的类型<type 'datetime.datetime'>
#
# dates = [today + timedelta(days=i) for i in range(10)]
# # values = [random.randint(1, 20) for i in range(10)]
# values = [3, 2, 8, 4, 5, 6, 7, 8, 11, 2]
# # plt.plot(dates,values)
# pylab.plot_date(dates, values, linestyle='-')
# grid(True)
# plt.xlabel("Time(day)")
# plt.ylabel("Value")
# plt.title("mlp_441")
# #
# # # savefig('simple_plot.png')
# #
# show()
import pandas as pd
import sys
sys.path.append('/home/darren/PycharmProjects/conv_lstm/utils')
from DataUtil import DataUtil

rawdata_test = pd.read_csv("./data/szcz0411.csv",encoding='gbk').as_matrix()

dataTest,len = DataUtil.getData(rawdata_test,startpoint=' 2016/12/23-11:30',endpoint=' 2017/01/10-10:30',n_hours=4)

# if(cmp('2016/11/21-10:30','2016/11/21-10:30') == 0):
#     print ('the same')
# else:
#     print ('different')

print (dataTest)
print (len)
print (dataTest.shape)

# import pandas as pd
# import numpy as np
# import csv
#
# def writeCSV(rawdata,n_clos):
#     csvfile = file('../conv_lstm/data/pems_jun_2014_train.csv', 'wb')#w means write;b means document
#     writer = csv.writer(csvfile)
#     # writer.writerow(str(n) for n in range(n_clos))#writerow writes one row
#
#     writer.writerows(rawdata)#writerows writes multiple rows
#
#     csvfile.close()
#
# rawdata_test = pd.read_csv("./data/rawdata/pems_July-2014.csv",encoding='gbk').as_matrix()
# # print (rawdata_test)
# # print (rawdata_test.shape)
# rawdata_test = np.transpose(rawdata_test)[4:rawdata_test.shape[1]]
# print (rawdata_test)
# # writeCSV(rawdata_test,rawdata_test.shape[1])
#
# rawdata_train = pd.read_csv("./data/pems_jun_2014_train.csv",encoding='gbk')
# print (rawdata_train.shape)