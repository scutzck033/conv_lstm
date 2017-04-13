import pandas as pd
import numpy as np
import csv

#change the unit of the window size

#write processed data
def writeCSV(rawdata):
    csvfile = file('../conv_lstm/data/szcz0411.csv', 'wb')#w means write;b means document
    writer = csv.writer(csvfile)
    writer.writerow(['datetime','open','max','min','close','volume'])#writerow writes one row

    writer.writerows(rawdata)#writerows writes multiple rows

    csvfile.close()

#count the total volume or total turnover
def volume(rawdata,index,scalar,column_index):
    temp = 0
    for i in range(scalar):
        temp+=rawdata[index+i][column_index]
    return temp

#get the max value of the list
def getMax(rawdata,index,scalar,column_index):
    temp = 0
    for i in range(scalar):
        if rawdata[index+i][column_index]>=temp:
            temp=rawdata[index+i][column_index]
    return temp

#get the min value of the list
def getMin(rawdata,index,scalar,column_index):
    temp2 = rawdata[index][column_index]
    for i in range(scalar):
        if rawdata[index+i][column_index]<=temp2:
            temp2=rawdata[index+i][column_index]
    return temp2


# #data processed -- change the time unit of the data
def dataProcessed(rawdata,scalar):
    temp=[]
    temp_close=[]
    temp_volume=[]
    temp_turnover=[]
    temp_max=[]
    temp_min=[]
    for i in range(rawdata.shape[0]):
        if i%(scalar) == 0:
            if (rawdata.shape[0]-1-i)>=scalar:
                temp.append(rawdata[i][0:3])
                temp_close.append(rawdata[i+scalar-1][5])
                temp_volume.append(volume(rawdata,i,scalar,column_index=6))
                temp_turnover.append(volume(rawdata, i, scalar,column_index=7))
                temp_max.append(getMax(rawdata,i,scalar,column_index=3))
                temp_min.append(getMin(rawdata, i, scalar, column_index=4))
            else:
                temp.append(rawdata[i][0:3])
                temp_close.append(rawdata[rawdata.shape[0]-1][5])
                temp_volume.append(volume(rawdata, i, rawdata.shape[0]-i,column_index=6))
                temp_turnover.append(volume(rawdata, i, rawdata.shape[0] - i,column_index=7))
                temp_max.append(getMax(rawdata, i, rawdata.shape[0] - i, column_index=3))
                temp_min.append(getMin(rawdata, i, rawdata.shape[0] - i, column_index=4))
    pieces=[temp,temp_max,temp_min,temp_close,temp_volume,temp_turnover]

    return np.column_stack(pieces)
#     writeCSV((np.column_stack(pieces)))
#
# #load raw data
# rawdata_train = pd.read_table("4h per day(1min).txt",encoding='gbk',delimiter='\t')
#
#
# rawdata_train=rawdata_train.as_matrix()
# dataProcessed(rawdata_train,60)#5min per unit changed to 1h per unit



#load raw data
rawdata = pd.read_excel("../conv_lstm/data/rawdata/szcz0411.xls").as_matrix()

# rawdata=dataProcessed(rawdata,12) #5min per unit changed to 1h per unit

rawdata = rawdata[3:rawdata.shape[0]-1]

rawdata_train=rawdata[:,[1,2,3,4,5]]

# print (rawdata_train)

# Normilization
# margin_list = []
# min_list = []
# for i in range(rawdata_train.shape[1]):
#     margin_list.append(rawdata_train[:,i].max()-rawdata_train[:,i].min())
#     min_list.append(rawdata_train[:,i].min())

max_list = []
min_list = []
for i in range(rawdata_train.shape[1]-1):
    max_list.append(rawdata_train[:,i].max())
    min_list.append(rawdata_train[:,i].min())
maxVal = max(max_list)
minVal = min(min_list)

for i in range(rawdata_train.shape[0]):
    for j in range(rawdata_train.shape[1]-1):
        rawdata_train[i][j]=float(rawdata_train[i][j]-minVal)/float(maxVal - minVal)

# # shift the Volume
rawdata_train[:,4]=rawdata_train[:,4]/100000
pieces=[rawdata[:,0],rawdata_train]
writeCSV((np.column_stack(pieces)))