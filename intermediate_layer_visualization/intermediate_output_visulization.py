from keras import backend as K
from keras.models import model_from_json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

# load the model
model = model_from_json(open('/home/darren/PycharmProjects/conv_lstm/conv_column/model_4241_conv_architecture.json').read())
model.load_weights('/home/darren/PycharmProjects/conv_lstm/conv_column/model_4241_conv_weights.h5')

# load the data
rawdata_test = pd.read_csv("../data/4h per day(1h)_normalized.csv",encoding='gbk')
rawdata_test = rawdata_test[[2]]#get one column
dataTest = rawdata_test.as_matrix()[160:244]
temp_dataX_Test = []


n_frames = 4
n_hours = 4
n_cols = 1

temp_dataX_Test=dataTest[0:(dataTest.shape[0]-n_hours)]

temp_dataX_Test=np.reshape(temp_dataX_Test,-1)


dataX_Test =[]


for i in range(temp_dataX_Test.shape[0]-n_frames*n_hours*n_cols+n_hours*n_cols):
    if i%(n_hours*n_cols) == 0:
        dataX_Test.append(temp_dataX_Test[i:i+n_frames*n_hours*n_cols])


dataX_Test=np.reshape(dataX_Test,(np.array(dataX_Test).shape[0],n_frames*n_hours,n_cols,1))



# with a Sequential model
get_1st_layer_output = K.function([model.layers[0].input],
                                [model.layers[0].output])
layer_output = get_1st_layer_output([dataX_Test])[0]


x = np.linspace(0, 1, 20)
# tempX = np.transpose(np.transpose(layer_output[0])[0:3])

tempY=np.transpose(np.reshape(layer_output[0],(layer_output[0].shape[0],layer_output[0].shape[2]))[:,0:3])
# print (tempY.shape)
# print (tempY)
x = [n for n in range(1,tempY.shape[1]+1)]
y = np.reshape(dataX_Test[0],dataX_Test.shape[1]) # the original data

label=['$intermidate1$','$intermidate2$','$intermidate3$']
color=['red','blue','green']

plt.figure(figsize=(8, 4))
for i in range(tempY.shape[0]):
    plt.plot(x, tempY[i], label=label[i], color=color[i], linewidth=2)#the feature data
plt.plot(x,y,label='$original$',color='black')
plt.xlabel("Time(s)")
plt.ylabel("Volt")
plt.title("intermidate")
plt.ylim(-0.125, 0.4)
plt.legend()
# plt.show()

savefig('../intermediate_layer_visualization/save.png')
