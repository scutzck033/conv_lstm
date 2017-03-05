from keras import backend as K
from keras.models import model_from_json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

# load the model
model = model_from_json(open('model_4241_architecture.json').read())
model.load_weights('model_4241_weights.h5')

# load the data
rawdata_test = pd.read_csv("EURUSD60_test.csv",encoding='gbk')
rawdata_test = rawdata_test[[2]]#get one column
dataTest = rawdata_test.as_matrix()[0:888]
temp_dataX_Test = []


n_frames = 4
n_hours = 24
n_cols = 1

temp_dataX_Test=dataTest[0:(dataTest.shape[0]-n_hours)]

temp_dataX_Test=np.reshape(temp_dataX_Test,-1)


dataX_Test =[]


for i in range(temp_dataX_Test.shape[0]-n_frames*n_hours*n_cols+n_hours*n_cols):
    if i%(n_hours*n_cols) == 0:
        dataX_Test.append(temp_dataX_Test[i:i+n_frames*n_hours*n_cols])


dataX_Test=np.reshape(dataX_Test,(np.array(dataX_Test).shape[0],n_frames,n_hours,n_cols,1))



# with a Sequential model
get_1st_layer_output = K.function([model.layers[0].input],
                                [model.layers[0].output])
layer_output = get_1st_layer_output([dataX_Test])[0]

x = np.linspace(0, 1, 26)
x = [n for n in range(1,np.transpose(np.transpose(layer_output[0][0])[0]).shape[0]+1)]
y = np.transpose(np.transpose(layer_output[0][0])[0])

plt.figure(figsize=(8, 4))
plt.plot(x, y, label="$intermidate1$", color="red", linewidth=2)
plt.xlabel("Time(s)")
plt.ylabel("Volt")
plt.title("intermediate")
plt.ylim(0.275, 0.2795)
plt.legend()
plt.show()

savefig('/home/slave1/PycharmProjects/conv_lstm/intermediate_layer_visualization/save.png')