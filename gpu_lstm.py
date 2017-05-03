'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function
import numpy as np
import scipy.io as sio
import os
import math
import matplotlib.pyplot as plt
import random
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM,SimpleRNN,GRU
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
def get_session(gpu_fraction=0.3):
    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
KTF.set_session(get_session())
def getData(filename):
	fea = sio.loadmat('../data/features/'+filename)
	labels = sio.loadmat('../data/labels/'+filename)
	labels = labels['labels_data']
	fea = fea['de_data']
	labels = labels.tolist()
	newlabels = []
	for i in labels:
		newlabels.append(i[0])
	newlabels = np.array(newlabels)
	return fea,newlabels
def create_dataset(dataset, look_back=1):
    dataX = []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
    return np.array(dataX)
#mean_squared_error
#['chengjiejie_20151129_noon.mat', 0.64310765738891151, 0, 13]]
#chenting_20151124_noon_2.mat', 6.1611676376744963e-16
'''
maxcoe = 0
maxi = 0
for i in range(10,15):
	mse,coe = processOnePerson('huqingli_20151122_night.mat',i)
	if(coe>maxcoe):
		maxcoe = coe
		maxi = i
		print('new coe is ',maxcoe,' max i is',maxi)
'''
def processOnePerson(filename,lookback=3,dropout_value=0.5,learning_rate=1e-1,epoch=10,layer_num=4):
    #initialize adam
    size_of_batch = 256
    myadam = optimizers.Adam(lr=learning_rate, epsilon=1e-8)
    fea,lab = getData(filename)
    #normalize 
    scaler = MinMaxScaler(feature_range=(0,1))
    fea = scaler.fit_transform(fea)
    # timestep
    fea = create_dataset(fea,lookback)#885xtimestepx85
    lab = lab[:885-1-lookback]        #885,
    lab = lab.tolist()
    #reshape input to [samples,time steps,features]
    length = 885-1-lookback
    PreLabConcat = []
    bestcoe=0
    bestmse=0
    gap0 = int(length*1/5)
    trainFea,preFea = fea[gap0:,:,:],fea[:gap0,:,:]
    trainLab,preLab = lab[gap0:],lab[:gap0]
    model = Sequential()
    model.add(GRU(layer_num,input_shape=(lookback,85)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=myadam)
    model.fit(trainFea, trainLab, nb_epoch=epoch, batch_size=size_of_batch, verbose=2)
    trainPredict = model.predict(preFea)
    trainPredict = trainPredict.tolist()
    for i in range(len(trainPredict)):
        PreLabConcat.append(trainPredict[i][0])
    for gapslice in range(1,4):
        gap0 = int(len(fea)*gapslice/5)
        gap1 = int(len(fea)*(gapslice+1)/5)
        trainFea,preFea = np.concatenate((fea[:gap0,:,:],fea[gap1:,:,:]),axis=0) ,fea[gap0:gap1,:,:]
        trainLab,preLab = np.concatenate((lab[:gap0],lab[gap1:]),axis=0) ,lab[gap0:gap1]
        model = Sequential()
        model.add(GRU(layer_num,input_shape=(lookback,85)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=myadam)
        model.fit(trainFea, trainLab, nb_epoch=epoch, batch_size=size_of_batch, verbose=2)
        trainPredict = model.predict(preFea)
        trainPredict = trainPredict.tolist()
        for i in range(len(trainPredict)):
            PreLabConcat.append(trainPredict[i][0])
    gap0 = int(len(fea)*4/5)
    trainFea,preFea = fea[:gap0,:,:],fea[gap0:,:,:]
    trainLab,preLab = lab[:gap0],lab[gap0:]
    model = Sequential()
    model.add(GRU(layer_num,input_shape=(lookback,85)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=myadam)
    model.fit(trainFea, trainLab, nb_epoch=epoch, batch_size=size_of_batch, verbose=2)
    trainPredict = model.predict(preFea)
    trainPredict = trainPredict.tolist()
    for i in range(len(trainPredict)):
        PreLabConcat.append(trainPredict[i][0])
    mse = mean_squared_error(PreLabConcat,lab)
    coe = pearsonr(PreLabConcat,lab)[0]
    print('mse',mse)
    return mse,coe
#*************************
def main():
    filelist = os.listdir('../data/labels')
    dataarray = []
    tobesaved = []
    for file_name in filelist:
        print(file_name)
        onevalue = []
        bestmse = 0
        bestcoe = 0
        besti = 0
        for timestep in [12]:
            for lr in range(50):
                di = random.random()*10
                zhishu = -random.randint(1,6)
                learning_rate = di*(10**zhishu)
                for layer_num in range(40,60):
                    mse,coe = processOnePerson(filename=file_name,lookback=timestep,dropout_value=0.5,learning_rate=learning_rate,epoch=10,layer_num=layer_num)
                    print('i am processing ',file_name,' the bestcoe by now',bestcoe)
                    if(coe>bestcoe):
                        bestcoe=coe
                        bestmse = mse
                        besti = timestep
        dataarray.append([file_name,bestcoe,bestmse,besti])
        tobesaved.append([bestcoe,bestmse,besti])
        sio.savemat('../data/output/jiaochaGRU.mat',{'allpeople':tobesaved})
        print('tempans is',dataarray)
    print(dataarray)
    
main()
"""
plt.plot(lab)
plt.plot(range(gap,885-1-lookback),trainPredict)
plt.plot(yuanPredict)
plt.show()
print(coe)
"""
#processOnePerson(filename='chengjiejie_20151129_noon.mat',lookback=10,dropout_value=0.5,learning_rate=1e-1,epoch=1,layer_num=10)
#pearsonr(trainPredict,lab)
#mean_squared_error10 0.80975890118 11 (0.81834044198867506  12 0.858959668415 0.0117

