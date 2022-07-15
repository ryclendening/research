#!/usr/bin/python3

# Import packages
import csv
import wave
import os
import re
import struct
import datetime
# import folium
# import webbrowser
# import branca.colormap as cm
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import tensorflow as tf
from tensorflow import keras
# import tensorflow_probability as tfp
from keras.models import Model
from keras.optimizers import adam_v2
from keras.layers import Input, Conv1D, SeparableConv1D, MaxPooling1D, Flatten, Dense, Dropout, \
    BatchNormalization, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# tfd = tfp.distributions

# User variables
retrain = False
drones = ['IF1200/', 'M600/', 'Phantom4/', 'Mavic/']
dates = ['20210823/', '20210824/', '20210825/', '20210826/',
         '20210827/']
weights_load = 'multLab_ID_detect3.hdf5'  # if retrain = False these are the weights that get loaded  # 3 is best so far
weights_save = 'multLab_ID_detect3.hdf5'  # if retrain = True these are the weights that get saved
dirWeights = '/home/bsmolenski/ESCAPE2_flightLogs_runSheets_audio/Weights_multiLab/'

dirRuns = '/home/bsmolenski/ESCAPE2_flightLogs_runSheets_audio/Runs_csv/'
dirGPS = '/home/bsmolenski/ESCAPE2_flightLogs_runSheets_audio/GPS_logs_csv/'
dirAudio = '/home/bsmolenski/ESCAPE2_flightLogs_runSheets_audio/Audio_Channel1/'
pathNoise = '/home/bsmolenski/Environmental_Noise/envNoise3.wav'
numClass = len(drones) + 1
# Audio data parameters
fs = 48000  # sampling frequency
lenChunk = int(fs / 2)  # chunk length for PSD calculation in number of samples
lenFrame = int(fs / 100)  # frame length in number of samples
frameInt = int(lenFrame / 2)  # frame interval in number of samples
win = np.hanning(lenFrame + 2)  # window function
win = win[1:-1]
numFeat = 1 + int(lenFrame / 2)
# Measured array and mic locations in meters   43.13'56"N 75.24'39"W (@480ft)
lat0 = 43 + 13. / 60 + 56. / 3600  # coordinates of array GPS
lon0 = -(75 + 24. / 60 + 39. / 3600)
alt0 = 146.304  # altitude in meters
# angle = np.pi*22/180  #magnetic north-west
# xArrayMics = [-1.524, -0.7112, -0.3048, -0.1016, 0.1016, 0.3048, 0.7112, 1.524]  # in meters
# yArrayMics = [-1.524, -0.7112, -0.3048, -0.1016, 0.1016, 0.3048, 0.7112, 1.524]
# zArrayMics = [-0.4318, -1.0922]  #-17", -43" derived from 21" 24" 26" starting from ground, with 80" for the GPS sensor at the top and 8" for mic height

'''
#Extract GPS log data and match to corresponding scene, pass, and run -
still in development
gps_timeStamp = []
lat = []
lon = []
alt = []
for day in dates:
    fileName_csv = os.listdir(dirRuns + day)
    with open(dirRuns + day + fileName_csv[0], newline='') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[1].isdigit():
                Pass = int(row[1])
                scene = row[2]
                run = int(row[3])
                UASs = [row[8], row[11], row[14], row[17]]
                start_time = int(row[20][9:]) - 40000
                stop_time = int(row[22][9:]) - 40000
                start_time2 = datetime.datetime(int(day[0:4]),
int(day[4:6]), int(day[6:8]), int(row[20][9:11])-4,
int(row[20][11:13]), int(row[20][13:])).timestamp()
                stop_time2 = datetime.datetime(int(day[0:4]),
int(day[4:6]), int(day[6:8]), int(row[22][9:11])-4,
int(row[22][11:13]), int(row[22][13:])).timestamp()
                flying = []
                for UAS in UASs:
                    if UAS:  #if not an empty string
                       flying.append(drones[int(UAS[2])-1])
                for UAS in flying:
                    if UAS == 'IF1200/':
                        names_logs = os.listdir(dirGPS + UAS + day)
                        for name in names_logs:
                            with open(dirGPS + UAS + day + name,
newline='') as logFile:
                                readLog = csv.DictReader(logFile)
                                for row in readLog:
                                    if
row['time_utc_usec'].isnumeric():
                                        if
round(int(row['time_utc_usec'])/1000000.,0) == start_time2 or
round(int(row['time_utc_usec'])/1000000.,0) == stop_time2:
                                        #if (int(row['GPS(0):Time']) >=
start_time and int(row['GPS(0):Time']) <=
stop_time):     #time_utc_usec
                                            print(UAS + day + name,
Pass, scene, run, row['lat'], row['lon'],
round(int(row['time_utc_usec'])/1000000.,0), row['alt'])
                                            #print(start_time2)
                                            break
                    else:
                        names_logs = os.listdir(dirGPS + UAS + day)
                        for name in names_logs:
                            with open(dirGPS + UAS + day + name,
newline='') as logFile:
                                readLog = csv.DictReader(logFile)
                                for row in readLog:
                                    if row['GPS(0):Time'].isnumeric():
                                        if int(row['GPS(0):Time']) ==
start_time or int(row['GPS(0):Time']) == stop_time2:
                                        #if (int(row['GPS(0):Time']) >=
start_time and int(row['GPS(0):Time']) <=
stop_time):     #time_utc_usec
                                            print(UAS + day + name,
Pass, scene, run, row['GPS(0):Lat'], row['GPS(0):Long'],
row['GPS(0):Time'], row['GPS(0):heightMSL'])   
                                            break

                                    if line[4]:   
                                        gps_timeStamp.append(line[1])
                                        lat.append(line[2])
                                        lon.append(line[3])
                                        alt.append(line[4])
                                        print(name, line[1], line[2],
line[3], line[4])
                                        break
                                    elif line[5]:
                                        gps_timeStamp.append(line[8])
                                        lat.append(line[6])
                                        lon.append(line[5])
                                        alt.append(line[10])
                                        print(name, line[8], line[6],
line[5], line[10])
                                        break

'''


# Function for log-PSD feature extraction from wav file path


def PSD_log(path, lenFrame, frameInt):
    waveObj = wave.open(path, 'rb')
    lengthH = waveObj.getnframes()
    temp = waveObj.readframes(lengthH)
    data = struct.unpack("<" + str(lengthH) + "h", temp)
    waveObj.close()
    cnt = 0
    numChunks = np.int((len(data) - lenChunk) / lenChunk)
    logPSD = []
    for t in range(numChunks):  # extract STFTlogmag from drone recording
        chunk = data[cnt:cnt + lenChunk]
        f_dum, PSD = signal.welch(chunk, fs, window=win,
                                  nperseg=lenFrame, noverlap=frameInt, nfft=lenFrame, average='mean')
    logPSD.append(np.log(PSD + np.finfo(float).eps))
    cnt = cnt + lenChunk
    return logPSD


# Extract UAS multi-labels and corresponding features from adudio file for each scene, pass, and run
features = []
labels = []
info = []
for day in dates:
    fileName_csv = os.listdir(dirRuns + day)
    fileNames_audio = os.listdir(dirAudio + day)
    with open(dirRuns + day + fileName_csv[0], newline='') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[1].isdigit():
                cnt = 0
                Pass = int(row[1])
                scene = row[2]
                run = int(row[3])
                UASs = [row[8], row[11], row[14], row[17]]
                labelVec = np.zeros(numClass)
                for UAS in UASs:
                    if UAS:  # if not an empty string
                        labelVec[int(UAS[2]) - 1] = 1.
                for audioName in fileNames_audio:
                    cnt = cnt + 1
                    if (audioName[5:7] == scene and
                            int(audioName[8:10]) == run and int(audioName[11:13]) == Pass):
                        path = dirAudio + day + audioName
                        logPSD = PSD_log(path, lenFrame, frameInt)
                        features.extend(logPSD)
                        for n in range(len(logPSD)):
                            labels.append(labelVec)
                            info.append(UASs)
                        print('Extracting features from: ' + dirAudio +
                              day + audioName)
                        break
                    elif cnt == len(fileNames_audio):
                        print('Could not find audio for: Pass =', Pass,
                              'Scene =', scene, 'Run =', run)

# Get environmental noise and add labels
labelVec = np.asarray([0., 0., 0., 0., 1.])
logPSD = PSD_log(pathNoise, lenFrame, frameInt)
features.extend(logPSD)
print('Extracting features from:', pathNoise)
for n in range(len(logPSD)):
    labels.append(labelVec)
    info.append('Noise')

# Create training/test split
trainFeatures, testFeatures, trainTargets, testTargets = train_test_split(features, labels, test_size=0.10,
                                                                          random_state=42)
train_info, test_info = train_test_split(info, test_size=0.10,
                                         random_state=42)

# Standardize features
trainFeatures = np.asarray(trainFeatures)
trainTargets = np.asarray(trainTargets)
mean = trainFeatures.mean(axis=0)
trainFeatures -= mean
std = trainFeatures.std(axis=0)
trainFeatures /= std
trainFeatures = np.reshape(trainFeatures,
                           (len(trainFeatures), numFeat, 1))
testFeatures = np.asarray(testFeatures)
testTargets = np.asarray(testTargets)
testFeatures -= mean
testFeatures /= std
testFeatures = np.reshape(testFeatures, (len(testFeatures), numFeat, 1))

# Probabilistic range regression network
# def NLL(y, distr):  #Loss function
#    return -distr.log_prob(y)
# def normal_sp(params):  #Distribution function
#    return tfd.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:, 1:2]))  # Both mean and variance parameters are learnable
inputs = Input(shape=(numFeat, 1))
hidden = Conv1D(32, 12, strides=2, use_bias=False, name='conv_1')(inputs)
hidden = BatchNormalization()(hidden)
hidden = Activation('relu')(hidden)
# hidden = MaxPooling1D(3, strides=2)(hidden)
hidden = SeparableConv1D(32, 6, strides=2, use_bias=False,
                         name='conv_2')(hidden)
hidden = BatchNormalization()(hidden)
hidden = Activation('relu')(hidden)
# hidden = MaxPooling1D(3, strides=2)(hidden)
hidden = SeparableConv1D(32, 3, strides=2, use_bias=False,
                         name='conv_3')(hidden)
hidden = BatchNormalization()(hidden)
hidden = Activation('relu')(hidden)
# hidden = MaxPooling1D(3, strides=2)(hidden)
hidden = Flatten()(hidden)
hidden = Dropout(0.1)(hidden)
hidden = Dense(864, activation='relu', name='dense_1')(hidden)  # 896,
864, 928
hidden = Dropout(0.1)(hidden)
dist = Dense(numClass, activation='sigmoid', name='dense_2')(hidden)
# params = Dense(2, name='dense_2')(hidden)
# dist = tfp.layers.DistributionLambda(normal_sp)(params)
network = Model(inputs=inputs, outputs=dist)
network.summary()
network.compile(optimizer='adam',
                loss='binary_crossentropy')  # loss=NLL  #, metrics=['accuracy']

# Train classifier
if retrain:
    call2 = ModelCheckpoint(filepath=(dirWeights + weights_save),
                            verbose=1, monitor='val_loss', save_best_only=True)
    reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                 verbose=1, patience=10, mode='auto')
    early = EarlyStopping(monitor='val_loss', min_delta=1e-4,
                          patience=40, mode='auto')
    history = network.fit(trainFeatures, trainTargets,
                          epochs=1000,
                          batch_size=32,
                          shuffle=True,
                          validation_split=0.2,
                          callbacks=[reduceLR, early, call2])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Learning Curves')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.ylabel('Binary Crossentropy')
    plt.xlabel('Epoch Number')
    plt.show()
    network.load_weights(dirWeights + weights_load, by_name=True)
else:
    network.load_weights(dirWeights + weights_load, by_name=True)

# Evaluate
predicted_test = network.predict(testFeatures)
FA = 0
indFA = []
Miss = 0
indMiss = []
cnt = 0
for n in range(len(predicted_test)):
    for k in range(numClass):
        if predicted_test[n][k] > 0.5:
            predicted_test[n][k] = 1.
            if predicted_test[n][k] != testTargets[n][k]:
                FA = FA + 1
                indFA.append(n)
                print('Predicted 1, but was 0 for UAS:', k + 1, n)
        else:
            predicted_test[n][k] = 0.
            if predicted_test[n][k] != testTargets[n][k]:
                Miss = Miss + 1
                indMiss.append(n)
                print('Predicted 0, but was 1 for UAS:',
                      testTargets[n], test_info[n], k + 1, n)
    if sum(predicted_test[n]) != sum(testTargets[n]):
        cnt = cnt + 1
    # for n in indFA:
    #    print(predicted_test[n], testTargets[n])
    # for n in indMiss:
    #    print(predicted_test[n], testTargets[n])
print('Number of False Alarms:', FA, 'Number of Missed Detections:', Miss, 'Out Of', numClass * len(predicted_test))
print('Number of times wrong number of UAS was detected:', cnt, 'Out Of', len(predicted_test))
# predicted_train = network.predict(trainFeatures)

# Create experimental runs for non-Bayesian model
# runs = 800
# nobay_cpd = np.zeros((runs, len(testFeatures)))
# for i in range(0,runs):
#    nobay_cpd[i,:] = np.reshape(network.predict(testFeatures), len(testFeatures))
