import os
import pathlib
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models
from IPython import display

features = []
labels = []
dataset_path: str = "C:\\Users\\rclendening\\researchData\\Training_Data_NM_RS"
data_dir = pathlib.Path(dataset_path)
droneDict = {  # One hot encoding for labels probs should do it like I did below?
    "IF1200": [1, 0, 0, 0, 0],
    "Matrice_600": [0, 1, 0, 0, 0],
    "Mavic_Pro": [0, 0, 1, 0, 0],
    "Phantom_4_Pro_V2": [0, 0, 0, 1, 0],
    "Noise": [0, 0, 0, 0, 1]
}
droneCountDict = {  # One hot encoding for labels
    "IF1200": 0,
    "Matrice_600": 1,
    "Mavic_Pro": 2,
    "Phantom_4_Pro_V2": 3,
    "Noise": 4
}
dataCount = [0, 0, 0, 0, 0]
drones = np.array(tf.io.gfile.listdir(str(data_dir)))
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
train_files = filenames
# x = round((len(train_files) / 4))
# train_files = train_files[:x]
print("Total num of samples: ", num_samples)
print("Number of examples per label:", len(tf.io.gfile.listdir(str(data_dir / drones[0]))))
print("Example file tensor: ", filenames[0])
print(drones)
print(train_files)
test_file = tf.io.read_file(
    "C:\\Users\\rclendening\\researchData\\Training_Data_NM_RS\\IF1200\\d301sA1r01p0120210823_6.wav")
test_audio, _ = tf.audio.decode_wav(contents=test_file)
test_audio.shape


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(input=file_path, sep=os.path.sep)
    return parts[-2]


def split_audio(waveData, labelName, sampleFreq, frame_duration):
    '''
    Frames audio data and converts to feature space (spectrogram)
    :param waveData: waveData array of time-domain audio
    :param frame_duration: Duration of frames desired
    :param startTime: Start for each clip
    :param sampleFreq: Sample Frequency (8Khz)
    :param labelName: Name of label
    @return list of features (ds), list of labels corresponding to feature dataset:
    '''
    features = []
    label = []
    # middle third of data
    duration = waveData.shape[0]
    startTime = np.round(duration / 3)
    endTime = np.round(duration * 2 / 3)
    frame_dur = frame_duration * sampleFreq
    t1 = startTime
    t2 = t1 + frame_dur
    frame_dur = int(frame_dur)
    t1 = int(t1)
    t2 = int(t2)
    if waveData.shape[0] != 0:
        while t2 < endTime:
            split = waveData[t1:t2]
            t1 = t2
            t2 = t2 + frame_dur
            split = tf.reshape(split, frame_dur)
            split = get_spectrogram(split, frame_dur)
            features.append(split)
            # label.append(labelName)
            dataCount[droneCountDict[labelName]] += 1
            label.append(droneDict[labelName])  # one hot encoding
    return features, label


def create_dataset(train_files):
    '''
    Creates feature dataset and label dataset.
    @param train_files: EagerTensor of file paths.
    @return list of features (ds), list of labels corresponding to feature dataset:
    '''
    i = 0
    features = []
    labels = []
    for x in train_files:
        test_file = tf.io.read_file(x)
        test_audio, sampleRate = tf.audio.decode_wav(contents=test_file)
        x = str(x)
        label = x.split('\\')
        label = label[10]
        test_audio = test_audio[0: test_audio.shape[0] - test_audio.shape[0] % 8000]  # trim to nearest second
        newFeats, newLabs = split_audio(test_audio, label, int(sampleRate), 0.5)
        features.extend(newFeats)
        labels.extend(newLabs)
        i = i + 1

    return features, labels


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


def get_spectrogram(waveform, length):
    # Zero-padding for an audio waveform with less than length samples
    input_len = length
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [length] - tf.shape(waveform),
        dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=128, frame_step=64)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def grabTrainingSamples(n, trainingData):
    IFCount = 0
    matriceCount = 0
    phantomCount = 0
    mavicCount = 0
    noiseCount = 0
    evenTrainingData = []
    evenLabelData = []
    for i in range(len(labels)):
        lab = trainingData[i][1]
        if lab == [1, 0, 0, 0, 0] and IFCount < n:
            IFCount += 1
            evenTrainingData.append(trainingData[i][0])
            evenLabelData.append(lab)
        elif lab == [0, 1, 0, 0, 0] and matriceCount < n:
            matriceCount += 1
            evenTrainingData.append(trainingData[i][0])
            evenLabelData.append(lab)
        elif lab == [0, 0, 1, 0, 0] and phantomCount < n:
            phantomCount += 1
            evenTrainingData.append(trainingData[i][0])
            evenLabelData.append(lab)
        elif lab == [0, 0, 0, 1, 0] and mavicCount < n:
            mavicCount += 1
            evenTrainingData.append(trainingData[i][0])
            evenLabelData.append(lab)
        elif lab == [0, 0, 0, 0, 1] and noiseCount < n:
            noiseCount += 1
            evenTrainingData.append(trainingData[i][0])
            evenLabelData.append(lab)
    return evenTrainingData, evenLabelData


features, labels = create_dataset(train_files)
newSet = list(zip(features, labels))
random.seed()
random.shuffle(newSet)  # Ensure data is mixed together
n = np.min(dataCount)  # Ensure data is symmetric (aka even amounts of training data for all classes)
print(n)
# features, labels = grabTrainingSamples(n, features, labels)
features, labels = grabTrainingSamples(n, newSet)
print(len(features), len(labels))

trainFeatures, testFeatures, trainTargets, testTargets = train_test_split(features, labels, test_size=0.10,
                                                                          random_state=42)
# trainFeatures = np.asarray(trainFeatures)
# trainTargets = np.asarray(trainTargets)
# mean = trainFeatures.mean(axis=0)
# trainFeatures -= mean
# std = trainFeatures.std(axis=0)
# trainFeatures /= std
# trainFeatures = np.reshape(trainFeatures,
#                             (len(trainFeatures), trainFeatures.shape[1:], 1))
# trainFeatures = np.squeeze(trainFeatures)
# testFeatures = np.asarray(testFeatures)
# testTargets = np.asarray(testTargets)
# testFeatures -= mean
# testFeatures /= std
# testFeatures = np.reshape(testFeatures, (len(testFeatures), numFeat, 1))
# testFeatures = np.squeeze(testFeatures)
trainFeatures=tf.convert_to_tensor(trainFeatures)
testFeatures=tf.convert_to_tensor(testFeatures)
norm_layer = tf.keras.layers.Normalization()
#norm_layer.adapt(np.squeeze(trainFeatures))
norm_layer.adapt(trainFeatures)

print('Input shape:', np.shape(trainFeatures)[1:])
model = models.Sequential([
    layers.Input(shape=np.shape(trainFeatures)[1:]),
    # Downsample the input.
    layers.Resizing(32, 32),
    # Normalize.
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(dataCount)),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)
EPOCHS = 10
reduceLR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                 verbose=1, patience=10, mode='auto')
early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4,
                          patience=40, mode='auto')
trainFeatures=np.asarray(trainFeatures)
trainTargets=np.asarray(trainTargets)
#trainTargets=tf.convert_to_tensor(trainTargets)
history = model.fit(
    trainFeatures,
    trainTargets,
    epochs=EPOCHS,
    validation_split=0.2,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)
)
y_pred = model.predict(testFeatures)
y_true = testTargets
test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')
    #callbacks=[reduceLR, early])
    #callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),