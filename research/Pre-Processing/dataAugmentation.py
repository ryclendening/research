import pathlib

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile
from librosa.effects import pitch_shift
from scipy.io import wavfile
import scipy.signal
from scipy.signal import welch
from scipy.fft import rfft, rfftfreq
import tensorflow as tf


# def pitch_shifter(data):
#     return pitch_shift(data, sr=8000, n_steps=0)
#
#
# def PSD_log(waveform, length):
#     input_len = length
#     waveform = waveform[:input_len]
#     logPSD = []
#     f_dum, PSD = welch(waveform, fs=8000, window='hann',
#                        nperseg=input_len, average='mean')
#     logPSD = np.log(PSD + np.finfo(float).eps)
#     plt.show()
#     # logPSD.append(np.log(PSD + np.finfo(float).eps))
#     return PSD
#
#
# def freqPlot(dat, len):
#     yf = rfft(dat)
#     xf = rfftfreq(len, 1 / 8000)  # 1/8000 =sample period
#     plt.plot(xf, np.abs(yf))
#     plt.ion()
#     plt.xlabel(f"Frequency (Hz)")
#     plt.ylabel(f"Amplitude")
#
#
# samplerate, data = wavfile.read(
#     "C:\\Users\\rclendening\\researchData\\high_fidel_NM_RS2\\Matrice_600\\d303sA1r02p0120210824.wav")
# signalDat, sr = librosa.load(
#     r"C:\Users\rclendening\researchData\high_fidel_NM_RS2\Phantom_4_Pro_V2\d301sA2r05p0120210826_1.wav", sr=None)
# shiftedTone = pitch_shifter(signalDat)
# soundfile.write("C:\\Users\\rclendening\\researchData\\test.wav",
#                 shiftedTone, sr, subtype='PCM_16')
# # normTone = PSD_log(signalDat, 8000)
# # shifted = PSD_log(shiftedTone, 8000)
# # # plt.semilogy(shifted, color='b')
# # # plt.semilogy(normTone, color='r')
# freqPlot(signalDat[13000:15000],2000)
# freqPlot(shiftedTone[13000:15000],2000)
# plt.show()
# for x in range(-2,3,1):
#     print(x)

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
        if sampleRate != 8000:
            break;
        x = str(x)
        label = x.split('\\')
        label = label[10]
        for c in range(-2, 3, 1):
            new_data = pitchShifter(data=test_audio.numpy().squeeze(), sr=sampleRate.numpy(), n_steps=c)
            soundfile.write("C:\\Users\\rclendening\\researchData\\high_fidel_NM_RS_PS\\" + label + '\\' + str(i)+str(c)+".wav",
                            new_data, sampleRate, subtype='PCM_16')
        i = i + 1
        print(i)


def pitchShifter(data, sr, n_steps):
    return pitch_shift(data, sr=sr, n_steps=n_steps)


dataset_path: str = "C:\\Users\\rclendening\\researchData\\high_fidel_NM_RS"
data_dir = pathlib.Path(dataset_path)
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
train_files = filenames
create_dataset(train_files)
