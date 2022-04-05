from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.fft import rfft, rfftfreq
from dataImporting import dataTools
from pydub import AudioSegment
import numpy as np
import os
import soundfile
import librosa
filePath: str = "E:\Training_Data\Mavic_Pro\MAVICPRO-FT-A-5871_20191126.wav"


def FFT(data, samplerate):
    N = data.size
    yf = rfft(data)
    xf = rfftfreq(N, 1 / samplerate)  # 1/8000 =sample period
    plt.plot(xf, np.abs(yf))
    plt.ion()
    plt.title(f"YOLO")
    plt.xlabel(f"Frequency (Hz)")
    plt.ylabel(f"Amplitude")
    plt.show()


def split_audio(waveFile, filename, droneName):
    duration = waveFile.duration_seconds
    t1 = 1000
    t2 = 1200
    i = 0
    while t2 < (duration * 1000) - 200:
        split_filename = str(i) + filename
        split = waveFile[t1:t2]
        split.export("E:\\Training_Data_parsed\\" + droneName + "\\" + split_filename + ".wav", format="wav")
        t1 = t2
        t2 = t2 + 200
        i = i + 1
        # samplerate, data = wavfile.read(
        #     "E:\\Training_Data_parsed\\" + droneName + split_filename + ".wav")
        # FFT(data, samplerate)


def reduce_audio(dataPath, fileName, droneName):
    resample = 8000
    newdat, s = librosa.load(dataPath, sr=resample)
    normalized_tone = np.int16((newdat / newdat.max()) * 32767)
    soundfile.write("C:\\Users\\rclendening\\researchData\\Training_Data_NM_RS\\" + droneName + "\\" + fileName, normalized_tone, resample, subtype='PCM_16')


# waveFile = AudioSegment.from_wav(filePath)
# split_audio(waveFile, 200, "E:\\Training_Data\\Mavic_Pro\\", "MAVICPRO-FT-A-5871_20191126")
directories = os.listdir("E:\\Training_Data\\")
i = 0
# while i < 4:
#     path = "E:\\Training_Data\\" + directories[i]
#     files = os.listdir(path)
#     for x in files:
#         waveFile = AudioSegment.from_wav(path + "\\" + x)
#         split_audio(waveFile, x, directories[i])
#     i = i + 1
while i < len(directories):
    path = "E:\\Training_Data\\" + directories[i]
    files = os.listdir(path)
    for x in files:
        #data, samplerate = soundfile.read(path + "\\" + x)
        reduce_audio(path + "\\" + x, x,directories[i])
    i = i + 1
