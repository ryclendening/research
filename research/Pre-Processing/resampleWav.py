import librosa
import soundfile
import numpy as np
import os
path: str = r"C:\Users\rclendening\researchData\resamplePls_v2"
for x in os.listdir(path):
    data, sr = librosa.load(path+"\\"+x, sr=8000)
    normalized_tone = np.int16((data / data.max()) * 32767)
    soundfile.write("C:\\Users\\rclendening\\researchData\\" + x + ".wav", data, samplerate=sr)
