from scipy.linalg import sqrtm
import numpy as np
from numpy import *
from scipy.io import wavfile

if __name__ == "__main__":
    samplerate1, data1 = wavfile.read(
        "C:\\Users\\rclendening\\researchData\\EscapeCell_DataWav\\A1\\A1R3P5\\Phone_5-3.wav")
    samplerate2, data2 = wavfile.read(
        "C:\\Users\\rclendening\\researchData\\EscapeCell_DataWav\\A1\\A1R3P5\\Phone_7-2.wav")
    samplerate3, data3 = wavfile.read(
        "C:\\Users\\rclendening\\researchData\\EscapeCell_DataWav\\A1\\A1R3P5\\Phone_1-1.wav")
    samplerate4, data4 = wavfile.read(
        "C:\\Users\\rclendening\\researchData\\EscapeCell_DataWav\\A1\\A1R3P5\\Phone_30.wav")
    data1 = data1[10000:50000]
    data2 = data2[10000:50000]
    data3 = data3[10000:50000]
    data4 = data4[10000:50000]
    xx = np.transpose([data1, data2])
    print(xx.shape)
    # yy= sqrtm(np.linalg.inv(np.cov(np.transpose(xx))))*(xx-np.tile(np.matrix.mean(xx,1),1,np.size(xx,1)))
    # [w, s, v] = np.linalg.svd(np.multiply(np.tile(np.matrix.sum(np.multiply(yy,yy),1),np.size(yy,1),1),yy)*np.transpose(yy))
    #

    # a= w*xx
    u, s, vh = linalg.svd(dot((tile(sum(xx * xx, 0), (xx.shape[0], 1)) * xx), xx.T))
    a = np.multiply(u*xx)
    wavfile.write("C:\\Users\\rclendening\\1.wav", 8000, a[0, :])
    wavfile.write("C:\\Users\\rclendening\\2.wav", 8000, a[1, :])
