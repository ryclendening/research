import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

#Fs, samples = wavfile.read(r"C:\Users\rclendening\researchData\RedVox_TrainingBinary_wYTVids\Noise\yt_BMWE30.wav.wav")
Fs, samples = wavfile.read(r"C:\Users\rclendening\researchData\RedVox_TrainingBinary_wYTVids\Noise\yt_StMaarten.wav.wav")

#frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(samples[86000:90000], Fs=Fs)
#plt.plot(samples[100000:104000])
#plt.title("Time Domain of IF1200 Drone")
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.savefig(".\\YouTube.pdf")
plt.show()