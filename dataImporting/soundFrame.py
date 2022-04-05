import numpy as np
from scipy.fft import rfft, rfftfreq
from matplotlib import pyplot as plt
from redvox.common.data_window import DataWindow, DataWindowConfig


class SoundFrame:
    def __init__(self, data, timeStamps, phoneName, start, end):
        self.dat = data
        self.timeStamps = timeStamps
        self.startTime = start
        self.endTime = end
        self.phoneName = phoneName

    # config = DataWindowConfig(
    #     input_dir=input_dir,
    #     # start_datetime=datetime.datetime(2021, 8, 23, 15, 29, 0),
    #     # end_datetime=datetime.datetime(2021, 8, 23, 16, 0, 0), #Error on endtime
    #     # station_ids=["1758144238"],
    #     structured_layout=False)  # Note: needs to be set to false or else returns error
    # datawindow = DataWindow(event_name="yolo", config=config)
    #
    # station = datawindow.first_station()
    def freqPlot(self):
        N = self.timeStamps.size
        yf = rfft(self.dat)
        xf = rfftfreq(N, 1 / 8000)  # 1/8000 =sample period
        plt.plot(xf, np.abs(yf))
        plt.ion()
        plt.title(f"{self.phoneName}")
        plt.xlabel(f"Frequency (Hz)")
        plt.ylabel(f"Amplitude")

    def timePlot(self):
        plt.plot(self.timeStamps, self.dat)
        plt.title(f"{self.phoneName} - audio data")
        plt.xlabel(f"Time")
        plt.ylabel(f"Amplitude")
        plt.show()

    def spectPlot(self):
        plt.specgram(self.dat, Fs=8000, cmap=plt.get_cmap('magma'))
        plt.ylabel('Freq [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
