import os.path
import datetime
import px4tools
import pandas as pd
import pylab as pl
from matplotlib import pyplot as plt
from scipy.signal import butter, lfilter, freqz
from redvox.common.data_window import DataWindow, DataWindowConfig


def importTruthData(file):
    data = pd.read_csv(file)
    return data


def printISOformatAll(data):
    for x in range(0, data.timestamp.size, 1):
        data.time_utc_usec[x] = data.time_utc_usec[x] / 1000000
        dobj = datetime.datetime.utcfromtimestamp(data.time_utc_usec[x])
        print(dobj.isoformat())


def printISOformat(UTC):
    UTC = UTC / 1000000
    dobj = datetime.datetime.utcfromtimestamp(UTC)
    print(dobj.isoformat())


def startOfRecord(data):
    return data.time_utc_usec[0]


def endOfRecord(data):
    return data.time_utc_usec[data.time_utc_usec.size - 1]


# Returns tuple of time (in microseconds), lat,lon, and alt
def testTruthData(data, startTime, endTime):
    start = 0
    end = 0
    startTime = startTime * 10 ** 6
    endTime = endTime * 10 ** 6
    for x in range(1, data.time_utc_usec.size, 1):
        if data.time_utc_usec[x] > startTime > data.time_utc_usec[x - 1]:
            start = x
        if data.time_utc_usec[x] >= endTime > data.time_utc_usec[x - 1]:
            end = x
    return data.time_utc_usec[start:end], data.lat[start:end], data.lon[start:end], data.alt[start:end]


def parseRedVox(data, dataTimeStamps, startTime, endTime):
    start = 0
    end = 0
    startTime = startTime * 10 ** 6
    endTime = endTime * 10 ** 6
    for x in range(1, data.size, 1):
        if dataTimeStamps[x] >= startTime >= dataTimeStamps[x - 1]:
            start = x
        if dataTimeStamps[x] >= endTime >= dataTimeStamps[x - 1]:
            end = x
            break
    return data[start:end], dataTimeStamps[start:end]


def butter_bandpass(lowCut, highCut, fs, order=5):
    nyq = 0.5 * fs
    normal_lowCut = lowCut / nyq
    normal_highCut = highCut / nyq
    b, a = butter(order, [normal_lowCut, normal_highCut], btype='bandpass', analog=False, output='ba')
    return b, a


def butter_bandpass_filter(data, lowCut, highCut, fs, order=5):
    b, a = butter_bandpass(lowCut, highCut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_lowpass(lowCut, fs, order=5):
    nyq = 0.5 * fs
    normal_lowCut = lowCut / nyq
    b, a = butter(order, [normal_lowCut], btype='lowpass', analog=False, output='ba')
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass(highCut, fs, order=5):
    nyq = 0.5 * fs
    normal_highCut = highCut / nyq
    b, a = butter(order, [normal_highCut], btype='highpass', analog=False, output='ba')
    return b, a


def import_redVoxData(file):
    config = DataWindowConfig(
        input_dir=file,
        # TODO Allows starttime but error on endtime, work around using parseRedVox function from flightLogCSV
        # start_datetime=datetime.datetime(2021, 8, 23, 15, 29, 0),
        # end_datetime=datetime.datetime(2021, 8, 23, 16, 0, 0), #Error on endtime
        # station_ids=["1758144238"],
        structured_layout=False)  # Note: needs to be set to false or else returns error
    datawindow = DataWindow(config=config)
    return datawindow


def freqShift(PSDData, label):
    alpha = 0.8
    freqShiftedData = []
    freqShiftedLabels = []
    while alpha <= 1.2:
        newDat = PSDData * alpha
        freqShiftedData.append(newDat)
        freqShiftedLabels.append(label)
        alpha = alpha + 0.02
        plt.plot(newDat)
