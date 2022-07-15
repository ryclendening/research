from scipy.io.wavfile import write
import os
import csv
import datetime
import numpy as np
from research.dataImporting import dataTools
#Big picture, can turn redVox to Wav file (used to convert to noise wavs)

rows = []
startTimes = []
endTimes = []
runName = []
vals = []
i = 0


def searchDir(rootdir, start, end):
    it = 0
    for it in os.scandir(rootdir):
        if it.is_dir():
            # print(it.path)
            searchDir(it, start, end)
        else:
            it = os.path.dirname(it)
            break
    if isinstance(it, str):
        splitPath = it.split('\\')
        convertToWav(it, start, end, splitPath[4], splitPath[5], splitPath[6])


# window = dataTools.import_redVoxData("C:\\Users\\rclendening\\EscapeTest_Data\\A2\\A2R2P3\\Phone_1-1\\")
# station = window.first_station()
# samples = station.audio_sensor().get_microphone_data()
# timeStamps = station.audio_sensor().data_timestamps()
# dataSamples, dataTime = dataTools.parseRedVox(samples, timeStamps, 1629817320, 1629817375)
#
# print("")

def convertToWav(path, start, end, scenario, runNum, cellName):
    window = dataTools.import_redVoxData(path)
    station = window.first_station()
    audioSensor = station.audio_sensor()
    samples = audioSensor.get_microphone_data()
    timeStamps = audioSensor.data_timestamps()
    dataSamples, dataTime = dataTools.parseRedVox(samples, timeStamps, start, end)
    if dataSamples.size:
        normalized_tone = np.int16((dataSamples / dataSamples.max()) * 32767)
        write("C:\\Users\\rclendening\\researchData\\" + runNum + cellName + ".wav",
              8000,
              normalized_tone)
    else:
        print("Skipped" + scenario + runNum, cellName)


# file = open("C:\\Users\\rclendening\\bash_scripting\\flightTimeStamps_New1redo.csv")
file = open("C:\\Users\\rclendening\\researchData\\researchCSVs_Scripts_etc\\redVoxNoiseData.csv")
csvreader = csv.reader(file)

for row in csvreader:
    rows.append(row)

for entry in rows:
    run = entry[2]
    scene = entry[1]
    passNum = entry[0]
    start = entry[3].split('_')
    end = entry[4].split('_')
    date = start[0][6:8]
    startTime = start[1]
    endTime = end[1]
    hourStart = int(startTime[0:2])
    minuteStart = int(startTime[2:4])
    secStart = int(startTime[4:5])
    hourEnd = int(endTime[0:2])
    minuteEnd = int(endTime[2:4])
    secEnd = int(endTime[4:6])
    tsStart = datetime.datetime(2021, 8, int(date), hourStart, minuteStart, secStart).timestamp()
    tsEnd = datetime.datetime(2021, 8, int(date), hourEnd, minuteEnd, secEnd).timestamp()
    startTimes.append(tsStart)
    endTimes.append(tsEnd)
    runName.append((scene + 'R' + run + 'P' + passNum).strip())

timeStamps = np.stack((startTimes, endTimes), axis=1)
for r in range(len(timeStamps)):
    start = timeStamps[r][0]
    end = timeStamps[r][1]
    name = runName[r]
    testName = "C:\\Users\\rclendening\\researchData\\cellNoise\\" + name
    searchDir(testName, start, end)

# searchDir("C:\\Users\\rclendening\\EscapeTest_Data\\cellNoise")

#convertToWav("C:\\Users\\rclendening\\researchData\\test123\\", timeStamps[0][0], timeStamps[0][1], "1", "2", "3")
