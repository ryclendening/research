from matplotlib import pyplot as plt

from scipy.io.wavfile import write
import csv
import numpy as np
from research.dataImporting import dataTools


def convertToGPS(path, start, end, scenario, runNum, cellName):
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


def readDirectoryCSV():
    file = open(r"C:\Users\rclendening\researchData\researchCSVs_Scripts_etc\flightDirectoryCSV.csv")
    csvreader = csv.reader(file)
    rows = []
    startTimes = []
    endTimes = []
    fileNames = []
    runNames = []
    for row in csvreader:
        rows.append(row)

    for entry in rows:
        run = entry[2]
        scene = entry[1]
        passNum = entry[0]
        start = entry[3].split('_')
        end = entry[5].split('_')
        date = start[0][6:8]
        startTime = start[1]
        endTime = end[1]
        fileName = entry[7]
        startTimes.append(startTime)
        endTimes.append(endTime)
        fileNames.append(fileName)
        runNames.append((scene + 'R' + run + 'P' + passNum).strip())
    return [runNames, startTimes, endTimes, fileNames]


def getDroneGPSData(fileName, start, end):
    file = open(
        "C:\\Users\\rclendening\\researchData\\ESCAPE II_AFRL_SSD\\UAS Campaign\\UAS_log_files\\A3\\" + fileName + ".csv")
    csvreader = csv.reader(file)
    rows = []
    GPSLat = []
    GPSLon = []
    timeStamps = []
    for row in csvreader:
        rows.append(row)

    for entry in rows[3:]:  # time is column 7, long is column 4, lat is column 5

        time = entry[7]
        time = int(time) - 40000
        if int(start) <= time < int(end):
            GPSLon.append(float(entry[4]))
            GPSLat.append(float(entry[5]))
            timeStamps.append(time)
    return {"lat": GPSLat, "long": GPSLon, "time": timeStamps}


flightDetails = readDirectoryCSV()
x = 0
GPSCords = {}
for x in range(len(flightDetails[0])):
    GPSCords[flightDetails[0][x]] = (
        getDroneGPSData(fileName=flightDetails[3][x], start=flightDetails[1][x], end=flightDetails[2][x]))

BBox = [-75.4300, -75.4036, 43.2257, 43.2435]
mymap = plt.imread(r"C:\Users\rclendening\Downloads\map2.png")
fig, ax = plt.subplots(figsize=(8, 7))
ax.scatter(x=np.asarray(GPSCords["A3R3P4"]["long"]), y=np.asarray(GPSCords["A3R3P4"]["lat"]), zorder=1, alpha=0.2, c='b',
           s=10)
ax.set_title('Plotting Spatial Data on Escape II Flight Path')
ax.set_xlim(BBox[0], BBox[1])
ax.set_ylim(BBox[2], BBox[3])
ax.imshow(mymap, zorder=0, extent=BBox, aspect='equal')
plt.show()