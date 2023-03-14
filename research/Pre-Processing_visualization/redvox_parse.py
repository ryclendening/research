import sys
import re
import shutil
import os
import csv
import datetime
import numpy as np

rows = []
startTimes = []
endTimes = []
runName = []
vals = []
i = 0


# def searchDir(rootdir, start, end, name):
def searchDir(rootdir, start, end, name):
    it = 0
    for it in os.scandir(rootdir):
        if it.is_dir():
            # print(it.path)
            searchDir(it, start, end, name)
        else:
            it = os.path.dirname(it)
            break
    if isinstance(it, str):
        parseEvent(it, start, end, name)


# def parseEvent(path, start, end, name):
def parseEvent(path, start, end, name):
    testStart = start * 1000000
    testEnd = end * 1000000
    fileList = os.listdir(path)
    fileList.sort()
    print(fileList)
    for i in range(len(os.listdir(path)) - 1):
        split = fileList[i].split('_')
        split = split[1].split('.')
        val = int(split[0])
        split = fileList[i + 1].split('_')
        split = split[1].split('.')
        nextVal = int(split[0])
        if val < testStart < nextVal:
            print(path)
            print(val)
            phoneName = path.split('\\')
            phone = phoneName[8]
            if not phone.startswith('Phone'):
                phone = phoneName[7]
            newDir = ("C:\\Users\\rclendening\\researchData\\cellNoise_v2\\" + name + "\\" + phone).strip()
            if not (os.path.exists(newDir) and os.path.isdir(path)):
                os.mkdir(newDir)
                j = i
                while val < testEnd and j < len(os.listdir(path)) - 1:
                    split = fileList[j].split('_')
                    split = split[1].split('.')
                    val = int(split[0])
                    # currLoc = (path + "/" + str(fileList[j])).strip()
                    currLoc = (path + "\\" + str(fileList[j])).strip()
                    # newLoc = ((newDir) + "/" + str(fileList[j])).strip()
                    print(currLoc)
                    newLoc = newDir
                    shutil.copy(currLoc, newLoc)
                    j += 1
                break
            # searchDir("/home/ruan/Documents/bash_scripting/Position_1")


# file = open("/home/ruan/Downloads/Book1.csv")
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
    # newDir = ("/media/sf_VM_Folder/EscapeTest_Data/" + name).strip()
    os.mkdir("C:\\Users\\rclendening\\researchData\\cellNoise_v2\\"+name)
    # searchDir("C:\\Users\\rclendening\\researchData\\ESCAPE_August_fresh", start, end, name, newDir)
    searchDir("C:\\Users\\rclendening\\researchData\\ESCAPE_August_fresh", start, end, name)
