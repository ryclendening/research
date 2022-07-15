import datetime
from matplotlib import pyplot as plt
import dataTools
import soundFrame

filePath: str = "C:\\Users\\rclendening\\EscapeTest_Data\\A2\\A2R7P1\\Phone_5-4\\54_1630002051410059.rdvxm"
# packet: WrappedRedvoxPacketM = WrappedRedvoxPacketM.from_compressed_path(filePath)
# station_information: 'StationInformation' = packet.get_station_information()
# input_dir: str = "C:\\Users\\rclendening\\PycharmProjects\\MLTesting\\research\\Phone_5-2(A2R7P1)"
input_dirMavic: str = r"C:\Users\rclendening\researchData\Unused_Datasets\EscapeCell_Data\A3\A3R5P3\Phone_5-4\54_1630078955065218.rdvxm"
#input_dirMatrice: str = "C:\\Users\\rclendening\\EscapeTest_Data\\A2\\A2R3P1\\Phone_3-2"
# for Windows, delete the above line and use the line below instead:
# input_dir: str = "C:\\path\\to\\api_dir
# truthDataFile = r"C:\Users\rclendening\PycharmProjects\MLTesting\research\truthData\log_16_2021-8-23-15-45" \
# r"-16_vehicle_gps_position_0.csv"

# truthData = flightLogCSV.importTruthData(truthDataFile)
start_Mav = datetime.datetime(2021, 8, 26, 14, 21, 49, 000000).timestamp()
end_Mav = datetime.datetime(2021, 8, 26, 14, 21, 49, 300000).timestamp()

#start_Mat = datetime.datetime(2021, 8, 24, 13, 25, 50, 000000).timestamp()
#end_Mat = datetime.datetime(2021, 8, 24, 13, 25, 50, 300000).timestamp()
if __name__ == "__main__":
    datawindowMav = dataTools.import_redVoxData(filePath)
    stationMav = datawindowMav.first_station()
    samplesMav = stationMav.audio_sensor().get_microphone_data()
    timeStampsMav = stationMav.audio_sensor().data_timestamps()
    order = 5
    fs = 8000
    cutoff = 800
    testSamplesMav, testTime = dataTools.parseRedVox(samplesMav, timeStampsMav, start_Mav, end_Mav)
    newData = dataTools.butter_bandpass_filter(testSamplesMav, 50, 3000, fs, order)
    mavPhone = soundFrame.SoundFrame(newData, testTime, "DJI Mavic Pro (A2R7P1)", start_Mav, end_Mav)
    mavPhone.spectPlot()
    mavPhone.freqPlot()

    #datawindowMat = dataTools.import_redVoxData(input_dirMatrice)
    #stationMat = datawindowMat.first_station()
    #samplesMat = stationMav.audio_sensor().get_microphone_data()
    #timeStampsMat = stationMat.audio_sensor().data_timestamps()
    # order = 5
    # fs = 8000
    # cutoff = 800
    # testSamplesMat, testTime = dataTools.parseRedVox(samplesMat, timeStampsMat, start_Mat, end_Mat)
    # newData = dataTools.butter_bandpass_filter(testSamplesMat,50, 3000, fs, order)
    # matPhone = soundFrame.SoundFrame(newData, testTime, "DJI Matrice (A2R3P1)", start_Mat, end_Mat)
    # matPhone.freqPlot()
    # mavPhone.freqPlot()
    plt.title('Mavic Drone Acoustics')
    plt.show()
    # normalized_tone = np.int16((testSamplesMat / testSamplesMat.max()) * 32767)
    # write("C:\\Users\\rclendening\\EscapeTest_DataWav\\Test1Audio.wav", 8000, normalized_tone)
