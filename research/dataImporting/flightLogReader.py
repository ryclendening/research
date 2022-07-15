import os.path
from datetime import datetime
import px4tools
import pandas as pd

def import_ulog(file, LoggerID=10001):
    data=px4tools.read_ulog(file)
    data=data["t_vehicle_gps_pos_0"].reset_index(drop=True)
    imported=pd.dataFrame()
    imported["Measurement_DateTime"]=pd.to_datetime(list(map(lambda x: datetime.utcfromtimestamp(x/1E+6).isoformat(),
                                                             data["f_time_utc_usec"])))
    imported["GPS_Lat"]=data["f_lat"]/10000000
    imported["GPS_Lon"] = data["f_lon"] / 10000000
    imported["GPS_Lat"] = data["f_lat"] / 1000
    imported["User_LoggerID"]= LoggerID
    return imported

#import_ulog(r"C:\Users\rclendening\PycharmProjects\MLTesting\research\truthData\log_15_2021-8-23-15-11-08_vehicle_gps_position_0.csv")