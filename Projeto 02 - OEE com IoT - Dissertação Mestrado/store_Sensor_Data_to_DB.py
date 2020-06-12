#------------------------------------------
#--- Author: Jaime Leite
#--- Date: 11th June 2020
#--- Version: 1.0
#--- Python Ver: 2.7
#--- Details At: 
#------------------------------------------

import json
import sqlite3

# SQLite DB Name
DB_Name =  "OEE.db"

#===============================================================
# Database Manager Class

class DatabaseManager():
	def __init__(self):
		self.conn = sqlite3.connect(DB_Name)
		self.conn.execute('pragma foreign_keys = on')
		self.conn.commit()
		self.cur = self.conn.cursor()
		
	def add_del_update_db_record(self, sql_query, args=()):
		self.cur.execute(sql_query, args)
		self.conn.commit()
		return

	def __del__(self):
		self.cur.close()
		self.conn.close()

#===============================================================
# Functions to push Sensor Data into Database

# Function to save Temperature to DB Table
def OEE_Data_Handler(jsonData):
	#Parse Data 
	json_Dict = json.loads(jsonData)
	SensorID  = "WISE-4050" #json_Dict['Sensor_ID']
	Data_and_Time = json_Dict['t']
	if json_Dict['di1']:
		SensorStatus = "Online"
	else:
		SensorStatus = "Downtime"
	Production = json_Dict['di2']
	Quality    = json_Dict['di3']
	#Push into DB Table
	dbObj = DatabaseManager()
	dbObj.add_del_update_db_record("insert into OEE_Data (SensorID, Date_n_Time, SensorStatus, Production, Quality) values (?,?,?,?,?)",[SensorID, Data_and_Time, SensorStatus, Production, Quality])
	del dbObj
	print("Inserted Telemetry Data into Database. "+ Date_n_Time + " P: "+str(Production)+ " Q: " + str(Quality))
	print("")

# Function to save Humidity to DB Table
def OEE_Status_Handler(jsonData):
	#Parse Data 
	json_Dict = json.loads(jsonData)
	SensorID = "WISE-4050" #json_Dict['Sensor_ID']
	Data_and_Time = json_Dict['t']
	OEE_Status = json_Dict['OEE_Status']
	#Push into DB Table
	dbObj = DatabaseManager()
	dbObj.add_del_update_db_record("insert into OEE_Status (SensorID, Date_n_Time, OEE_Status) values (?,?,?)",[SensorID, Data_and_Time, OEE_Status])
	del dbObj
	print("Device_Status LOG inserted into Database.")
	print("")


#===============================================================
# Master Function to Select DB Funtion based on MQTT Topic

def sensor_Data_Handler(Topic, jsonData):
	if Topic == "Advantech/00D0C9E38B74/data":
		OEE_Data_Handler(jsonData)
	elif Topic == "Advantech/00D0C9E38B74/Device_Status":
		OEE_Status_Handler(jsonData)	
	
#===============================================================