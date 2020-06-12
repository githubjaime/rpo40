#------------------------------------------
#--- Author: Pradeep Singh
#--- Date: 20th January 2017
#--- Version: 1.0
#--- Python Ver: 2.7
#--- Details At: https://iotbytes.wordpress.com/store-mqtt-data-from-sensors-into-sql-database/
#------------------------------------------

import sqlite3

# SQLite DB Name
DB_Name =  "OEE.db"

# SQLite DB Table Schema
TableSchema="""
drop table if exists OEE_Data ;
create table OEE_Data (
  id integer primary key autoincrement,
  SensorID text,
  Date_n_Time text,
  SensorStatus text,
  Production text,
  Quality    text,
  Reason     text
);


drop table if exists OEE_Device_Status ;
create table OEE_Status (
  id integer primary key autoincrement,
  SensorID text,
  Date_n_Time text,
  OEE_Status  text
);
"""

#Connect or Create DB File
conn = sqlite3.connect(DB_Name)
curs = conn.cursor()

#Create Tables
sqlite3.complete_statement(TableSchema)
curs.executescript(TableSchema)

#Close DB
curs.close()
conn.close()
