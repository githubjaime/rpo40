#------------------------------------------
#--- Author: Jaime Leite
#--- Date: 11th June 2020
#--- Version: 1.0
#--- Python Ver: 2.7
#--- Details At: 
#------------------------------------------

import paho.mqtt.client as mqtt
from store_Sensor_Data_to_DB import sensor_Data_Handler

# MQTT Settings 
MQTT_Broker = "tailor.cloudmqtt.com"
MQTT_Port = 12836
Keep_Alive_Interval = 45
MQTT_Topic = "Advantech/00D0C9E38B74/#" 
#MQTT_Topic = "Home/BedRoom/#"
cUSER = "kvnfxwil"
cPASS = "IufLKbam_lJZ"

#Subscribe to all Sensors at Base Topic
def on_connect(mosq, obj, rc):
	if rc != 0:
		pass
		print("Unable to connect to MQTT Broker...")
				
	else:
		print("Connected with MQTT Broker: " + str(MQTT_Broker))
				
	mqttc.subscribe(MQTT_Topic, 0)

#Save Data into DB Table
def on_message(mosq, obj, msg):
	# This is the Master Call for saving MQTT Data into DB
	# For details of "sensor_Data_Handler" function please refer "sensor_data_to_db.py"
	print("Inputing data from Topic " + msg.topic + str(msg.payload))
	#print("")
	#print("MQTT Data: " + str(msg.payload))
	sensor_Data_Handler(msg.topic, msg.payload)

def on_subscribe(mosq, obj, mid, granted_qos):
    pass

mqttc = mqtt.Client("SQLite")

# Assign event callbacks
mqttc.on_message = on_message
mqttc.on_connect = on_connect
mqttc.on_subscribe = on_subscribe
mqttc.username_pw_set(cUSER, password=cPASS)
# Connect
mqttc.connect(MQTT_Broker, int(MQTT_Port), int(Keep_Alive_Interval))
mqttc.subscribe(MQTT_Topic)

# Continue the network loop
#mqttc.loop_forever()

cur = True
while cur:
	mqttc.loop()