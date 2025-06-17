import os
import csv
import struct
import asyncio
import aiofiles
import argparse
import json
from bleak import BleakClient,BleakScanner
from datetime import datetime
from aiocsv import AsyncWriter

"""
Methods to investigate:

1.) Round Robin Buffer, each sensor fills a temporary buffer once, which is then appended to a file buffer
    which is then batch inserted to disk
2.) Send sample number + data from arduino 
3.) Use dask for storing values which may be large for in memory
4.) IMPORTANT!!!! : Number of diverging samples could be caused by differing sample rates.
    Graph number of samples vs time to verify this.
5.) Test performance with and without asyncio writes to disk, and single thread vs multiprocess
6.) Create a single threaded version of this script for archival/testing purposes.

"""


#UUID for custom characteristic used by the arduino
ACCEL_DATA_UUID =  "00002101-0000-1000-8000-00805f9b34fb".lower()

#setting up arguments for CLI
parser = argparse.ArgumentParser()

parser.add_argument("-n","--name",type=str,help="Sets sensor to be read from.",default="SENSOR0")
#parser.add_argument("-t","--scan_time",type=int,help="Sets duration of BLE scan.",default=1)
parser.add_argument("-d","--duration",type=int,help="Gets data from sensors for a set duration",default=0)
parser.add_argument("-b","--batch_size",type=int,help="Sets max size of buffer before inserting",default=100)
parser.add_argument("-a","--address",type = str,help="Sets device address.",default="")

#parsing user provided args
args = parser.parse_args()

name = args.name
#scan_time = args.scan_time
duration = args.duration
batch_size = args.batch_size
address = args.address

#to get index, use this: index = name[-1]

#setting up list for sensor buffer
buffer = []
#buffer = defaultdict(list)

async def insert_data():

    global buffer
    
    while True:
     
        if(len(buffer)) >= batch_size:

            async with aiofiles.open(f"{name}_data.csv",mode="a",newline="") as my_file:

                writer = AsyncWriter(my_file, dialect="excel")
                await writer.writerows(buffer)

                await my_file.flush()

                buffer = []

        await asyncio.sleep(0.1)
            


def get_data(sender,data):

    current_time = datetime.now()
    # Format the date and time using strftime
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    #callback function to unpack bytes from sensor
    x, y, z, index = struct.unpack('<fffh', data)
    buffer.append([formatted_time,x,y,z])
    #print([formatted_time,x,y,z])

    #when defaultdict was used
    #buffer[index].append(append([formatted_time,x,y,z]))

    #test for lag
    #print("buffer0 {len(buffer[1])} buffer1 {len(buffer[1])}")
            


async def handle_sensor(name,address):
    
    async with BleakClient(address) as client:
        
        #check if client specified is still connected
        if client.is_connected:

            #allocating time to settle down
            await asyncio.sleep(0.5)

            #forcing service discovery on device
            services = client.services
            
            #allocating time to settle down
            #await asyncio.sleep(0.5)

            #enabling notifications to come through
            await client.start_notify(ACCEL_DATA_UUID,get_data)

            #loop to read the sensor, if duration is set to 0
            if(duration == 0):
                while True:
                    await asyncio.sleep(0.5)
                    pass
            else:
                #set custom duration for sensor data read
                await asyncio.sleep(duration)
        else:

            print(f"Connection lost with device {name}")

         #terminate the notification    
        await client.stop_notify(ACCEL_DATA_UUID)

    #exiting the context manager terminates the connection

async def start_scan(scan_time):

    #functon to discover all BLE devices 

    #set up async conext manager to accomodate bleak.BleakScanner() class
    async with BleakScanner() as scanner:

        #sleep for a set time, this causes the event loop to suspend main for a given amount of time, and till then scanning continues.
        await asyncio.sleep(scan_time)
        
    #create a dictionary to store the result    
    return {device.name:device.address for device in scanner.discovered_devices if device.name != None }

async def main():
    
    print("Writing to:", os.getcwd())


    #main function to handle all calls to other coroutines

    #call the scan function and get a dict containing all devices names and addresses
    #address_dict = await start_scan(scan_time)

    #create a list of all coroutines with necessary params using address_dict
    #dev_list = [handle_sensor(dev_name,dev_address) for (dev_name,dev_address) in address_dict.items()]

    #gather all couroutines and execute concurrently
    #await asyncio.gather(*dev_list)    

    task_list = [handle_sensor(name,address),insert_data()]
    await asyncio.gather(*task_list)

try:

    #submit main() to async event loop
    asyncio.run(main())

except KeyboardInterrupt:

    #handle any keyboard interrupts
    print("Ending connection")

