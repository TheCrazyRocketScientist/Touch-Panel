import os
import csv
import sys

sys.path.insert(0,r"/home/pi/Touch-Panel/pi/multiread/build")

import argparse
import asyncio
import aiofiles
from datetime import datetime
from aiocsv import AsyncWriter
from adxl import *


parser = argparse.ArgumentParser()

parser.add_argument("-b","--bus",type=int,help="Sets up I2C Bus for ADXL",default=0)
parser.add_argument("-a","--addr_sel",type=lambda x: x.lower() == 'true',help="Sets I2C Device Number",default=0,required=True)
parser.add_argument("-p","--int_pin",type=int,help="Sets interrupt pin for ADXL",default=4)
parser.add_argument("-s","--batch_size",type=int,help="Sets batch size for writing data to disk",default=100)

args = parser.parse_args()

address_select = args.addr_sel
batch_size = args.batch_size
int_pin = args.int_pin
bus = args.bus


folder_name = "data"
os.makedirs(folder_name,exist_ok=True)


file_name = f"SENSOR{int(bus)+int(address_select)}_data.csv"
file_path = os.path.join(folder_name,file_name)


buffer = []
reading = []
val_buffer = []
buffer_lock = asyncio.Lock()

event_loop = None

sensor = ADXL345(bus,address_select,int_pin)

async def insert_data():

    global buffer
    
    while True:
        
        async with buffer_lock:

            if(len(buffer)) >= batch_size:

                file_buffer = buffer
                buffer = []

                async with aiofiles.open(file_path,mode="a",newline="") as my_file:

                    writer = AsyncWriter(my_file, dialect="excel")
                    await writer.writerows(file_buffer)

                    await my_file.flush()


        await asyncio.sleep(0.001)


def read_sensor():

    global reading
    global buffer

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    #does not work
    
    event_loop.call_soon_threadsafe(buffer.append,[formatted_time,sensor.x,sensor.y,sensor.z])

    #possible race condition
    #buffer.append([formatted_time,sensor.x,sensor.y,sensor.z])


async def main():

    tasks = [insert_data()]
    await asyncio.gather(*tasks)

if __name__ == "__main__":

    try:

        event_loop = asyncio.get_event_loop()
        asyncio.set_event_loop(event_loop)

        sensor.register_callback(read_sensor)

        if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        # Write header with standard csv.writer (sync I/O)
            with open(file_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "x", "y", "z"])

        sensor.startup()
        sensor.get_data()

        #replace with lower level coro execution
        #asyncio.run(main())

        event_loop.run_until_complete(main())


    except KeyboardInterrupt:

        print("Interrupted, flushing buffer to disk...")
        if buffer:
            with open(file_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(buffer)
        sensor.close()
        print("Sensor closed. Exiting.")