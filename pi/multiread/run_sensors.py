import os
import csv
import adxl
import argparse
import asyncio
import aiofiles
from datetime import datetime
from aiocsv import AsyncWriter


parser = argparse.ArgumentParser()

parser.add_argument("-n","--sensor_num",type=int,help="Sets SPI Device Number",default=0)
parser.add_argument("-p","--int_pin",type=int,help="Sets interrupt pin for ADXL",default=4)
parser.add_argument("-b","--batch_size",type=int,help="Sets batch size for writing data to disk",default=100)

args = parser.parse_args()

sensor_num = args.sensor_num
batch_size = args.batch_size
int_pin = args.int_pin


file_name = f"SENSOR{sensor_num}_data.csv"

buffer = []
buffer_lock = asyncio.Lock()


sensor = ADXL345(sensor_num,int_pin)

async def insert_data():

    global buffer
    
    while True:
        
        if(len(buffer)) >= batch_size:

            async with buffer_lock:
                file_buffer = buffer
                buffer = []

            async with aiofiles.open(file_name,mode="a",newline="") as my_file:

                writer = AsyncWriter(my_file, dialect="excel")
                await writer.writerows(file_buffer)

                await my_file.flush()


        await asyncio.sleep(0.001)
            
async def read_sensor():

    while True:
        async with buffer_lock:
            current_time = datetime.now()
            # Format the date and time using strftime
            formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            buffer.append([formatted_time,sensor.x,sensor.y,sensor.z])
            #print([formatted_time,x,y,z])

        await asyncio.sleep(0.005)


async def main():

    tasks = [insert_data(),read_sensor()]
    await asyncio.gather(*tasks)

if __name__ == "__main__":

    try:

        if not os.path.exists(file_name) or os.stat(file_name).st_size == 0:
        # Write header with standard csv.writer (sync I/O)
            with open(file_name, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "x", "y", "z"])

        sensor.startup()
        sensor.get_data()
        asyncio.run(main())

    except KeyboardInterrupt:
        sensor.close()
        print("Interrupted, Ending Process")



    