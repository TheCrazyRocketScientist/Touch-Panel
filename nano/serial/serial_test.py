import os
import struct
import asyncio
import serial_asyncio
import aiofiles
from datetime import datetime
from aiocsv import AsyncWriter

#batch every 5 seconds
batch_size = 2500

class my_buffer():
    
    def __init__(self,sensor_number):
        
        self.sensor_number = sensor_number
        self.data = []
        self.lock = asyncio.Lock()

buffer_one = my_buffer(1)
buffer_two = my_buffer(2)
buffer_three = my_buffer(3)

buffers = {1:buffer_one,2:buffer_two,3:buffer_three}

async def manage_buffer(buffer):

    global batch_size

    file_path = f"SENSOR{buffer.sensor_number}_data.csv"

    if(os.path.exists(file_path) == False):
         async with buffer.lock:

                async with aiofiles.open(file_path,mode="w",newline="") as my_file:

                    writer = AsyncWriter(my_file, dialect="excel")
                    await writer.writerow(['timestamp,x,y,z'])

    while True:
    
        if(len(buffer.data)) >= batch_size:

            async with buffer.lock:

                async with aiofiles.open(file_path,mode="a",newline="") as my_file:

                    writer = AsyncWriter(my_file, dialect="excel")
                    await writer.writerows(buffer.data)

                    buffer.data.clear()

        await asyncio.sleep(0.01)

async def read_serial(device):

    port = rf"/dev/tty{device}"

    reader,writer = await serial_asyncio.open_serial_connection(
        baudrate = 115200,
        url = port
    )

    while True:

        current_time = datetime.now()
        # Format the date and time using strftime
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        try:
            data = await reader.readexactly(14)
            x,y,z,sensor_num = struct.unpack('<fffh',data)

            buffer = buffers[sensor_num]

            async with buffer.lock:
                buffer.data.append([formatted_time,x,y,z])
                print(f"{x:.3f} {y:.3f} {z:.3f} {sensor_num}")


        except asyncio.IncompleteReadError:
            print("Incomplete Data Received")

        except KeyError:
            print("Buffer Provided Not Available")

        await asyncio.sleep(0.01)

async def main():

    serial_tasks = [read_serial('ACM0'),read_serial('ACM1'),read_serial('ACM2')]
    buffer_tasks = [manage_buffer(buffer_one),manage_buffer(buffer_two),manage_buffer(buffer_three)]
    await asyncio.gather(*serial_tasks,*buffer_tasks)

if __name__ == "__main__":
    asyncio.run(main())

