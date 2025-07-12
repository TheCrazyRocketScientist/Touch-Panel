import os
import csv
import sys
import time
from collections import deque

sys.path.insert(0, r"/home/pi/Touch-Panel/pi/multiread/build")

import argparse
import asyncio
import aiofiles
from datetime import datetime
from aiocsv import AsyncWriter
from adxl import *

# -------------------- ARGUMENT PARSING -------------------- #
parser = argparse.ArgumentParser()

parser.add_argument("-b", "--bus", type=int, help="Sets up I2C Bus for ADXL", default=0)
parser.add_argument("-a", "--addr_sel", type=lambda x: x.lower() == 'true', help="Sets I2C Device Address", required=True)
parser.add_argument("-p", "--int_pin", type=int, help="Sets interrupt pin for ADXL", default=4)
parser.add_argument("-s", "--batch_size", type=int, help="Sets batch size for writing data to disk", default=100)

args = parser.parse_args()

bus = args.bus
addr_sel = args.addr_sel
int_pin = args.int_pin
batch_size = args.batch_size

folder_name = "data"
os.makedirs(folder_name,exist_ok=True)

sensor_id = int(bus) + int(addr_sel)
file_name = f"SENSOR{sensor_id}_data.csv"
file_path = os.path.join(folder_name,file_name)


# -------------------- GLOBAL STATE -------------------- #
buffer = []
reading = []
buffer_lock = asyncio.Lock()
event_loop = None

# Sample rate tracking
last_callback_time = None
sample_intervals = deque(maxlen=100)
sample_rate = 0.0

sensor = ADXL345(bus, addr_sel, int_pin)

# -------------------- DATA LOGGING -------------------- #
async def insert_data():
    global buffer
    while True:
        async with buffer_lock:
            if len(buffer) >= batch_size:
                file_buffer = buffer
                buffer = []

                async with aiofiles.open(file_path, mode="a", newline="") as my_file:
                    writer = AsyncWriter(my_file, dialect="excel")
                    await writer.writerows(file_buffer)
                    await my_file.flush()

        await asyncio.sleep(0.001)

# -------------------- INTERRUPT CALLBACK -------------------- #
def read_sensor():
    global reading, buffer, event_loop
    global last_callback_time, sample_intervals, sample_rate

    now = time.time()

    if last_callback_time is not None:
        dt = now - last_callback_time
        if dt > 0:
            sample_intervals.append(dt)
            sample_rate = 1.0 / (sum(sample_intervals) / len(sample_intervals))
    last_callback_time = now

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    reading = [formatted_time, sensor.x, sensor.y, sensor.z]

    if event_loop:
        event_loop.call_soon_threadsafe(buffer.append, reading)

# -------------------- SAMPLE RATE LOGGING -------------------- #
async def log_sample_rate():
    while True:
        await asyncio.sleep(1.0)
        print(f"[Sensor {sensor_id}] Sample rate: {sample_rate:.2f} Hz")

# -------------------- MAIN ASYNC TASK -------------------- #
async def main():
    await asyncio.gather(insert_data(), log_sample_rate())

# -------------------- SCRIPT ENTRY -------------------- #
if __name__ == "__main__":
    try:
        event_loop = asyncio.get_event_loop()
        asyncio.set_event_loop(event_loop)

        sensor.register_callback(read_sensor)

        if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
            with open(file_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "x", "y", "z"])

        sensor.startup()
        sensor.get_data()

        event_loop.run_until_complete(main())

    except KeyboardInterrupt:
        print("Interrupted, flushing buffer to disk...")
        if buffer:
            with open(file_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(buffer)
        sensor.close()
        print("Sensor closed. Exiting.")
