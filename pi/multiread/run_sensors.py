import os
import csv
import sys
import argparse
import asyncio
import aiofiles
from datetime import datetime
from aiocsv import AsyncWriter

# --- NEW: Import the pigpio Python library ---
import pigpio
# --- END NEW ---

# Adjust this path if your 'adxl' module is located elsewhere
sys.path.insert(0, r"/home/pi/Touch-Panel/pi/multiread/build")

# --- NEW: Import your C++ ADXL binding module ---
# Assuming 'adxl' here is the name of your compiled pybind11 module (e.g., adxl.so)
try:
    from adxl import ADXL345
except ImportError as e:
    print(f"Error importing ADXL345 from adxl module: {e}")
    print("Please ensure your C++ module (e.g., adxl.so) is correctly built and in the specified path.")
    sys.exit(1)
# --- END NEW ---


parser = argparse.ArgumentParser()

parser.add_argument("-b", "--bus", type=int, help="Sets up I2C Bus for ADXL", default=0)
# Changed default for addr_sel to False based on common ADXL345 default address
parser.add_argument("-a", "--addr_sel", type=lambda x: x.lower() == 'true', help="Sets I2C Device Number", default='false', required=True)
parser.add_argument("-p", "--int_pin", type=int, help="Sets interrupt pin for ADXL", default=4)
parser.add_argument("-s", "--batch_size", type=int, help="Sets batch size for writing data to disk", default=100)

args = parser.parse_args()

address_select = args.addr_sel
batch_size = args.batch_size
int_pin = args.int_pin
bus = args.bus

folder_name = "data"
os.makedirs(folder_name, exist_ok=True)

file_name = f"SENSOR{int(bus) + (1 if address_select else 0)}_data.csv" # Adjusted filename to reflect address_select
file_path = os.path.join(folder_name, file_name)

buffer = []
# reading = [] # This variable 'reading' is not used, can be removed
# val_buffer = [] # This variable 'val_buffer' is not used, can be removed
buffer_lock = asyncio.Lock()

# --- NEW: Global variable for pigpio.pi() instance ---
pi_instance = None
# --- END NEW ---

# --- Removed global event_loop = None, it will be handled in main ---


async def insert_data():
    global buffer

    while True:
        async with buffer_lock:
            if len(buffer) >= batch_size:
                file_buffer = buffer
                buffer = [] # Clear the buffer ONLY after copying it

                async with aiofiles.open(file_path, mode="a", newline="") as my_file:
                    writer = AsyncWriter(my_file, dialect="excel")
                    await writer.writerows(file_buffer)
                    await my_file.flush() # Ensure data is written to disk

        await asyncio.sleep(0.001)


def read_sensor():
    global buffer
    # The event_loop is now obtained correctly in main, so this should work without a global variable
    # Ensure this callback is only added after the event loop is ready

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    # No change here, this line correctly schedules the append on the main event loop
    asyncio.get_event_loop().call_soon_threadsafe(buffer.append, [formatted_time, sensor.x, sensor.y, sensor.z])


async def main():
    global pi_instance
    global sensor # Make sensor global within main if it needs to be accessed
                  # It's already global in the script, so this is just for clarity within async functions.

    # --- NEW: Connect to pigpiod here, once per process ---
    pi_instance = pigpio.pi()
    if not pi_instance.connected:
        print(f"PID {os.getpid()}: Could not connect to pigpiod! Please ensure it is running.")
        # Exit this process if pigpiod connection fails
        os._exit(1) # Use os._exit for immediate termination of the child process

    print(f"PID {os.getpid()}: Successfully connected to pigpiod.")
    # --- END NEW ---

    # Instantiate your C++ ADXL345 object
    # The C++ constructor should *not* call gpioInitialise()
    sensor = ADXL345(bus, address_select, int_pin)

    try:
        if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
            # Write header with standard csv.writer (sync I/O)
            with open(file_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "x", "y", "z"])

        sensor.register_callback(read_sensor) # Register callback AFTER sensor is initialized
        print(f"PID {os.getpid()}: ADXL345 sensor initialized and callback registered.")

        sensor.startup()
        sensor.get_data() # This call sets up the GPIO interrupt alert

        print(f"PID {os.getpid()}: Sensor data acquisition started.")

        tasks = [insert_data()]
        await asyncio.gather(*tasks)

    except Exception as e:
        print(f"PID {os.getpid()}: An error occurred in main loop: {e}")
    finally:
        # --- NEW: Ensure proper cleanup for pigpio and sensor ---
        if sensor:
            print(f"PID {os.getpid()}: Closing ADXL345 sensor.")
            sensor.close() # This will call i2cClose and gpioTerminate from C++

        if pi_instance and pi_instance.connected:
            print(f"PID {os.getpid()}: Disconnecting from pigpiod.")
            pi_instance.stop() # Disconnects this process from pigpiod
        # --- END NEW ---


if __name__ == "__main__":
    try:
        # --- NEW: Correct way to run asyncio event loop ---
        # This resolves the "There is no current event loop" warning for Python 3.10+
        asyncio.run(main())
        # --- END NEW ---

    except KeyboardInterrupt:
        print(f"PID {os.getpid()}: KeyboardInterrupt detected. Initiating graceful shutdown.")
        # The cleanup is now handled within the main() coroutine's finally block
        # when the asyncio loop is stopped by the KeyboardInterrupt.
        if buffer:
            # You might want to consider making this a synchronous write to ensure it flushes on exit
            # For simplicity, keeping it synchronous here.
            with open(file_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(buffer)
            print(f"PID {os.getpid()}: Final buffer flushed to disk.")
        print(f"PID {os.getpid()}: Process exiting.")
    except Exception as e:
        print(f"PID {os.getpid()}: Unhandled exception in __main__: {e}")
        # The cleanup in main() should still be called if the loop exits via an exception