import os
import sys
import time
import smbus2
import argparse
import RPi.GPIO as GPIO
from pi.ADXL import ADXL345
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument("-n", "--sensor_num", type=int, help="Sensor number", default=0)
parser.add_argument("-p", "--int_pin", type=int, help="Interrupt pin for ADXL", default=4)
parser.add_argument("-b", "--bus", type=int, help="I2C bus number", default=0)
parser.add_argument("-a", "--alt_addr", action="store_true", help="Use alternate ADXL address (0x1D), default 0x53")

args = parser.parse_args()

sensor_num = args.sensor_num
int_pin = args.int_pin
bus = args.bus
addr_sel = args.alt_addr

# Initialize I2C bus and GPIO
bus_obj = smbus2.SMBus(bus)
GPIO.setmode(GPIO.BCM)
GPIO.setup(int_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# Create sensor instance
sensor = ADXL345(f"sensor{sensor_num}", bus_obj, int_pin, address_select=addr_sel, number=sensor_num)

sample_limit = 1000
samples_collected = 0
timestamps = []

def sample_callback(readings):
    global samples_collected, timestamps

    now = time.time()
    timestamps.append(now)
    samples_collected += 1

    if samples_collected >= sample_limit:
        sensor.stop_data()  # Remove interrupt
        print_stats()
        cleanup_and_exit()

def print_stats():
    if len(timestamps) < 2:
        print("Not enough samples collected.")
        return
    duration = timestamps[-1] - timestamps[0]
    avg_rate = (len(timestamps) - 1) / duration
    print(f"Collected {len(timestamps)} samples in {duration:.3f} seconds")
    print(f"Average sample rate: {avg_rate:.2f} Hz")

def cleanup_and_exit():
    sensor.close()
    GPIO.cleanup()
    print("Sensor closed and GPIO cleaned up. Exiting.")
    sys.exit(0)

if __name__ == "__main__":
    try:
        sensor.register_callback(sample_callback)
        sensor.startup()
        sensor.get_data()
        print(f"Starting sample collection for {sample_limit} samples...")
        
        # Loop forever, callback handles exit
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("Interrupted by user.")
        cleanup_and_exit()
