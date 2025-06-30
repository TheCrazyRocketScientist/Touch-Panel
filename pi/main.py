import time
import struct
import smbus2
import RPi.GPIO as GPIO
from pi.ADXL_debug import ADXL345  # make sure path is correct

# --- Setup I²C bus 0 ---
bus0 = smbus2.SMBus(0)

# --- Setup GPIO mode ---
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

# --- Setup ADXL345 Sensor (Address = 0x1D, GPIO17 = INT0) ---
sensor = ADXL345("sensor0", bus0, 17, address_select=True, number=0)

# --- Initialize and start data acquisition ---
sensor.startup()
sensor.get_data()

# --- Main Loop: Read 100 samples and print them ---
try:
    for _ in range(100):
        time.sleep(0.01)  # 10ms for ~100Hz effective rate
        print(f"X: {sensor.x_vals}, Y: {sensor.y_vals}, Z: {sensor.z_vals}, Tap: {getattr(sensor, 'tap', 0)}")
except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    sensor.close()
         