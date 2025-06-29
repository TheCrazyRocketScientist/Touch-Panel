import serial
import time
from collections import deque

# === CONFIG ===
PORT = 'COM14'         # Change to your Arduino's COM port
BAUD_RATE = 500000
PACKET_SIZE = 14      # bytes sent by Serial.write(buffer, 14)
WINDOW_SIZE = 1000    # number of samples to average over

# === SETUP ===
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
print(f"Connected to {PORT} at {BAUD_RATE} baud")

timestamps = deque(maxlen=WINDOW_SIZE)

try:
    while True:
        # Read exactly 14 bytes
        data = ser.read(PACKET_SIZE)
        if len(data) == PACKET_SIZE:
            now = time.time()
            timestamps.append(now)

            # Estimate sample rate
            if len(timestamps) >= 2:
                dt = timestamps[-1] - timestamps[0]
                if dt > 0:
                    rate = (len(timestamps) - 1) / dt
                    print(f"[{now:.3f}] Sample Rate: {rate:.2f} Hz")

except KeyboardInterrupt:
    print("\nDisconnected.")
    ser.close()
