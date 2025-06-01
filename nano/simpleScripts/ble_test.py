import asyncio
import struct
from bleak import BleakClient


"""
This script is used for testing BLE Scanner class's effectiveness.

This script connects to the currently configured SENSOR0

"""
ADDRESS = "82:AB:58:D0:24:3A"  # Replace with your Arduino's MAC address

# Your custom characteristic, specified in the ble_peripheral.ino file.
#128 bit version, the first segment of this id is user defined, evrything else is set by the device.
#UUID defined in the ble_peripheral.ino file is "2101"
CHARACTERISTIC_UUID = "00002101-0000-1000-8000-00805f9b34fb" 

def notification_handler(sender, data):

    # Unpack 12 bytes into 3 floats
    x, y, z,index = struct.unpack("<fffh", data)
    print(f"x: {x:.3f}, y: {y:.3f}, z: {z:.3f}")

async def main():

    #set up an async context manager for the BleakClient class instance.
    #consult the ble_scanner.py script for detailed information on how async context manager handles objects

    async with BleakClient(ADDRESS) as client:

        #main enters the context manager, and client is an alias for the connected sensor

        print("Connecting...")

        #suspend main in the event loop till is_connected() method returns true.
        await client.is_connected()
        print("Connected!")

        #suspend main till current machine starts recieving notifications from the arduino nano
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)

        while True:

            #sleep to avoid busy/tight loop, suspending main temporarily handles processes/threads opened by bleak's c/cpp bindings
            await asyncio.sleep(1)

if __name__ == "__main__":

    #this check is redundant, remove later, as this script is always executed directly by the user for testing.
    try:
        
        #submit main to the event loop
        asyncio.run(main())

    except KeyboardInterrupt:

        #handle the keyboard interrupt.
        print("Disconnected.")
