import asyncio
from bleak import BleakClient
import argparse
import struct
import bleak
#this script grabs data from a given sensor
#this script is to be run with other processes running other instances of this script


#setting up arguments for CLI

parser = argparse.ArgumentParser()

parser.add_argument("-n","--name",type=str,help="Sets sensor name to grab data from.",default="Upper_Left")
parser.add_argument("-a","--address",type=str,help="Sets the adress of BLE server.",default="")

args = parser.parse_args()
name = args.name
address = args.address

ACCEL_DATA_UUID =  "00002101-0000-1000-8000-00805f9b34fb".lower()



def get_data(sender, data):
    x, y, z = struct.unpack('<fff', data)
    print(f"[{name}] x: {x:.3f}, y: {y:.3f}, z: {z:.3f}")

async def main():
    
    async with BleakClient(address) as client:
        if client.is_connected:
            await asyncio.sleep(1)
            services = client.services
            await asyncio.sleep(1)
            await client.start_notify(ACCEL_DATA_UUID,get_data)
            while True:
                await asyncio.sleep(0.5)
try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("Ending connection")

