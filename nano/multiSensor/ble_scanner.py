import asyncio
import bleak
import json
import argparse

#this script runs and maps all sensor names to MAC/BLE addresses.
#Run this before gathering data, this script outputs to a json file


#setting up arguments for CLI

parser = argparse.ArgumentParser()
parser.add_argument("-t","--scan_time",type=int,help="Sets duration of BLE scan.",default=1)

args = parser.parse_args()
scan_time = args.scan_time

async def main():

    """
    Explaination for the workings of BleakScanner/BleakClient async context manager.

    set up a context manager with the bleakScanner class, it can do this because it implements methods which define a python context
    manager.

    we create a BleakScanner object, and when we enter the context manager, the __aenter__() method is called, and like a normal python 
    context manager, it returns an object of the class which was used to setup the context manager.

    entering the manager sets up the ble adapter on this machine,(via another async method) and then all statements within it are executed
    and then __aexit__() is called, cleaning up all resources needed. 

    """
    async with bleak.BleakScanner() as scanner:
        #sleep for a set time, this causes the event loop to suspend main for a given amount of time, and till then scanning continues.
        await asyncio.sleep(scan_time)
        
    #create a dictionary to store the result    
    address_dict = {device.name:device.address for device in scanner.discovered_devices if device.name != None }

    #convert this dict to a json string
    json_data = json.dumps(address_dict,indent=2)

    #write this data to a json file
    with open('addresses.json','w',encoding='utf-8') as f:
        f.write(json_data)

asyncio.run(main())