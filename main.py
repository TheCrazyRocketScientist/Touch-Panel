"""
Obtain set number of readings and estimate optimum values of DUR and THRESH_TAP


Reading Procedure for sensors:
1.)Bind interrupts for data_ready only, No interrupt for tap detection
2.)Every time the interrupt is triggered, check the int source register
3.)If tap detected bit is set, then set the tap flag to 1 in the buffer
4.)After all sensors are read, write their data in the combined buffer
5.)buffer batch inserts data into the file


All sensor reads are I/O bound tasks, not to mention file handling.
"""

"""
Raspberry Pi Pinout

BUS0:
SDA GPIO0
SCL GPIO1

BUS1:
SDA GPIO2
SCL GPIO3

SENSOR0:
INT0 GPIO5

SENSOR1:
INT0 GPIO6

SENSOR2:
INT0 GPIO23

SENSOR3:
INT0: GPIO24


"""
from ADXL import ADXL345
from smbus2 import SMBus

#this import changes depending on platform, mock is used for development, Rpi.GPIO is used on the Pi board
try:
   import RPi.GPIO as GPIO
except ImportError:
   import Mock.GPIO as GPIO




bus0 = SMBus(0)
bus1 = SMBus(1)

#uncomment set to broadcom numbering scheme
GPIO.setmode(GPIO.BCM)
#comment to set to board numbering scheme
#GPIO.setmode(GPIO.BOARD)
#set interrput to input wtih pull down to reduce noise and weirdness
#this is used to handle active high interrupts

sensor0 = ADXL345("upper left",bus0,5,False,0)
sensor1 = ADXL345("upper right",bus0,6,True,1)
sensor2 = ADXL345("lower left",bus1,23,False,2)
sensor3 = ADXL345("lower right",bus1,24,True,3)

sensor0.startup()
sensor1.startup()
sensor2.startup()
sensor3.startup()

ADXL345.close()



