import sys
import time
import logging
import colorlog
import struct
import smbus2


#this import changes depending on platform, mock is used for development, Rpi.GPIO is used on the Pi board
try:
   import RPi.GPIO as GPIO
except ImportError:
   import Mock.GPIO as GPIO




"""
To-Do

No tasks remaining finished as of 22/05/25
"""

class ADXL_Helper:

   attempts = 5
   data_rate = 200
   data_rate_200 = 0x0B
   device_id = 0xE5
   ADXL345_ALT_ADDRESS = 0x1D
   ADXL345_DEFAULT_ADDRESS = 0x53

class ADXL_Registers:

   DEVID = 0x00
   OFSX = 0x1E
   OFSY = 0x1F
   OFSZ = 0x20
   LATENT = 0x22
   WINDOW = 0x23
   THRESH_ACT = 0x24
   THRESH_INACT = 0x25
   ACT_INACT_CTL = 0x27
   TIME_INACT = 0x26
   THRESH_FF = 0x28
   TIME_FF = 0x29
   TAP_AXES = 0x2A
   BW_RATE = 0x2C
   POWER_CTL = 0x2D
   INT_ENABLE = 0x2E
   INT_MAP = 0x2F
   INT_SOURCE = 0x30
   DATA_FORMAT = 0x31
   DATAX0 = 0x32
   DATAX1 = 0x33
   DATAY0 = 0x34
   DATAY1 = 0x35
   DATAZ0 = 0x36
   DATAZ1 = 0x37
   FIFO_CTL = 0x38



class ADXL345:

   def __init__(self,name,bus,pin0,address_select=False,number=0):
      self.name = name
      self.bus = bus
      self.number = number
      #Both pins must follow BCM numbering scheme
      self.INT0 = pin0
      #buffers to store x,y,z vals
      self.x_vals = []
      self.y_vals = []
      self.z_vals = []
      self.tap = 0

      if address_select == False:
         self.address = ADXL_Helper.ADXL345_DEFAULT_ADDRESS
      else:
         self.address = ADXL_Helper.ADXL345_ALT_ADDRESS

      GPIO.setup(self.INT0,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)

      """
      Set up logging here, for each sensor instance a seperate logger will be used.
      > Each logger is identified by the sensor number of the ADXL instance
      > Logger format is colored for easy identification
      > Specify format and color scheme
      > Logger output goes to standard output

      """
      logger = logging.getLogger(f"SENSOR{self.number}")
      logger.setLevel(logging.DEBUG)
      console_handler = logging.StreamHandler(stream=sys.stdout)
      #formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s',datefmt='%H:%M:%S')
      formatter = colorlog.ColoredFormatter(
         fmt=(
            '%(asctime)s | '
            '%(white)s%(name)s | '
            '%(log_color)s%(levelname)s | '
            '%(log_color)s%(message)s'
         ),
         datefmt='%H:%M:%S',
         log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'white,bg_red'
         },
         secondary_log_colors={
            'asctime': {
                  'DEBUG':    'white',
                  'INFO':     'white',
                  'WARNING':  'white',
                  'ERROR':    'white',
                  'CRITICAL': 'white',
            },
            'name': {
                  'DEBUG':    'white',
                  'INFO':     'white',
                  'WARNING':  'white',
                  'ERROR':    'white',
                  'CRITICAL': 'white',
            }
         },
         style='%'
      )

      console_handler.setFormatter(formatter)
      logger.addHandler(console_handler)

      self.logger = logger



   def write_to_register(self,register_addr,content):
      """
      This is the function which is used to handle all read/writes to necessary registers.
      Only use with context manager outside function call
      Use read_data_byte for reading data from smbus2
      """
      self.logger.info(f"Writing to the register {hex(register_addr)} with data: {hex(content)}")

      for _ in range(ADXL_Helper.attempts):
         self.bus.write_byte_data(self.address,register_addr,content)
         read_content = self.bus.read_byte_data(self.address,register_addr)
         if(read_content == content):
            return
         
      raise IOError(f"Failed to write to sensor {self.number} at register {hex(register_addr)} after {ADXL_Helper.attempts} attempts") 

   def get_data(self):
      """
      Function to handle reads to the sensor during operation.
      INT 0 is DATA_READY interrupt
      INT 1 is TAP_DETECTED interrupt, for now INT1 is left to not be used for simplicity 
      
      This fuction sets up and attaches an interrupt to this sensor and handles the buffers
      """
      
      self.x_vals = 0
      self.y_vals = 0
      self.z_vals = 0
      self.tap = 0

      self.logger.info("Enabling data acquistion.")

      GPIO.add_event_detect(self.INT0, GPIO.RISING, callback=self.read_data)


   def stop_data(self):
   
      #Disables all sensor reads by removing interrupts responsible for reading the data.

      self.logger.info("Disabling data acquistion.")
      GPIO.remove_event_detect(self.INT0)


   def read_data(self,channel=None):

      """
      Function for reading data, will store latest values and tap flag when interrupt is triggered

      This will always use single integers, the lists are overidden to ensure only one value remaines in this buffer
      """

      try:
         source_content = self.bus.read_byte_data(self.address,ADXL_Registers.INT_SOURCE)
      except (IOError,OSError) as e:
         self.logger.error(f"I2C read error during data read [INT_SOURCE]: {e}")
         sys.exit()
      except (TimeoutError) as e:
         self.logger.error(f"I2C bus timed out during data read. [INT_SOURCE]")
         sys.exit()
      except Exception as e:
         self.logger.error(f"I2C error occured during data read. [INT_SOURCE]")
         sys.exit()


      if not (source_content & 0x80):
         return
      
      else:

         try:
            data = self.bus.read_i2c_block_data(self.address,ADXL_Registers.DATAX0,6)
         except (IOError,OSError) as e:
            self.logger.error(f"I2C read error while reading data block during sensor read: {e}")
         except (TimeoutError) as e:
            self.logger.error(f"I2C bus timed out while reading data block during sensor read.")
         except Exception as e:
            self.logger.error(f"I2C error occured while reading data block during sensor read.")

         data_bytes = bytes(data)
         x,y,z = struct.unpack('<hhh',data_bytes)

         self.x_vals = x
         self.y_vals = y
         self.z_vals = z

      if (source_content & 0x40):
         self.tap = 1
      else:
         self.tap = 0

   
   def startup(self):
      """
      Startup function which manages initialization and calibration
      After this stage, the sensor is armed

      > Init sensor,
      > Calibrate and insert values in offset registers
      > Reinit sensor, (sensor restart)
      > Sensor is armed.

      """
      self.logger.info("Sensor startup has begun.")

      self.start_init()
      self.calibrate()
      self.start_init()

      self.logger.info("Sensor startup has ended.")
      self.logger.warning("Sensor is armed.")

   def start_init(self):
      """
      This function initializes the sensor

      Sequence of operations:

      > Reads DEVID register to check if sensor is healthy
      > Sets the sensor to normal operation
      > Sets the sensor to measurement mode
      > DATA_READY and SINGLE_TAP interrupts are enabled
      > Sets interrupt mode, data format, set FULL_RES and set range to 2G
      > INT_INVERT is set to 0, active high
      > Disable FIFO by using bypass mode
      > Set DATA_READY to INT0
      > Set SINGLE_TAP to INT1
      > Set DATA_RATE to 200Hz

      """

      self.logger.info("Sensor initialization has started.")

      try:

         device_id = self.bus.read_byte_data(self.address,ADXL_Registers.DEVID)
         if(device_id != ADXL_Helper.device_id):
            raise IOError(f"Sensor {self.number} device id invalid, stopping operation.")
         

         self.write_to_register(ADXL_Registers.BW_RATE,0x00)
         self.write_to_register(ADXL_Registers.POWER_CTL,0x08)
         self.write_to_register(ADXL_Registers.INT_ENABLE,0xC0)
         self.write_to_register(ADXL_Registers.DATA_FORMAT,0x08)
         self.write_to_register(ADXL_Registers.FIFO_CTL,0x00)
         self.write_to_register(ADXL_Registers.INT_MAP,0x40)
         self.write_to_register(ADXL_Registers.BW_RATE,ADXL_Helper.data_rate_200)

         self.logger.info("Initial values have been written to registers.")

         """
         > Next sequence; disable freefall, double tap, act/inact; set tap axis
         
         """
         self.logger.info("Disabling freefall mode.")
         self.disable_freefall()

         self.logger.info("Disabling activity mode.")
         self.disable_act()

         self.logger.info("Disabling double tap mode.")
         self.disable_double_tap()

         self.logger.info("Setting tap axis")
         self.set_axis()

      except (IOError,OSError) as e:
         self.logger.error(f"I2C Write Error: {e}")
         sys.exit()
      except (TimeoutError) as e:
         self.logger.error(f"I2C bus timed out during write.")
         sys.exit()
      except Exception as e:
         self.logger.error(f"I2C error occured during write.")
         sys.exit()


   def calibrate(self):
      """
      Calibration routine, will run after first initialization of sensor
      > Set all offsets to 0 to get raw values from DATA registers
      > Calculate minimum number of samples, 0.1*sampling rate
      > Set up DATA_READY interrupt pin on the pi, with a pull down resistor
      > Set up the interrupt, and specify callback function

      """
      self.logger.info("Resetting offsets")
      self.reset_offsets()

      self.calibration_samples =  int(0.1*ADXL_Helper.data_rate)

      # Attach interrupt on rising edge (INT0 goes from LOW to HIGH on data ready)
      GPIO.add_event_detect(self.INT0, GPIO.RISING, callback=self.calibration_callback)

      while len(self.x_vals) < self.calibration_samples:
         time.sleep(0.01)
      
      GPIO.remove_event_detect(self.INT0)
      
      mean_x = sum(self.x_vals) / len(self.x_vals)
      mean_y = sum(self.y_vals) / len(self.y_vals)
      mean_z = sum(self.z_vals) / len(self.z_vals)

      # Compute offsets (divide by 4 per datasheet)
      offset_x = int(-mean_x / 4)
      offset_y = int(-mean_y / 4)
      offset_z = int(-mean_z / 4)

      # Convert to 8-bit 2's complement (if negative)
      # This is actually just the lower 8 bits, as the register is 8-bit signed
      offset_x_byte = offset_x & 0xFF
      offset_y_byte = offset_y & 0xFF
      offset_z_byte = offset_z & 0xFF

      # Write offsets to sensor registers
      try:
         self.write_to_register(ADXL_Registers.OFSX, offset_x_byte)
         self.write_to_register(ADXL_Registers.OFSY, offset_y_byte)
         self.write_to_register(ADXL_Registers.OFSZ, offset_z_byte)
      except (IOError,OSError) as e:
         self.logger.error(f"I2C write error while setting offsets: {e}")
      except (TimeoutError) as e:
         self.logger.error(f"I2C bus timed out while setting offsets.")
      except Exception as e:
         self.logger.error(f"I2C error occured during setting offsets.")


      self.x_vals,self.y_vals,self.z_vals = [],[],[]

      self.logger.info("Offsets have been written.")


   def calibration_callback(self,channel):
      """
      Callback function for calibration sequence
      > Confirm from INT_SOURCE that Data Ready was triggered
      > Check if number of samples in the bus is more than specified samples for calibration
      > Conduct multi-byte read, convert obtained list to bytes() object
      > Unpack into 3 short signed ints using struct
      > append it to x, y and z buffers
      """

      if len(self.x_vals) >= self.calibration_samples:
         return
      
      try:
         source_content = self.bus.read_byte_data(self.address,ADXL_Registers.INT_SOURCE)
      except (IOError,OSError) as e:
         self.logger.error(f"I2C read error while calibrating [INT_SOURCE]: {e}")
         sys.exit()
      except (TimeoutError) as e:
         self.logger.error(f"I2C bus timed out while calibrating. [INT_SOURCE]")
         sys.exit()
      except Exception as e:
         self.logger.error(f"I2C error occured during calibration. [INT_SOURCE]")
         sys.exit()


      if not (source_content & 0x80):
         return

      try:
         data = self.bus.read_i2c_block_data(self.address,ADXL_Registers.DATAX0,6)
      except (IOError,OSError) as e:
         self.logger.error(f"I2C read error while reading data block for calibration: {e}")
      except (TimeoutError) as e:
         self.logger.error(f"I2C bus timed out while reading data block for calibration.")
      except Exception as e:
         self.logger.error(f"I2C error occured while reading data block for calibration.")

      data_bytes = bytes(data)
      x,y,z = struct.unpack('<hhh',data_bytes)

      self.x_vals.append(x)
      self.y_vals.append(y)
      self.z_vals.append(z)  

   def reset_offsets(self):
      """
      Helper function to reset offsets along each axis.
      """
      try:
         self.write_to_register(ADXL_Registers.OFSX, 0x00)
         self.write_to_register(ADXL_Registers.OFSY, 0x00)
         self.write_to_register(ADXL_Registers.OFSZ, 0x00)
      except (IOError,OSError) as e:
         self.logger.error(f"I2C write error while resetting offsets: {e}")
      except (TimeoutError) as e:
         self.logger.error(f"I2C bus timed out while resetting offsets.")
      except Exception as e:
         self.logger.error(f"Error occured during resetting offsets.")


   def set_axis(self):
      """
      For now, enable tap detection on all axes

      """
      self.write_to_register(ADXL_Registers.TAP_AXES,0x0F)
      

   def disable_freefall(self):
      """
      Disabling Free-Fall Detection:
      > Ensure Free Fall interrupt is disabled before starting
      > Disable THRESH_FF by setting all bits to 0.
      > Disable TIME_FF by setting all bits to 0.

      """
      self.write_to_register(ADXL_Registers.THRESH_FF,0x00)
      self.write_to_register(ADXL_Registers.TIME_FF,0x00)
      
   def disable_double_tap(self):
      """
      > Disable LATENT Register to disable double tap detection.
      > Disable WINDOW Register to disable double tap detection.
      > Set Supress bit to 1 in TAP_AXES Register to ensure tap_detection is suppressed.

      """
      self.write_to_register(ADXL_Registers.LATENT,0x00)
      self.write_to_register(ADXL_Registers.WINDOW,0x00)
      #SUPRESS bit is set to 1, use bitwise OR/ADD to ensure TAP_AXES values are not overriden
      self.write_to_register(ADXL_Registers.TAP_AXES,0x08)
      
      
   def disable_act(self):
      """
      > Disable ACT/INACT Interrupt before starting.
      > Disable THRESH_ACT by setting all bits to 0.
      > Disable THRESH_INACT by setting all bits to 0.
      > Disable ACT_INACT_CTL by setting all bits to 0..
      > MIGHT CAUSE ERROR:Set TIME_INACT to 0, this will cause an interrupt when the output data is less than the value in THRESH_INACT
      
      """
      self.write_to_register(ADXL_Registers.THRESH_ACT,0x00)
      self.write_to_register(ADXL_Registers.THRESH_INACT,0x00)
      self.write_to_register(ADXL_Registers.ACT_INACT_CTL,0x00)
      #Remove the TIME_INACT line if any errors are caused
      self.write_to_register(ADXL_Registers.TIME_INACT,0x00)

   def close(self):
      """
      Method to semantically indicate sensor is out of operation after this method

      Calling any one sensor will clean GPIO for all
      """
      self.logger.info("Closing sensor")

      GPIO.cleanup()

