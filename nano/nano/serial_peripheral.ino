/*

Written By: Paranjay Lokesh Chaudhary
Last revision as of 26-06-2025

Sketch is based on the official battery monitor notification example by Arduino.
Consult said sketch if any bugs/issues are encountered which are not addressed below.

This sketch works for the Arduino Nano 33 BLE Sense Rev 1/2.

TO-DO:

Create a python script to replace lines in accelerometer library to change sample rate and sensitivity. Not immediate priority.

Automate upload process in the future by addressing above point and use python to change necessary variables in this program as well.

Pinout:

As of this revision, there is nothing connected to any pin on the Arduino.
Connect only a microUSB cable.
This program does not account for instabilities to the BLE module caused by using pins as inputs/outputs.

Additional Info for future work:

1.) All RGB LEDs are active LOW, current pin numbers are correct for all color channels.
2.) Changed dependency from delay() to millis() in main loop, set to 5 millseconds per iteration for stability
3.) Removed .is_subscribed() check in the main loop as it was found to be redundant and did not improve reliability
4.) Appended sensor number at the end of the data packet to help centeral organize sensor data more effectively.
    This has caused the buffer size to increase from 12 to 14 bytes; a short int was used to indicate sensor number.
    On the client side, to unpack using struct, change input string from '<fff' to '<fffh' and change buffer size to 14.

Expected Output:

1.)Green power LED should be always on, stop if any flickering is observed.
2.)Orange LED_BUILTIN will always be off, unless a (upload) error has occured. 
3.)Blue LED in RGB package will glow briefly to indicate sucessful initalization of the board. If any error occurs, the LED_BUILTIN will 
   blink in a regular pattern. Check if all necessary header files are installed correctly,check if IMU library is installed correctly and correct
   library is uncommented according to the board revision.
4.)If the Red LED in the RGB Package blinks in a regular manner, this indicates that the device is waiting for a host to connect.
5.)If the Green LED in the RGB Package is switched on, then the device is connected and boradcasting data.

Any deviation from this sequence means something has gone wrong.
Consult official documentation.

For Serial Version: LED will only blink when serial connection is not detected
Other rules are applicable

*/


//Not included for serial version
//#include <ArduinoBLE.h>

// Uncomment based on your board revision
// This will use modified libarires located in libs folder in project directory

//#include <Arduino_LSM6DSOX.h> // keep for backup

//#include "E:\Projects\Vibration Sensing Touch Panel\libs\Arduino_BMI270_BMM150\src\Arduino_BMI270_BMM150.h" //rev2
//#include "E:\Projects\Vibration Sensing Touch Panel\libs\Arduino_LSM9DS1\src\Arduino_LSM9DS1.h" //rev1


#define LED_RGB_RED 22
#define LED_RGB_GREEN 23
#define LED_RGB_BLUE 24

float x, y, z;
int sensor_number;
unsigned long last_time = 0;

void indicateError() {
  while (true) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(100);
    digitalWrite(LED_BUILTIN, LOW);
    delay(100);
  }
}

void setup() {
  pinMode(LED_RGB_RED, OUTPUT);
  pinMode(LED_RGB_GREEN, OUTPUT);
  pinMode(LED_RGB_BLUE,OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);

  digitalWrite(LED_RGB_BLUE,LOW);
  digitalWrite(LED_RGB_RED,HIGH);
  digitalWrite(LED_RGB_GREEN,HIGH);

  //select a high baud rate for serial
  Serial.begin(115200);

  //start serial
  while(!Serial){
    indicateError();;
  }
  // IMU start
  if (!IMU.begin()) {
    indicateError();
  }
  
  sensor_number = 2;

  delay(100);
  digitalWrite(LED_RGB_BLUE,HIGH);


}

void loop() {

  if (Serial) {

    digitalWrite(LED_RGB_GREEN, LOW);
    digitalWrite(LED_RGB_RED, HIGH);

    while (Serial) {
      if(micros()-last_time >= 2000){
        last_time = micros();
      if (IMU.accelerationAvailable()) {
        if (IMU.readAcceleration(x, y, z)) {

          uint8_t buffer[14];
          memcpy(buffer, &x, 4);
          memcpy(buffer + 4, &y, 4);
          memcpy(buffer + 8, &z, 4);
          memcpy(buffer + 12, &sensor_number,2);

          Serial.write(buffer, 14);
        } 
      }
    }

    }
    digitalWrite(LED_RGB_GREEN, HIGH);

  }

  else{
        digitalWrite(LED_RGB_RED, HIGH);
        delay(100);
        digitalWrite(LED_RGB_RED, LOW);
        delay(100);
  }
    

}
