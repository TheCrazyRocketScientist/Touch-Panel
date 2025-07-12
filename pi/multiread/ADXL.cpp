#include <pigpio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#include "ADXL.hpp"


#define BUS_SPEED 1000000

namespace py = pybind11;
using namespace std;

py::function ADXL345::python_callback;


ADXL345* global_instance = nullptr;


   ADXL345::ADXL345(int bus, bool address_select ,int pin):pin(pin),bus(bus),attempts(ADXL_Helper::attempts),address_select(address_select)
   
   {
      //channel arg in spiOpen automatically allots a bus and cs pin

      global_instance = this;

      if(gpioInitialise() < 0){
         std::cerr << "PIGPIO was not initialized\n" ; 
      }

      if(address_select == true){
         this->address = ADXL345_ALT_ADDRESS;
      }
      else{
         this->address = ADXL345_DEFAULT_ADDRESS;
      }

      this->i2c_handle = i2cOpen(this->bus,this->address,0);

      if(this->i2c_handle < 0){
         std::cerr << "PIGPIO I2C was not initialized\n" ; 
      }

      //set up interrupt pin
      gpioSetMode(this->pin,PI_INPUT);
      gpioSetPullUpDown(this->pin, PI_PUD_DOWN);

   }


   uint8_t ADXL345::read_register(uint8_t register_addr) {

      this->reg_content = i2cReadByteData(this->i2c_handle, register_addr);

      if (this->reg_content < 0){
         throw std::runtime_error("I2C read failed");
      }

    return this->reg_content;
}
   
   void ADXL345::write_to_register(uint8_t register_addr, uint8_t content){

    for (int i = 0; i < this->attempts; ++i) {
        if (i2cWriteByteData(this->i2c_handle, register_addr, content) < 0)
            continue;
        if (read_register(register_addr) == content)
            return;
    }
    throw std::runtime_error("I2C write failed after several attempts.");

}


   void ADXL345::calibrate(){

      reset_offsets();

      this->calibration_samples = static_cast<int>(0.1*ADXL_Helper::data_rate);
      /*
      de latch the interrupt
      */
      this->source_content = read_register(ADXL_Registers::INT_SOURCE);
      this->refresh = i2cReadI2CBlockData(this->i2c_handle, ADXL_Registers::DATAX0, (char*)out_buffer, 6);

      gpioSetAlertFunc(this->pin, calibration_callback);

      while(this->x_calib.size() < this->calibration_samples){
         gpioDelay(100);
      }

      gpioSetAlertFunc(this->pin, nullptr);

      this->offset_x = std::accumulate(x_calib.begin(), x_calib.end(), 0);
      this->offset_y = std::accumulate(y_calib.begin(), y_calib.end(), 0);
      this->offset_z = std::accumulate(z_calib.begin(), z_calib.end(), 0);

      this->offset_x = (-1*(this->offset_x))/(4*x_calib.size());
      this->offset_y = (-1*(this->offset_y))/(4*y_calib.size());
      this->offset_z = (-1*(this->offset_z))/(4*z_calib.size());

      this->offset_x = static_cast<int8_t>(this->offset_x);
      this->offset_y = static_cast<int8_t>(this->offset_y);
      this->offset_z = static_cast<int8_t>(this->offset_z);

      try{

         write_to_register(ADXL_Registers::OFSX, static_cast<uint8_t>(this->offset_x));
         write_to_register(ADXL_Registers::OFSY, static_cast<uint8_t>(this->offset_y));
         write_to_register(ADXL_Registers::OFSZ, static_cast<uint8_t>(this->offset_z));

      }
      catch (const std::exception& e) {

         std::cerr << "Error writing Offsets: " << e.what() << std::endl;
         return;
      }

   }

   void ADXL345::reset_offsets(){

      write_to_register(ADXL_Registers::OFSX, 0x00);
      write_to_register(ADXL_Registers::OFSY, 0x00);
      write_to_register(ADXL_Registers::OFSZ, 0x00);

   }

   void ADXL345::set_axis(){

      write_to_register(ADXL_Registers::THRESH_FF,0x00);
      write_to_register(ADXL_Registers::TIME_FF,0x00);

   }

   void ADXL345::disable_freefall(){

      write_to_register(ADXL_Registers::THRESH_FF,0x00);
      write_to_register(ADXL_Registers::TIME_FF,0x00);
   }

   void ADXL345::disable_double_tap(){

      write_to_register(ADXL_Registers::LATENT,0x00);
      write_to_register(ADXL_Registers::WINDOW,0x00);
      //SUPRESS bit is set to 1, use bitwise OR/ADD to ensure TAP_AXES values are not overriden
      write_to_register(ADXL_Registers::TAP_AXES,0x08);
   }

   void ADXL345::disable_act(){

      write_to_register(ADXL_Registers::THRESH_ACT,0x00);
      write_to_register(ADXL_Registers::THRESH_INACT,0x00);
      write_to_register(ADXL_Registers::ACT_INACT_CTL,0x00);
      //Remove the TIME_INACT line if any errors are caused
      write_to_register(ADXL_Registers::TIME_INACT,0x00);
   }

   void ADXL345::start_init(){

      try{

         if(read_register(ADXL_Registers::DEVID) != ADXL_Helper::device_id){
            throw std::runtime_error("Sensor Device ID is invalid.");

         }

      write_to_register(ADXL_Registers::BW_RATE,0x00);
      write_to_register(ADXL_Registers::POWER_CTL,0x08);
      write_to_register(ADXL_Registers::INT_ENABLE,0x80);
      write_to_register(ADXL_Registers::DATA_FORMAT,0x08);
      write_to_register(ADXL_Registers::FIFO_CTL,0x00);
      write_to_register(ADXL_Registers::INT_MAP,0x00);
      write_to_register(ADXL_Registers::BW_RATE,ADXL_Helper::data_rate_200);

      this->disable_freefall();
      this->disable_act();
      this->disable_double_tap();
      this->set_axis();

      }

   catch (const std::exception& e) {
         std::cerr << "Error during start init: " << e.what() << std::endl;
         return;
      }
   }

   void ADXL345::startup(){

      start_init();
      calibrate();

   }


   void ADXL345::get_data(){
      /*
      de latch the interrupt
      */
      this->source_content = read_register(ADXL_Registers::INT_SOURCE);
      this->refresh = i2cReadI2CBlockData(this->i2c_handle, this->address, (char*)out_buffer, 6);

      gpioSetAlertFunc(this->pin, read_data);

   }

   void ADXL345::close(){

      i2cClose(this->i2c_handle);
      gpioSetAlertFunc(this->pin, nullptr);
      gpioTerminate();

   }

   void ADXL345::register_callback(py::function sent_callback){

      this->python_callback = sent_callback;

   }

void calibration_callback(int gpio, int level, uint32_t tick){

         if(level == 1 && global_instance != nullptr){

            if (global_instance->x_calib.size() >= global_instance->calibration_samples){
               return;
            }
            try{
               global_instance->source_content = global_instance->read_register(ADXL_Registers::INT_SOURCE);
            }
            catch (const std::exception& e) {
               std::cerr << "Error reading INT_SOURCE: " << e.what() << std::endl;
               return;
            }

            if((global_instance->source_content & 0x80) == 0x00){
               return;
            }

            global_instance->refresh = i2cReadI2CBlockData(global_instance->i2c_handle,ADXL_Registers::DATAX0,(char*)global_instance->out_buffer,6);

            if(global_instance->refresh != 6){
               return;
            }

            global_instance->raw_x = static_cast<int16_t> ((global_instance->out_buffer[1] << 8) | global_instance->out_buffer[0]);
            global_instance->raw_y = static_cast<int16_t> ((global_instance->out_buffer[3] << 8) | global_instance->out_buffer[2]);
            global_instance->raw_z = static_cast<int16_t> ((global_instance->out_buffer[5] << 8) | global_instance->out_buffer[4]);

            global_instance->x_calib.push_back(global_instance->raw_x);
            global_instance->y_calib.push_back(global_instance->raw_y);
            global_instance->z_calib.push_back(global_instance->raw_z);

         }

      }


void read_data(int gpio, int level, uint32_t tick){
      
      if(level == 1 && global_instance != nullptr){

            try{
               global_instance->source_content = global_instance->read_register(ADXL_Registers::INT_SOURCE);
            }
            catch (const std::exception& e) {
               std::cerr << "Error reading INT_SOURCE: " << e.what() << std::endl;
               return;
            }

            if((global_instance->source_content & 0x80) == 0x00){
               return;
            }

            global_instance->refresh = i2cReadI2CBlockData(global_instance->i2c_handle,ADXL_Registers::DATAX0,(char*)global_instance->out_buffer,6);
            
            if(global_instance->refresh != 6){
               return;
            }

            global_instance->raw_x = static_cast<int16_t> ((global_instance->out_buffer[1] << 8) | global_instance->out_buffer[0]);
            global_instance->raw_y = static_cast<int16_t> ((global_instance->out_buffer[3] << 8) | global_instance->out_buffer[2]);
            global_instance->raw_z = static_cast<int16_t> ((global_instance->out_buffer[5] << 8) | global_instance->out_buffer[4]);

            global_instance->x = ((static_cast<float> (global_instance->raw_x))/256.0);
            global_instance->y = ((static_cast<float> (global_instance->raw_y))/256.0);
            global_instance->z = ((static_cast<float> (global_instance->raw_z))/256.0);

            if(global_instance->python_callback){

               try{

               py::gil_scoped_acquire acquire;
               global_instance->python_callback();

               }
               catch (const std::exception& e) {

               std::cerr << "Python Error: " << e.what() << std::endl;
               return;
      }

            }

   }

   }