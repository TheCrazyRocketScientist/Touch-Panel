#include <pigpio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <cstdint>

#include "ADXL.hpp"


#define BUS_SPEED 1000000

using namespace std;

ADXL345* global_instance = nullptr;


   ADXL345::ADXL345(int number, int pin):pin(pin),number(number),attempts(ADXL_Helper::attempts)
   
   {
      //channel arg in spiOpen automatically allots a bus and cs pin

      global_instance = this;

      if(gpioInitialise() < 0){
         std::cerr << "PIGPIO was not initialized\n" ; 
      }

      this->spi_handle = spiOpen(this->number,BUS_SPEED,0);

      if(this->spi_handle < 0){
         std::cerr << "PIGPIO was not initialized\n" ; 
      }

      //set up interrupt pin
      gpioSetMode(this->pin,PI_INPUT);
      gpioSetPullUpDown(this->pin, PI_PUD_DOWN);

   }

   uint8_t ADXL345::read_register(uint8_t register_addr){

      this->write_buffer[0] = register_addr | 0x80;
      this->write_buffer[1] = 0x00;

      this->count = spiXfer(this->spi_handle,(char*)this->write_buffer,(char*)this->read_buffer,2);

      if(this->count != 2){
         throw std::runtime_error("SPI Read Failed.");
      }
      else{
         return this->read_buffer[1];
      }

   }

   void ADXL345::write_to_register(uint8_t register_addr, uint8_t content){

      for(int i = 0; i < this->attempts; i++){

         this->write_buffer[0] = register_addr;
         this->write_buffer[1] = content;

         this->count = spiXfer(this->spi_handle,(char*)this->write_buffer,(char*)this->read_buffer,2);

         if(this->count != 2){
            throw std::runtime_error("SPI Write Failed.");
         }
         else{

            if(read_register(register_addr) == content){
               return;
            }

         }
      }
      
      throw std::runtime_error("SPI Write Failed After Several Attempts.");
   }


   void ADXL345::calibrate(){

      reset_offsets();

      this->in_buffer[0] = ADXL_Registers::DATAX0 | 0xC0;
      this->in_buffer[1] = 0x00;

      this->calibration_samples = static_cast<int>(0.1*ADXL_Helper::data_rate);
      /*
      de latch the interrupt
      */
      this->source_content = read_register(ADXL_Registers::INT_SOURCE);
      this->refresh = spiXfer(this->spi_handle,(char*)in_buffer,(char*)out_buffer,6);

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

      this->in_buffer[0] = ADXL_Registers::DATAX0 | 0xC0;
      this->in_buffer[1] = 0x00;


      /*
      de latch the interrupt
      */
      this->source_content = read_register(ADXL_Registers::INT_SOURCE);
      this->refresh = spiXfer(this->spi_handle,(char*)in_buffer,(char*)out_buffer,6);

      gpioSetAlertFunc(this->pin, read_data);

   }

   void ADXL345::close(){

      spiClose(this->spi_handle);
      gpioSetAlertFunc(this->pin, nullptr);
      gpioTerminate();

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
            
            global_instance->in_buffer[0] = ADXL_Registers::DATAX0 | 0xC0;
            global_instance->in_buffer[1] = 0x00;

            global_instance->refresh = spiXfer(global_instance->spi_handle,(char*)global_instance->in_buffer,(char*)global_instance->out_buffer,6);
            if(global_instance->refresh != 6){
               return;
            }

            global_instance->x = static_cast<int16_t> ((global_instance->out_buffer[1] << 8) | global_instance->out_buffer[0]);
            global_instance->y = static_cast<int16_t> ((global_instance->out_buffer[3] << 8) | global_instance->out_buffer[2]);
            global_instance->z = static_cast<int16_t> ((global_instance->out_buffer[5] << 8) | global_instance->out_buffer[4]);

            global_instance->x_calib.push_back(global_instance->x);
            global_instance->y_calib.push_back(global_instance->y);
            global_instance->z_calib.push_back(global_instance->z);

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

            global_instance->in_buffer[0] = ADXL_Registers::DATAX0 | 0xC0;
            global_instance->in_buffer[1] = 0x00;


            global_instance->refresh = spiXfer(global_instance->spi_handle,(char*)global_instance->in_buffer,(char*)global_instance->out_buffer,6);
            if(global_instance->refresh != 6){
               return;
            }

            global_instance->x = static_cast<int16_t> ((global_instance->out_buffer[1] << 8) | global_instance->out_buffer[0]);
            global_instance->y = static_cast<int16_t> ((global_instance->out_buffer[3] << 8) | global_instance->out_buffer[2]);
            global_instance->z = static_cast<int16_t> ((global_instance->out_buffer[5] << 8) | global_instance->out_buffer[4]);

   }

   }