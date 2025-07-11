#pragma once

#include <vector>
#include <cstdint>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;


class ADXL_Helper
{
   public:
      static constexpr uint8_t attempts = 5;
      static constexpr uint8_t data_rate = 200;
      static constexpr uint8_t data_rate_200 = 0x0B;
      static constexpr uint8_t device_id = 0xE5;
      static constexpr uint8_t ADXL345_ALT_ADDRESS = 0x1D;
      static constexpr uint8_t ADXL345_DEFAULT_ADDRESS = 0x53;
};

class ADXL_Registers
{
   public:
      static constexpr uint8_t DEVID = 0x00;
      static constexpr uint8_t OFSX = 0x1E;
      static constexpr uint8_t OFSY = 0x1F;
      static constexpr uint8_t OFSZ = 0x20;
      static constexpr uint8_t LATENT = 0x22;
      static constexpr uint8_t WINDOW = 0x23;
      static constexpr uint8_t THRESH_ACT = 0x24;
      static constexpr uint8_t THRESH_INACT = 0x25;
      static constexpr uint8_t ACT_INACT_CTL = 0x27;
      static constexpr uint8_t TIME_INACT = 0x26;
      static constexpr uint8_t THRESH_FF = 0x28;
      static constexpr uint8_t TIME_FF = 0x29;
      static constexpr uint8_t TAP_AXES = 0x2A;
      static constexpr uint8_t BW_RATE = 0x2C;
      static constexpr uint8_t POWER_CTL = 0x2D;
      static constexpr uint8_t INT_ENABLE = 0x2E;
      static constexpr uint8_t INT_MAP = 0x2F;
      static constexpr uint8_t INT_SOURCE = 0x30;
      static constexpr uint8_t DATA_FORMAT = 0x31;
      static constexpr uint8_t DATAX0 = 0x32;
      static constexpr uint8_t DATAX1 = 0x33;
      static constexpr uint8_t DATAY0 = 0x34;
      static constexpr uint8_t DATAY1 = 0x35;
      static constexpr uint8_t DATAZ0 = 0x36;
      static constexpr uint8_t DATAZ1 = 0x37;
      static constexpr uint8_t FIFO_CTL = 0x38;
};

class ADXL345{

   public:

      int16_t raw_x;
      int16_t raw_y;
      int16_t raw_z;

      float x;
      float y;
      float z;

      int8_t offset_x;
      int8_t offset_y;
      int8_t offset_z;

      int pin;
      int number;
      int count;
      int refresh;
      int attempts;
      int spi_handle;
      int calibration_samples;

      std::vector<int16_t> x_calib;
      std::vector<int16_t> y_calib;
      std::vector<int16_t> z_calib;

      uint8_t source_content;


      uint8_t read_buffer[2];
      uint8_t write_buffer[2];
      uint8_t in_buffer[7];
      uint8_t out_buffer[7];

      static py::function python_callback;


      ADXL345(int number, int pin);
      uint8_t read_register(uint8_t register_addr);
      void write_to_register(uint8_t register_addr, uint8_t content);
      void calibrate();
      void reset_offsets();
      void set_axis();
      void disable_freefall();
      void disable_double_tap();
      void disable_act();
      void start_init();
      void startup();
      void register_callback(py::function sent_callback);
      void get_data();
      void close();

};

void calibration_callback(int gpio, int level, uint32_t tick);
void read_data(int gpio, int level, uint32_t tick);
