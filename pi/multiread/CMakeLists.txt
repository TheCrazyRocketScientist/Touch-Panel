cmake_minimum_required(VERSION 3.12)
project(adxl_module)

set(CMAKE_CXX_STANDARD 17)

# Find pybind11
find_package(pybind11 REQUIRED)

# Find pigpio using pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(PIGPIO REQUIRED pigpio)

# Create the Python module
add_library(adxl MODULE
    ADXL.cpp
    bindings.cpp
)

# Include dirs
target_include_directories(adxl PRIVATE
    ${PIGPIO_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link to libraries
target_link_libraries(adxl PRIVATE
    pybind11::module
    ${PIGPIO_LIBRARIES}
)

# Define pigpio flags
target_compile_definitions(adxl PRIVATE ${PIGPIO_CFLAGS_OTHER})

# Output as .so file (for import in Python)
set_target_properties(adxl PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)
