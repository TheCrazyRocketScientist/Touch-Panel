cmake_minimum_required(VERSION 3.12)
project(adxl_module)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # Needed for pybind11 shared libs

# Find pybind11
find_package(pybind11 REQUIRED)

# Define module target
pybind11_add_module(adxl 
    ADXL.cpp
    bindings.cpp
)

# Ensure pigpio is linked
target_link_libraries(adxl PRIVATE
    pigpio
)

# Include dirs (if needed — optional here because pybind11 handles most)
target_include_directories(adxl PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Force output as .so (important for Python module)
set_target_properties(adxl PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)
