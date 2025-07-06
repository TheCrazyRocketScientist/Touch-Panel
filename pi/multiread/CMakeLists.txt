cmake_minimum_required(VERSION 3.12)
project(adxl_module)

set(CMAKE_CXX_STANDARD 17)

# Find pybind11
find_package(pybind11 REQUIRED)

# Find pigpio using pkg-config

# Create the Python module
add_library(adxl MODULE
    ADXL.cpp
    bindings.cpp
)

# Include dirs
target_include_directories(adxl PRIVATE
    /usr/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link to pigpio directly
target_link_libraries(adxl PRIVATE
    pybind11::module
    pigpio
)


# Output as .so file (for import in Python)
set_target_properties(adxl PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)
