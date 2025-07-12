#!/bin/bash

sudo -E $(which python) run_sensor.py --bus 1 --addr_sel False --int_pin 4 &
sudo -E $(which python) run_sensor.py --bus 1 --addr_sel True  --int_pin 5 &
sudo -E $(which python) run_sensor.py --bus 0 --addr_sel False --int_pin 6 &
sudo -E $(which python) run_sensor.py --bus 0 --addr_sel True  --int_pin 7 &