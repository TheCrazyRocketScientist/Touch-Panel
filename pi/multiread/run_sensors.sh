#!/bin/bash

# --- Ensure pigpiod is running ONCE and ONLY ONCE ---
# Check if pigpiod is already running using pgrep.
# If it's not running, or if it died and left a stale PID, clean it up and start it.
if ! pgrep pigpiod > /dev/null; then
    echo "pigpiod not running. Attempting to start..."
    sudo killall -9 pigpiod 2>/dev/null # Kill any lingering processes forcefully, suppress errors
    sudo rm -f /var/run/pigpio.pid 2>/dev/null # Remove any stale PID file, suppress errors

    # Start pigpiod in the background
    sudo pigpiod
    sleep 1 # Give pigpiod a moment to start up

    # Verify pigpiod is now running
    if ! pgrep pigpiod > /dev/null; then
        echo "Error: pigpiod failed to start. Cannot proceed."
        exit 1
    else
        echo "pigpiod successfully started."
    fi
else
    echo "pigpiod is already running."
fi
# --- End pigpiod management ---


# Start all four Python processes in a single subshell in the background
# Then wait for them to finish (or be terminated)
{
    sudo -E $(which python) run_sensor.py --bus 1 --addr_sel False --int_pin 4
    sudo -E $(which python) run_sensor.py --bus 1 --addr_sel True --int_pin 5
    sudo -E $(which python) run_sensor.py --bus 0 --addr_sel False --int_pin 6
    sudo -E $(which python) run_sensor.py --bus 0 --addr_sel True --int_pin 7
} &

# Get the PID of the last background process (the subshell)
PID=$!

echo "Sensor processes started. To stop them, press Ctrl+C here or use 'kill $PID'"

# Wait for the subshell to complete (i.e., all its child processes to exit or be killed)
wait $PID

echo "All sensor processes have terminated."