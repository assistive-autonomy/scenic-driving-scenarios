#!/bin/bash

# Function to start Carla simulator in a new terminal and save the terminal's PID
start_carla() {
    gnome-terminal -- ~/CARLA_0.9.15/CarlaUE4.sh &
    carla_terminal_pid=$!
}

# Function to start Scenic program in a new terminal and save the terminal's PID
start_scenic() {
    local scenic_program="$1"
    gnome-terminal -- bash -c "scenic -S ./$scenic_program --count 3 ; read -p 'Press Enter to exit...'" &
    scenic_terminal_pid=$!
}

# Function to check if the Scenic terminal is still running
is_scenic_running() {
    if ps -p $scenic_terminal_pid > /dev/null; then
        return 0  # Scenic terminal is running
    else
        return 1  # Scenic terminal is not running
    fi
}

# Function to clean up all Carla instances if running
cleanup_carla() {
    carla_pids=$(pgrep carla)
    if [ -n "$carla_pids" ]; then
        echo "Terminating Carla..."
        for pid in $carla_pids; do
            kill -TERM "$pid"
            wait "$pid"
        done
        echo "All Carla instances terminated."
    fi
}

# Trap the SIGINT (Ctrl+C) signal to clean up Carla before exiting
trap cleanup_carla INT

# Check if a Scenic program argument is provided
if [ $# -eq 0 ]; then
    echo "Please provide the Scenic program name as an argument."
    exit 1
fi

# Start Carla in the first terminal and save its PID
start_carla

sleep 10

# Start Scenic in the second terminal and pass the program name as an argument
start_scenic "$1"

# Check if Scenic terminal is still running and wait for it to exit
while is_scenic_running; do
    sleep 1
done

# Clean up Carla when the Scenic terminal is closed
cleanup_carla
