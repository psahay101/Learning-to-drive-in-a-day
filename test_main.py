#!/usr/bin/env python3

import glob
import os
import sys

# Adjust the path as necessary for your CARLA installation
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import time

# Local imports from other modules
from env import CarEnv

def test_carla_connection():
    env = CarEnv()  # Initialize the Carla environment
    try:
        state, image = env.reset()  # Reset environment and spawn the car
        print("Connection established and vehicle spawned!")
        print("Initial state:", state)
        time.sleep(10)  # Hold the script to observe the car in the simulator

        if env.front_camera is not None:
            print("Camera is set up and capturing images.")
        else:
            print("Camera setup failed.")
    finally:
        env.cleanup()  # Ensure all actors are destroyed properly

if __name__ == '__main__':
    test_carla_connection()
