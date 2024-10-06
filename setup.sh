#!/bin/bash

# Update package list
sudo apt-get update

# Install system packages
sudo apt-get install -y libtool libffi-dev python3-dev build-essential

# Install Python packages
pip install -r requirements.txt
