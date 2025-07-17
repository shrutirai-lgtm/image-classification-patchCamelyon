#!/bin/bash

# Exit on error
set -e

# Create virtual environment with Python 3.9
python3.9 -m venv venv

echo "Virtual environment created."

# Activate the virtual environment
source venv/bin/activate

echo "Virtual environment activated."

# Upgrade pip
pip install --upgrade pip

echo "pip upgraded."

# Install dependencies
pip install -r requirements.txt

echo "Dependencies installed."

echo "Setup complete! To activate the environment later, run: source venv/bin/activate" 