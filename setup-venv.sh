#!/bin/bash

# Set environment name (optional)
ENV_NAME="mlops_env"

# Create the virtual environment
python3 -m venv $ENV_NAME

# Activate the environment
source $ENV_NAME/bin/activate

# Upgrade pip (recommended)
pip install --upgrade pip

# Install requirements
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
    echo "Requirements installed successfully."
else
    echo "requirements.txt not found."
fi
