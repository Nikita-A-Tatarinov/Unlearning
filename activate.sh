#!/bin/bash

# Define the environment file
ENV_FILE="environment.yml"

# Extract environment name from the environment file
ENV_NAME=$(grep 'name:' $ENV_FILE | cut -d ' ' -f 2)

# Check if the conda environment exists
conda env list | grep -q "^$ENV_NAME"

if [ $? -eq 0 ]; then
  echo "Conda environment '$ENV_NAME' already exists."
  # Activate the environment
  echo "Activating conda environment '$ENV_NAME'..."
  conda activate $ENV_NAME
else
  echo "Creating conda environment '$ENV_NAME' from $ENV_FILE..."
  conda env create -f $ENV_FILE

  # Activate the environment
  echo "Activating conda environment '$ENV_NAME'..."
  conda activate $ENV_NAME

  pip install -r torch_requirements.txt
fi