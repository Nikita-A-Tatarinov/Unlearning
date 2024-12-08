#!/bin/bash

# Define the environment file
ENV_FILE="environment.yml"

# Extract environment name from the environment file
ENV_NAME=$(grep 'name:' $ENV_FILE | cut -d ' ' -f 2)

if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
  echo "Activating conda environment '$ENV_NAME'..."
  conda activate $ENV_NAME
fi

# Update the environment.yml file with the current environment's packages, excluding the 'prefix' line
echo "Updating dependencies in $ENV_FILE..."
conda env update --file $ENV_FILE --prune

echo "Dependencies have been updated in $ENV_FILE."