#!/bin/bash

# Check if the Python script name is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <python_script.py>"
    exit 1
fi

# Assign the first argument to a variable
python_script="$1"

# Loop from 0 to 4
for seed in {0..4}
do
    echo "Running $python_script with seed $seed"
    python3 "$python_script" --seed $seed
done
