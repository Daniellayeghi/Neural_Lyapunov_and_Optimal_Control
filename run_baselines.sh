#!/bin/bash

# Check for two arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <env1> <env2>"
    exit 1
fi

# Assign the environment names to variables
ENV1=$1
ENV2=$2

# Function to run a task
run_task() {
    local env=$1
    local solver=$2
    echo "Running task for environment: $env with solver: $solver"
    python3 gym_gen_baselines.py --env "$env" --solver "$solver"
}

# Main loop
for i in {1..4}; do
    # Alternate between environments and solvers
    run_task "$ENV1" "sac"
    run_task "$ENV2" "ppo"
    run_task "$ENV1" "sac"
    run_task "$ENV2" "ppo"
done

echo "All tasks completed."
