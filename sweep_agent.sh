#!/bin/bash

# Check if GPU list and command arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <gpu_list> <sweep_long_id>"
    exit 1
fi

# Get the list of GPUs and command from command line arguments
GPU_LIST="$1"
SWEEP_ID="${@:2}"

# Loop through the provided GPU list and start a Wand agent on each GPU
for gpu in $GPU_LIST; do
    CUDA_VISIBLE_DEVICES=$gpu wandb agent $SWEEP_ID &
    echo "Started Wand agent on GPU $gpu"
    sleep 1 # Adjust sleep time if necessary to avoid race conditions
done

echo "All Wand agents started successfully"