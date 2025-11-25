#!/bin/bash

# Usage:
# ./run_pipeline.sh <dataset> <device> <method>
# Example:
# ./run_pipeline.sh ohsumed 0 Chunk

# -------------------------
# Parameter check
# -------------------------
if [ $# -lt 3 ]; then
    echo "Usage: $0 <dataset> <device> <method>"
    echo "  <dataset>: Dataset name (e.g., R8, ohsumed)"
    echo "  <device>: GPU device index"
    echo "  <method>: Graph construction method (e.g., Chunk)"
    exit 1
fi

dataset=$1
device=$2
method=$3

# -------------------------
# Environment setup
# -------------------------
# Clear any external library paths to avoid conflicts
unset LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(conda info --base)/envs/t2g_new/lib

# -------------------------
# Graph building log directory
# -------------------------
build_log_dir="./result_build_graph"
mkdir -p "${build_log_dir}"
build_log_file="${build_log_dir}/${dataset}_${method}.log"

# -------------------------
# Step 1: Build graph
# -------------------------
echo "Step 1: Building graph for dataset ${dataset} using method ${method} on device ${device} ..."
nohup python -u build_graph.py \
    --dataset "${dataset}" \
    --method "${method}" \
    --weighted \
    --device "${device}" \
    > "${build_log_file}" 2>&1

echo "Graph build launched. Logs are saved at ${build_log_file}"

# -------------------------
# Training log directory
# -------------------------
train_log_dir="./result_train/${dataset}"
mkdir -p "${train_log_dir}"
train_log_file="${train_log_dir}/${dataset}_${method}.log"

# -------------------------
# Step 2: Train model
# -------------------------
echo "Step 2: Launching training on device ${device} ..."
input_dim=768
epochs=200
early_stopping=100
edges='coreference,window,same,syntax,self'
model='g2'
shuffle=true
use_g2=true

nohup python -u train.py \
    --dataset "${dataset}" \
    --device "${device}" \
    --method "${method}" \
    --model "${model}" \
    --edges "${edges}" \
    --input_dim "${input_dim}" \
    --epochs "${epochs}" \
    --early_stopping "${early_stopping}" \
    --shuffle "${shuffle}" \
    --use_g2 "${use_g2}" \
    > "${train_log_file}" 2>&1 &

echo "Training launched. Logs are saved at ${train_log_file}"
