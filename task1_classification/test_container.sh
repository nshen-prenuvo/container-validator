#!/bin/bash

# Test script for the built task1_classification.sif container
# This script tests the container with fake data

set -e  # Exit on any error

echo "Testing task1_classification.sif container..."
echo "============================================="

# Check if container exists
if [ ! -f "task1_classification.sif" ]; then
    echo "Error: Container file not found at task1_classification.sif"
    echo "Please build the container first using ./build_apptainer.sh"
    exit 1
fi

# Check if fake data exists
FAKE_DATA_DIR="/home/ubuntu/fomo25_challenge/pipelines/pretrain_mim_med3d/container-validator/fake_data/fomo25/fomo-task1-val/preprocessed/sub_2/ses_1"
if [ ! -d "$FAKE_DATA_DIR" ]; then
    echo "Error: Fake data directory not found at $FAKE_DATA_DIR"
    exit 1
fi

# Create output directory
mkdir -p ./output

echo "Container file: task1_classification.sif"
echo "Input data: $FAKE_DATA_DIR"
echo "Output directory: ./output"
echo ""

echo "Running container test..."
echo "Command: apptainer run --bind $FAKE_DATA_DIR:/input:ro --bind ./output:/output --nv task1_classification.sif --flair /input/flair.nii.gz --adc /input/adc.nii.gz --dwi_b1000 /input/dwi_b1000.nii.gz --t2s /input/t2s.nii.gz --output /output/classification_result.txt"
echo ""

# Run the container
apptainer run \
    --bind "$FAKE_DATA_DIR:/input:ro" \
    --bind ./output:/output \
    --nv \
    task1_classification.sif \
    --flair /input/flair.nii.gz \
    --adc /input/adc.nii.gz \
    --dwi_b1000 /input/dwi_b1000.nii.gz \
    --swi /input/swi.nii.gz \
    --output /output/classification_result.txt

echo ""
echo "Test completed!"
echo "Check ./output/classification_result.txt for results"
echo ""

# List output files
if [ -f "./output/classification_result.txt" ]; then
    echo "✅ SUCCESS: Output file created successfully"
    echo "Classification result:"
    cat ./output/classification_result.txt
    echo ""
    ls -la ./output/
else
    echo "❌ FAILED: Output file not found"
    exit 1
fi 