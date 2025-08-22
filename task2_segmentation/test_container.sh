#!/bin/bash

# Test script for the built task2_segmentation.sif container
# This script tests the container with fake data

set -e  # Exit on any error

echo "Testing task2_segmentation.sif container..."
echo "=========================================="

# Check if container exists
if [ ! -f "task2_segmentation.sif" ]; then
    echo "Error: Container file not found at task2_segmentation.sif"
    echo "Please build the container first using ./build_apptainer.sh"
    exit 1
fi

# Check if fake data exists
FAKE_DATA_DIR="/home/ubuntu/fomo25_challenge/pipelines/pretrain_mim_med3d/container-validator/fake_data/fomo25/fomo-task2-val/preprocessed/sub_2/ses_1"
if [ ! -d "$FAKE_DATA_DIR" ]; then
    echo "Error: Fake data directory not found at $FAKE_DATA_DIR"
    exit 1
fi

# Create output directory
mkdir -p ./output

echo "Container file: task2_segmentation.sif"
echo "Input data: $FAKE_DATA_DIR"
echo "Output directory: ./output"
echo ""

echo "Running container test..."
echo "Command: apptainer run --bind $FAKE_DATA_DIR:/input:ro --bind ./output:/output --nv task2_segmentation.sif --flair /input/flair.nii.gz --dwi_b1000 /input/dwi_b1000.nii.gz --t2s /input/t2s.nii.gz --output /output/segmentation.nii.gz"
echo ""

# Run the container
apptainer run \
    --bind "$FAKE_DATA_DIR:/input:ro" \
    --bind ./output:/output \
    --nv \
    task2_segmentation.sif \
    --flair /input/flair.nii.gz \
    --dwi_b1000 /input/dwi_b1000.nii.gz \
    --t2s /input/t2s.nii.gz \
    --output /output/segmentation.nii.gz

echo ""
echo "Test completed!"
echo "Check ./output/segmentation.nii.gz for results"
echo ""

# List output files
if [ -f "./output/segmentation.nii.gz" ]; then
    echo "✅ SUCCESS: Output file created successfully"
    ls -la ./output/
else
    echo "❌ FAILED: Output file not found"
    exit 1
fi 