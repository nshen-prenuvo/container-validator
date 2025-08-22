#!/bin/bash

# Test script for the built task3_regression.sif container
# This script tests the container with fake data

set -e  # Exit on any error

SAMPLE_SUBJECT="sub_1"
SAMPLE_SESSION="ses_1"
echo "Testing task3_regression.sif container..."
echo "========================================="

# Check if container exists
if [ ! -f "task3_regression.sif" ]; then
    echo "Error: Container file not found at task3_regression.sif"
    echo "Please build the container first using ./build_apptainer.sh"
    exit 1
fi

# Check if fake data exists
BASE_DATA_DIR="/home/ubuntu/fomo25_challenge/data/raw/finetune/fomo-task3"
if [ ! -d "$BASE_DATA_DIR" ]; then
    echo "Error: Fake data directory not found at $BASE_DATA_DIR"
    echo "Expected path: $BASE_DATA_DIR"
    echo "Please ensure the fake data directory exists for task 3 regression"
    exit 1
fi

# Create output directory
mkdir -p ./output

IMAGE_DIR=$BASE_DATA_DIR/preprocessed/$SAMPLE_SUBJECT/$SAMPLE_SESSION/
LABEL_FILE=$BASE_DATA_DIR/labels/$SAMPLE_SUBJECT/$SAMPLE_SESSION/label.txt

echo "Container file: task3_regression.sif"
echo "Input data: $IMAGE_DIR"
echo "Output directory: ./output"
echo ""

echo "Running container test..."
echo "Command: apptainer run --bind $IMAGE_DIR:/input:ro --bind ./output:/output --nv task3_regression.sif --t1 /input/t1.nii.gz --t2 /input/t2.nii.gz --output /output/brain_age_prediction.txt"
echo ""

# Run the container
apptainer run \
    --bind "$IMAGE_DIR:/input:ro" \
    --bind ./output:/output \
    --nv \
    task3_regression.sif \
    --t1 /input/t1.nii.gz \
    --t2 /input/t2.nii.gz \
    --output /output/brain_age_prediction.txt


## print the ground truth label from the label file
echo "Ground truth label:"
cat $LABEL_FILE

echo ""
echo "Test completed!"
echo "Check ./output/brain_age_prediction.txt for results"
echo ""

# List output files
if [ -f "./output/brain_age_prediction.txt" ]; then
    echo "✅ SUCCESS: Output file created successfully"
    echo "Brain age prediction result:"
    cat ./output/brain_age_prediction.txt
    echo ""
    echo "Output directory contents:"
    ls -la ./output/

    
    # Check if the prediction is a reasonable age value
    PREDICTION=$(cat ./output/brain_age_prediction.txt)
    if [[ "$PREDICTION" =~ ^[0-9]+\.?[0-9]*$ ]] && (( $(echo "$PREDICTION >= 0" | bc -l) )) && (( $(echo "$PREDICTION <= 120" | bc -l) )); then
        echo "✅ SUCCESS: Prediction value $PREDICTION is within reasonable age range (0-120 years)"
    else
        echo "⚠️  WARNING: Prediction value $PREDICTION may be outside reasonable age range"
    fi
else
    echo "❌ FAILED: Output file not found"
    exit 1
fi

echo ""
echo "Test Summary:"
echo "- Container: task3_regression.sif"
echo "- Input modalities: T1 and T2"
echo "- Output: Brain age prediction in years"
echo "- Dual modality processing: T1 and T2 predictions averaged"
echo "- Individual modality predictions logged during execution" 