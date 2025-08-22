#!/bin/bash

# FOMO25 Challenge - Task 3 Regression
# Script to apply the trained model to validation samples and collect predictions

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_DIR="$(dirname "$SCRIPT_DIR")/.."
CONTAINER_PATH="/home/ubuntu/fomo25_challenge/pipelines/pretrain_mim_med3d/container-validator/task3_regression_t2only_registered/task3_regression_t2only_registered.sif"
INPUT_DATA_DIR="/home/ubuntu/fomo25_challenge/data/raw/finetune/fomo-task3/preprocessed"
LABELS_DIR="/home/ubuntu/fomo25_challenge/data/raw/finetune/fomo-task3/labels"
OUTPUT_DIR="$SCRIPT_DIR/predictions"
PREDICTIONS_FILE="$OUTPUT_DIR/all_predictions.csv"
VAL_SUBJECTS_FILE="/home/ubuntu/fomo25_challenge/data/metadata/preprocessed_registered_t2_task3/Task003_FOMO3/val_subjects.txt"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Initialize predictions file
echo "subject_id,session_id,predicted_age,true_age,status" > "$PREDICTIONS_FILE"

# Function to process a single subject
process_subject() {
    local subject_dir="$1"
    local subject_id=$(basename "$subject_dir")
    local status="SUCCESS"
    local predicted_age=""
    local true_age=""
    
    # Find session directory
    local session_dirs=($(find "$subject_dir" -mindepth 1 -maxdepth 1 -type d -name "ses_*"))
    
    if [[ ${#session_dirs[@]} -eq 0 ]]; then
        status="NO_SESSION"
    else
        local session_dir="${session_dirs[0]}"
        local session_id=$(basename "$session_dir")
        local t2_file="$session_dir/t2.nii.gz"
        
        # Extract true age from labels
        local label_file="$LABELS_DIR/$subject_id/$session_id/label.txt"
        if [[ -f "$label_file" ]]; then
            true_age=$(cat "$label_file" | tr -d '\n\r')
        else
            true_age=""
        fi
        
        if [[ ! -f "$t2_file" ]]; then
            status="NO_T2_FILE"
        else
            # Run inference
            if apptainer run --bind "$INPUT_DATA_DIR:/input" \
                            --bind "$LABELS_DIR:/labels" \
                            --bind "$OUTPUT_DIR:/output" \
                            --nv \
                            "$CONTAINER_PATH" \
                            --t1 "/input/$(basename "$subject_dir")/$(basename "$session_dir")/t1.nii.gz" \
                            --t2 "/input/$(basename "$subject_dir")/$(basename "$session_dir")/t2.nii.gz" \
                            --output "/output/$(basename "$subject_id")_$(basename "$session_id")_prediction.txt"; then
                
                # Extract predicted age
                local temp_output="$OUTPUT_DIR/${subject_id}_${session_id}_prediction.txt"
                if [[ -f "$temp_output" ]]; then
                    # The output file contains just the predicted age number
                    predicted_age=$(cat "$temp_output" | tr -d '\n\r' | grep -oP '^[0-9.]+$' || echo "")
                    if [[ -z "$predicted_age" ]]; then
                        status="NO_PREDICTION"
                    fi
                    # Clean up the temporary file
                    rm -f "$temp_output"
                else
                    status="NO_OUTPUT"
                fi
            else
                status="INFERENCE_FAILED"
            fi
        fi
    fi
    
    # Write to predictions file
    echo "$subject_id,$session_id,$predicted_age,$true_age,$status" >> "$PREDICTIONS_FILE"
}

# Check if validation subjects file exists
if [[ ! -f "$VAL_SUBJECTS_FILE" ]]; then
    echo "Error: Validation subjects file not found: $VAL_SUBJECTS_FILE"
    exit 1
fi

# Read validation subjects from file
echo "Reading validation subjects from: $VAL_SUBJECTS_FILE"
val_subjects=($(cat "$VAL_SUBJECTS_FILE" | grep -v '^$' | sort))

if [[ ${#val_subjects[@]} -eq 0 ]]; then
    echo "No validation subjects found in file"
    exit 1
fi

echo "Found ${#val_subjects[@]} validation subjects"

# Main processing - only process validation subjects
echo "Processing validation samples..."
for subject_id in "${val_subjects[@]}"; do
    subject_dir="$INPUT_DATA_DIR/$subject_id"
    
    if [[ ! -d "$subject_dir" ]]; then
        echo "Warning: Subject directory not found: $subject_dir"
        # Try to get true age even if subject directory not found
        local true_age=""
        local session_id=""
        if [[ -d "$LABELS_DIR/$subject_id" ]]; then
            local session_dirs=($(find "$LABELS_DIR/$subject_id" -mindepth 1 -maxdepth 1 -type d -name "ses_*"))
            if [[ ${#session_dirs[@]} -gt 0 ]]; then
                session_id=$(basename "${session_dirs[0]}")
                local label_file="$LABELS_DIR/$subject_id/$session_id/label.txt"
                if [[ -f "$label_file" ]]; then
                    true_age=$(cat "$label_file" | tr -d '\n\r')
                fi
            fi
        fi
        echo "$subject_id,$session_id,,$true_age,SUBJECT_NOT_FOUND" >> "$PREDICTIONS_FILE"
        continue
    fi
    
    process_subject "$subject_dir"
done

echo "Completed! Results saved to: $PREDICTIONS_FILE"
