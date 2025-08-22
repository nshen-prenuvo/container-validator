#!/usr/bin/env python3
"""
Test script to demonstrate the new usage pattern for FOMO25 Classification scripts
"""
import subprocess
import sys
from pathlib import Path

def test_predict_usage():
    """Test the new predict.py usage with individual modality files for classification."""
    
    print("New usage pattern:")
    print("python predict.py \\")
    print("  --flair /input/flair.nii.gz \\")
    print("  --adc /input/adc.nii.gz \\")
    print("  --dwi_b1000 /input/dwi_b1000.nii.gz \\")
    print("  --t2s /input/t2s.nii.gz \\")  # Optional
    print("  --swi /input/swi.nii.gz \\")  # Optional
    print("  --output /output/classification_result.txt")
    print()
    
    print("Note: At least one of --t2s or --swi must be provided")
    print("Note: Model configuration will use default path: /app/model_config.json")
    print()
    
    print("Example:")
    
    print("  # With only SWI:")
    print("  python predict.py \\")
    print("    --flair /input/flair.nii.gz \\")
    print("    --adc /input/adc.nii.gz \\")
    print("    --dwi_b1000 /input/dwi_b1000.nii.gz \\")
    print("    --swi /input/swi.nii.gz \\")
    print("    --output /output/classification.txt")
    print()
    
    print("  # With only T2*:")
    print("  python predict.py \\")
    print("    --flair /input/flair.nii.gz \\")
    print("    --adc /input/adc.nii.gz \\")
    print("    --dwi_b1000 /input/dwi_b1000.nii.gz \\")
    print("    --t2s /input/t2s.nii.gz \\")
    print("    --output /output/classification.txt")
    print()
    
    print("Additional options:")
    print("  --temp_dir /tmp/preprocessed    # Temporary directory for preprocessing")
    print("  --num_classes 2                 # Number of classes (default: 2 for binary)")
    print()
    
    print("The script will:")
    print("1. Create a temporary directory structure for baseline preprocessing")
    print("2. Run preprocessing using the baseline codebase (in pyenv 3.12.6 environment)")
    print("3. Run classification inference using the ViT mean pooling model (in conda pytorch_p38 environment)")
    print("4. Clean up temporary files")
    print("5. Save the final classification probability to the specified output path")
    print()
    print("Output Format:")
    print("  - Single probability value (e.g., 0.750)")
    print("  - Represents probability of positive class (class 1)")
    print("  - For binary classification: 0.0 = negative, 1.0 = positive")
    print()
    print("Directory Structure:")
    print("  - Input files are copied to: /tmp/fomo25_preprocess/input/preprocessed/subject_001/")
    print("  - Preprocessing output goes to: /tmp/preprocessed/Task001_FOMO2/")
    print("  - Inference reads from: /tmp/preprocessed/Task001_FOMO2/")
    print("  - Final output saved to: user-specified --output path")
    print("  - Model configuration automatically loaded from: /app/model_config.json")


if __name__ == "__main__":
    print("FOMO25 Challenge - Classification Script Usage")
    print("=" * 50)
    print()
    
    test_predict_usage()
    print()
