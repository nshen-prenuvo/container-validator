#!/usr/bin/env python3
"""
Test script to demonstrate the new usage pattern for FOMO25 scripts
"""
import subprocess
import sys
from pathlib import Path

def test_predict_usage():
    """Test the new predict.py usage with individual modality files."""
    
    print("New usage pattern:")
    print("python predict.py \\")
    print("  --flair /input/flair.nii.gz \\")
    print("  --dwi_b1000 /input/dwi_b1000.nii.gz \\")
    print("  --t2s /input/t2s.nii.gz \\")  # Optional
    print("  --swi /input/swi.nii.gz \\")  # Optional
    print("  --output /output/output.nii.gz")
    print()
    
    print("Note: At least one of --t2s or --swi must be provided")
    print()
    
    print("Example:")
    
    # print("  # With both T2* and SWI:")
    print("  python predict.py \\")
    print("    --flair /input/flair.nii.gz \\")
    print("    --dwi_b1000 /input/dwi_b1000.nii.gz \\")
    print("    --t2s /input/t2s.nii.gz \\")
    print("    --swi /input/swi.nii.gz \\")
    print("    --output /output/seg.nii.gz")
    print()
    
    print("Additional options:")
    print("  --temp_dir /tmp/preprocessed    # Temporary directory for preprocessing")
    print("  --taskid 2                     # Task ID (default: 2 for segmentation)")
    print("  --num_workers 4                # Number of parallel workers")
    print("  --skip_preprocessing           # Skip preprocessing and run inference only")
    print()
    
    print("The script will:")
    print("1. Create a temporary directory structure for baseline preprocessing")
    print("2. Run preprocessing using the baseline codebase (in pyenv fomo25-3.12 environment)")
    print("3. Run inference using the UNETR model (in conda pytorch_p38 environment)")
    print("4. Clean up temporary files")
    print("5. Save the final segmentation to the specified output path")
    print()
    print("Directory Structure:")
    print("  - Input files are copied to: /tmp/fomo25_preprocess/input/preprocessed/subject_001/")
    print("  - Preprocessing output goes to: /tmp/preprocessed/Task002_FOMO2/")
    print("  - Inference reads from: /tmp/preprocessed/Task002_FOMO2/")
    print("  - Final output saved to: user-specified --output path")


if __name__ == "__main__":
    print("FOMO25 Challenge - Updated Script Usage")
    print("=" * 50)
    print()
    
    test_predict_usage()
    print()
