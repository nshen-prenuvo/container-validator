#!/usr/bin/env python3
"""
FOMO25 Challenge - Complete Pipeline Runner
Runs preprocessing, inference, and postprocessing in sequence
"""
import argparse
import subprocess
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_preprocessing(flair_path, dwi_b1000_path, t2s_path, swi_path, output_dir):
    """Run preprocessing using base Python installation."""
    logger.info("Starting preprocessing phase...")
    
    # Check if the preprocessing script exists
    preprocess_script = Path("/app/preprocess.py")
    if not preprocess_script.exists():
        logger.error(f"Preprocessing script not found at {preprocess_script}")
        return False
    
    # Check if the base Python installation exists
    python_path = Path("/root/.pyenv/versions/3.12.6/bin/python")
    if not python_path.exists():
        logger.error(f"Python installation not found at {python_path}")
        return False
    
    # Build the preprocessing command using the base Python directly
    cmd = [
        "/root/.pyenv/versions/3.12.6/bin/python",
        "/app/preprocess.py",
        "--flair", flair_path,
        "--dwi_b1000", dwi_b1000_path,
        "--output", output_dir
    ]
    
    # Add optional modalities if provided
    if t2s_path:
        cmd.extend(["--t2s", t2s_path])
    if swi_path:
        cmd.extend(["--swi", swi_path])
    
    logger.info("Running preprocessing...")
    
    # Try to run the command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
    except subprocess.TimeoutExpired:
        logger.error("Preprocessing timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"Failed to run preprocessing subprocess: {e}")
        return False
    
    if result.returncode != 0:
        logger.error(f"Preprocessing failed with return code {result.returncode}")
        if result.stderr:
            logger.error(f"Preprocessing error: {result.stderr}")
        return False
    
    logger.info("Preprocessing completed successfully!")
    return True

def run_inference(preprocessed_dir, final_output):
    """Run inference using conda environment (fomo-condaenv)."""
    logger.info("Starting inference phase...")
    
    # Build inference command with proper conda environment activation
    cmd = [
        "bash", "-c",
        f"source /opt/conda/etc/profile.d/conda.sh && "
        f"conda activate fomo-condaenv && "
        f"python /app/inference.py "
        f"--input_dir {preprocessed_dir} "
        f"--output {final_output}"
    ]
    
    logger.info("Running inference...")
    
    # Run inference with real-time output instead of capturing it
    try:
        result = subprocess.run(cmd, timeout=600)  # 5 minute timeout, no capture_output
        
        if result.returncode != 0:
            logger.error(f"Inference failed with return code {result.returncode}")
            return False
        
        logger.info("Inference completed successfully!")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("Inference timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"Failed to run inference subprocess: {e}")
        return False

def run_postprocessing(segmentation_output, case_properties_path, final_output, affine_path,
                       num_classes=1, model_config_path="model_config.json"):
    """Run postprocessing using base Python installation."""
    logger.info("Starting postprocessing phase...")
    
    # Check if the postprocessing script exists
    postprocess_script = Path(__file__).parent / "postprocess.py"
    if not postprocess_script.exists():
        logger.error(f"Postprocessing script not found at {postprocess_script}")
        return False
    
    # Check if the base Python installation exists
    python_path = Path("/root/.pyenv/versions/3.12.6/bin/python")
    if not python_path.exists():
        logger.error(f"Python installation not found at {python_path}")
        return False
    
    # Build the postprocessing command using the base Python directly
    cmd = [
        "/root/.pyenv/versions/3.12.6/bin/python",
        str(postprocess_script),
        "--input", segmentation_output,
        "--properties_file", case_properties_path,
        "--output", final_output,
        "--num_classes", str(num_classes),
        "--model_config", model_config_path,
        "--original_affine", affine_path
    ]
    
    logger.info("Running postprocessing...")
    
    # Run postprocessing
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
    except subprocess.TimeoutExpired:
        logger.error("Postprocessing timed out after 5 minutes")
        return False
    except Exception as e:
        logger.error(f"Failed to run postprocessing subprocess: {e}")
        return False
    
    if result.returncode != 0:
        logger.error(f"Postprocessing failed with return code {result.returncode}")
        if result.stderr:
            logger.error(f"Postprocessing error: {result.stderr}")
        return False
    
    logger.info("Postprocessing completed successfully!")
    return True

def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(description="FOMO25 Complete Pipeline")
    
    parser.add_argument("--flair", type=str, required=True,
                       help="Path to T2 FLAIR image")
    parser.add_argument("--dwi_b1000", type=str, required=True,
                       help="Path to DWI b1000 image")
    parser.add_argument("--t2s", type=str, default=None,
                       help="Path to T2* image (optional, can be replaced with SWI)")
    parser.add_argument("--swi", type=str, default=None,
                       help="Path to SWI image (optional, can be replaced with T2*)")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to save final postprocessed segmentation NIfTI file")
    parser.add_argument("--temp_dir", type=str, default="/tmp/preprocessed",
                       help="Temporary directory for preprocessed data")
    parser.add_argument("--num_classes", type=int, default=1,
                       help="Number of segmentation classes (default: 1)")
    parser.add_argument("--model_config", type=str, default="/app/model_config.json",
                       help="Path to model configuration file (default: model_config.json)")
    
    
    args = parser.parse_args()
    
    # Validate that at least one of t2s or swi is provided
    if args.t2s is None and args.swi is None:
        parser.error("At least one of --t2s or --swi must be provided")
    
    # Create directories
    Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Define intermediate output paths
    affine_path = Path(args.temp_dir) / "affine.npy"
    intermediate_segmentation = Path(args.temp_dir) / "intermediate_segmentation.nii.gz"
    case_properties_path = Path(args.temp_dir) / "preprocessing_properties.pkl"
    
    # Run preprocessing
    if not run_preprocessing(args.flair, args.dwi_b1000, args.t2s, args.swi, args.temp_dir):
        logger.error("Pipeline failed at preprocessing stage")
        sys.exit(1)
    
    # Run inference
    if not run_inference(args.temp_dir, str(intermediate_segmentation)):
        logger.error("Pipeline failed at inference stage")
        sys.exit(1)
    
    # Run postprocessing
    if not run_postprocessing(str(intermediate_segmentation), str(case_properties_path), 
                             args.output, affine_path, args.num_classes, args.model_config):
        logger.error("Pipeline failed at postprocessing stage")
        sys.exit(1)
    
    logger.info("Complete pipeline finished successfully!")
    logger.info(f"Final postprocessed output saved to: {args.output}")

if __name__ == "__main__":
    main() 