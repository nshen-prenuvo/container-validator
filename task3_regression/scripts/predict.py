#!/usr/bin/env python3
"""
FOMO25 Challenge - Complete Regression Pipeline Runner
Runs preprocessing and inference in sequence for brain age regression task
"""
import argparse
import subprocess
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_preprocessing(t1_path, t2_path, output_dir, config_path="/app/model_config.json"):
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
        "--t1", t1_path,
        "--t2", t2_path,
        "--output", output_dir,
        "--config", config_path
    ]
    
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

def run_inference(preprocessed_dir, final_output, config_path="/app/model_config.json"):
    """Run regression inference using conda environment (fomo-condaenv)."""
    logger.info("Starting regression inference phase...")
    
    # Read checkpoint path from config file
    checkpoint_path = None
    try:
        import json
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        if 'inference_config' in config_data and 'checkpoints' in config_data['inference_config']:
            checkpoint_path = config_data['inference_config']['checkpoints']
            logger.info(f"Found checkpoint path in config: {checkpoint_path}")
    except Exception as e:
        logger.warning(f"Failed to read checkpoint path from config: {e}")
    
    # Build inference command with proper conda environment activation
    cmd = [
        "bash", "-c",
        f"source /opt/conda/etc/profile.d/conda.sh && "
        f"conda activate fomo-condaenv && "
        f"python /app/inference.py "
        f"--input_dir {preprocessed_dir} "
        f"--output {final_output} "
        f"--config {config_path}"
    ]
    
    # Add checkpoint argument if available
    if checkpoint_path:
        cmd.append(f"--checkpoint {checkpoint_path}")
    
    logger.info("Running regression inference...")
    
    # Run inference with real-time output instead of capturing it
    try:
        result = subprocess.run(cmd, timeout=600)  # 10 minute timeout, no capture_output
        
        if result.returncode != 0:
            logger.error(f"Regression inference failed with return code {result.returncode}")
            return False
        
        logger.info("Regression inference completed successfully!")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("Regression inference timed out after 10 minutes")
        return False
    except Exception as e:
        logger.error(f"Failed to run regression inference subprocess: {e}")
        return False

def main():
    """Main regression pipeline function."""
    parser = argparse.ArgumentParser(description="FOMO25 Complete Regression Pipeline for Brain Age Prediction")
    
    parser.add_argument("--t1", type=str, required=True,
                       help="Path to T1-weighted image")
    parser.add_argument("--t2", type=str, required=True,
                       help="Path to T2-weighted image")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to save output .txt file with predicted brain age")
    parser.add_argument("--temp_dir", type=str, default="/tmp/preprocessed",
                       help="Temporary directory for preprocessed data")
    parser.add_argument("--config", type=str, default="/app/model_config.json",
                       help="Path to model configuration file")
    parser.add_argument("--output_size", type=int, default=1,
                       help="Output size for regression (default: 1)")
    
    args = parser.parse_args()
    
    # Create directories
    Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Run preprocessing
    if not run_preprocessing(args.t1, args.t2, args.temp_dir, args.config):
        logger.error("Pipeline failed at preprocessing stage")
        sys.exit(1)
    
    # Run regression inference
    if not run_inference(args.temp_dir, args.output, args.config):
        logger.error("Pipeline failed at regression inference stage")
        sys.exit(1)
    
    logger.info("Complete regression pipeline finished successfully!")
    logger.info(f"Final brain age prediction saved to: {args.output}")

if __name__ == "__main__":
    main() 