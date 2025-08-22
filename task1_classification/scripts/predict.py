#!/usr/bin/env python3
"""
FOMO25 Challenge - Complete Classification Pipeline Runner
Runs preprocessing and inference in sequence for classification task
"""
import argparse
import subprocess
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_preprocessing(flair_path, adc_path, dwi_b1000_path, t2s_path, swi_path, output_dir, config_path="/app/model_config.json"):
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
        "--adc", adc_path,
        "--dwi_b1000", dwi_b1000_path,
        "--output", output_dir,
        "--config", config_path
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

def run_inference(preprocessed_dir, final_output, config_path="/app/model_config.json"):
    """Run classification inference using conda environment (fomo-condaenv)."""
    logger.info("Starting classification inference phase...")
    
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
    
    logger.info("Running classification inference...")
    
    # Run inference with real-time output instead of capturing it
    try:
        result = subprocess.run(cmd, timeout=600)  # 10 minute timeout, no capture_output
        
        if result.returncode != 0:
            logger.error(f"Classification inference failed with return code {result.returncode}")
            return False
        
        logger.info("Classification inference completed successfully!")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error("Classification inference timed out after 10 minutes")
        return False
    except Exception as e:
        logger.error(f"Failed to run classification inference subprocess: {e}")
        return False

def main():
    """Main classification pipeline function."""
    parser = argparse.ArgumentParser(description="FOMO25 Complete Classification Pipeline")
    
    parser.add_argument("--flair", type=str, required=True,
                       help="Path to T2 FLAIR image")
    parser.add_argument("--adc", type=str, required=True,
                       help="Path to ADC image")
    parser.add_argument("--dwi_b1000", type=str, required=True,
                       help="Path to DWI b1000 image")
    parser.add_argument("--t2s", type=str, default=None,
                       help="Path to T2* image (optional, can be replaced with SWI)")
    parser.add_argument("--swi", type=str, default=None,
                       help="Path to SWI image (optional, can be replaced with T2*)")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to save output .txt file with probability")
    parser.add_argument("--temp_dir", type=str, default="/tmp/preprocessed",
                       help="Temporary directory for preprocessed data")
    parser.add_argument("--config", type=str, default="/app/model_config.json",
                       help="Path to model configuration file")
    parser.add_argument("--num_classes", type=int, default=2,
                       help="Number of classes for classification (default: 2)")
    
    args = parser.parse_args()
    
    # Validate that at least one of t2s or swi is provided
    if args.t2s is None and args.swi is None:
        parser.error("At least one of --t2s or --swi must be provided")
    
    # Create directories
    Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Run preprocessing
    if not run_preprocessing(args.flair, args.adc, args.dwi_b1000, args.t2s, args.swi, args.temp_dir, args.config):
        logger.error("Pipeline failed at preprocessing stage")
        sys.exit(1)
    
    # Run classification inference
    if not run_inference(args.temp_dir, args.output, args.config):
        logger.error("Pipeline failed at classification inference stage")
        sys.exit(1)
    
    logger.info("Complete classification pipeline finished successfully!")
    logger.info(f"Final classification probability saved to: {args.output}")

if __name__ == "__main__":
    main() 