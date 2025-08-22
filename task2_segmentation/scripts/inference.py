#!/usr/bin/env python3
"""
FOMO25 Challenge - Task 2: Binary Segmentation
Modular inference system using UNETR model
"""
import argparse
from re import T
import nibabel as nib
import numpy as np
from pathlib import Path
import glob
import logging
import json
import sys
import os
import torch
import torch.nn.functional as F

# Add the MIM-Med3D code path
sys.path.append('/app/MIM-Med3D/code')

from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete
from monai.data import decollate_batch
from models import UNETR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fill_nan_with_zero(array):
    """Fill NaN values with 0 in the array"""
    array = np.array(array, dtype=float)  # force numeric
    return np.nan_to_num(array, nan=0.0)


class FOMOInferenceEngine:
    """Modular inference engine for FOMO25 Task 2 segmentation"""
    
    def __init__(self, model_config: dict, device: str = 'auto'):
        """
        Initialize the inference engine
        
        Args:
            model_config: Dictionary containing model configuration
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.model_config = model_config
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load and initialize the UNETR model"""
        logger.info("Loading UNETR model...")
        
        # Initialize model with configuration
        model = UNETR(revise_keys = [("model.", "")], **self.model_config)
        logger.info("UNETR model created successfully")

        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_data(self, data_array: np.ndarray) -> torch.Tensor:
        """
        Preprocess input data for inference
        
        Args:
            data_array: Input data array from preprocessing (all modalities are images)
            
        Returns:
            Preprocessed tensor ready for model input (FLAIR modality only)
        """

        ## Only use the FLAIR modality
        flair_data = data_array[:, 1:2]  # FLAIR is at index 1
        # logger.info(f"Extracted FLAIR modality, shape: {flair_data.shape}")
        
        # Fill NaN values and convert to tensor
        flair_data = fill_nan_with_zero(flair_data)
        image_tensor = torch.from_numpy(flair_data).float()
        
        # logger.info(f"Final tensor shape: {image_tensor.shape}")
        
        return image_tensor.to(self.device)
    
    def run_inference(self, image_tensor: torch.Tensor, 
                     roi_size: tuple = (96, 96, 16),
                     sw_batch_size: int = 4,
                     overlap: float = 0.5) -> np.ndarray:
        """
        Run sliding window inference on input tensor
        
        Args:
            image_tensor: Preprocessed input tensor
            roi_size: ROI size for sliding window
            sw_batch_size: Batch size for sliding window
            overlap: Overlap ratio for sliding window
            
        Returns:
            Predicted segmentation mask as numpy array
        """
        # logger.info(f"Running inference with roi_size={roi_size}, overlap={overlap}")
        
        with torch.no_grad():
            # Run sliding window inference
            outputs = sliding_window_inference(
                image_tensor,
                roi_size,
                sw_batch_size,
                self.model,
                overlap=overlap
            )
            
            # Convert to prediction mask
            pred_masks = decollate_batch(outputs)
            
            # Get the first (and only) prediction
            pred_mask = pred_masks[0].cpu().numpy()
            
            # Convert from one-hot to binary mask
            if pred_mask.ndim == 4 and pred_mask.shape[0] > 1:
                # Take argmax to get class predictions
                pred_mask = np.argmax(pred_mask, axis=0)
            elif pred_mask.ndim == 4:
                # Single channel, remove channel dimension
                pred_mask = pred_mask[0]
            
            return pred_mask ## return the logits for downstream interpolation in postprocess.py
    
    def predict_single_subject(self, data_array: np.ndarray, properties: dict = None,
                              roi_size: tuple = (96, 96, 16),
                              sw_batch_size: int = 4,
                              overlap: float = 0.5) -> tuple:
        """
        Run complete inference pipeline for a single subject
        
        Args:
            data_array: Preprocessed data array
            properties: Properties dictionary with metadata
            roi_size: ROI size for sliding window
            sw_batch_size: Batch size for sliding window
            overlap: Overlap ratio for sliding window
            
        Returns:
            Tuple of (segmentation_mask, properties)
        """
        # Preprocess data
        image_tensor = self.preprocess_data(data_array)
        
        # Run inference
        pred_mask = self.run_inference(image_tensor, roi_size, sw_batch_size, overlap)
        
        return pred_mask, properties

def load_preprocessed_data(input_dir: str) -> tuple:
    """
    Load preprocessed data for a single subject from the new direct preprocessing script.
    
    Args:
        input_dir: Directory containing preprocessed .npy files for one subject
        
    Returns:
        Tuple of (data, properties) for the single subject
    """
    logger.info(f"Loading preprocessed data for single subject from {input_dir}")
    
    input_path = Path(input_dir)
    
    # Check if the main preprocessed data file exists
    main_data_file = input_path / "preprocessed_data.npy"
    properties_file = input_path / "preprocessing_properties.pkl"
    
    if main_data_file.exists():
        logger.info("Found main preprocessed data file")
        
        try:
            # Load the main preprocessed data
            data = np.load(main_data_file, allow_pickle=True)
            
            # Load properties
            properties = None
            if properties_file.exists():
                import pickle
                with open(properties_file, 'rb') as f:
                    properties = pickle.load(f)
                logger.info("Loaded preprocessing properties")
            
            return data, properties
            
        except Exception as e:
            logger.error(f"Failed to load main preprocessed data: {e}")
            raise
    
    raise ValueError(f"No preprocessed data found in {input_dir}")

def create_model_config(config_path: str = None) -> dict:
    """Create UNETR model configuration for FOMO25 Task 2"""
    # Default configuration
    default_config = {
        "in_channels": 1,
        "out_channels": 2,
        "img_size": [96, 96, 16],
        "feature_size": 16,
        "hidden_size": 1024,
        "mlp_dim": 4096,
        "num_layers": 8,
        "num_heads": 16,
        "pos_embed": "perceptron",
        "norm_name": "instance",
        "res_block": True,
        "conv_block": True,
        "dropout_rate": 0.0
    }
    
    # Load from JSON file if provided
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading model configuration from {config_path}")
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Update default config with loaded values
            if 'model_config' in config_data:
                model_config = config_data['model_config']
                # Convert lists to tuples for img_size
                if 'img_size' in model_config and isinstance(model_config['img_size'], list):
                    model_config['img_size'] = tuple(model_config['img_size'])
                default_config.update(model_config)
                
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}. Using default config.")
    
    return default_config

def save_segmentation(segmentation_mask: np.ndarray, 
                     properties: dict,
                     output_path: str,
                     subject_name: str = None):
    """
    Save segmentation mask as NIfTI file
    
    Args:
        segmentation_mask: Continuous-valued segmentation mask
        properties: Properties dictionary with metadata
        output_path: Output file path
        subject_name: Subject name for multi-subject outputs
    """
    # Use properties to reconstruct proper affine and header
    if properties and 'original_affine' in properties:
        affine = properties['original_affine']
    else:
        # Fallback affine
        affine = np.eye(4)
    
    # Convert to uint8
    segmentation_mask = segmentation_mask.astype(np.uint8)
    
    # Create NIfTI image
    output_img = nib.Nifti1Image(segmentation_mask, affine)
    
    # Determine output path
    if subject_name and len(subject_name) > 0:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / f"{subject_name}_segmentation.nii.gz"
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        final_path = output_path
    
    # Save segmentation mask
    nib.save(output_img, str(final_path))
    logger.info(f"Segmentation saved to {final_path}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FOMO25 Task 2 Binary Segmentation Inference")
    
    # Input directory containing preprocessed .npy files
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Path to directory containing preprocessed .npy files")
    
    # Output path for segmentation mask
    parser.add_argument("--output", type=str, required=True, 
                       help="Path to save segmentation NIfTI (or directory for multiple subjects)")
    
    # Model configuration
    parser.add_argument("--config", type=str, default="/app/model_config.json",
                       help="Path to model configuration JSON file")
    
    parser.add_argument("--device", type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help="Device to run inference on")
    
    # Inference parameters
    parser.add_argument("--roi_size", nargs=3, type=int, default=[96, 96, 16],
                       help="ROI size for sliding window inference")
    
    parser.add_argument("--overlap", type=float, default=0.5,
                       help="Overlap ratio for sliding window inference")
    
    parser.add_argument("--sw_batch_size", type=int, default=4,
                       help="Batch size for sliding window inference")
    
    return parser.parse_args()

def main():
    """Main execution function."""
    logger.info("Starting main function...")
    
    args = parse_args()
    logger.info(f"Arguments parsed: input_dir={args.input_dir}, output={args.output}")
    
    # Load preprocessed data for single subject
    logger.info("About to load preprocessed data...")
    try:
        data, properties = load_preprocessed_data(args.input_dir)
    except Exception as e:
        logger.error(f"Failed to load preprocessed data: {e}")
        sys.exit(1)
    
    if data is None:
        logger.error("No preprocessed data found")
        sys.exit(1)
    
    logger.info(f"Data loaded successfully. Data type: {type(data)}, Shape: {data.shape if hasattr(data, 'shape') else 'N/A'}")
    
    # Create model configuration
    logger.info("Creating model configuration...")
    model_config = create_model_config(args.config)
    
    # Load inference configuration if available
    inference_config = {}
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
            if 'inference_config' in config_data:
                inference_config = config_data['inference_config']
        except Exception as e:
            logger.warning(f"Failed to load inference config: {e}")
    
    # Override with command line arguments if provided
    roi_size = tuple(args.roi_size) if args.roi_size != [96, 96, 16] else tuple(inference_config.get('roi_size', [96, 96, 16]))
    overlap = args.overlap if args.overlap != 0.5 else inference_config.get('overlap', 0.5)
    sw_batch_size = args.sw_batch_size if args.sw_batch_size != 4 else inference_config.get('sw_batch_size', 4)
    device = args.device if args.device != 'auto' else inference_config.get('device', 'auto')
    
    # Initialize inference engine
    inference_engine = FOMOInferenceEngine(
        model_config=model_config,
        device=device
    )
    
    logger.info("Processing single subject")
    
    try:
        # Run inference
        logger.info("Running inference...")
        segmentation_mask, _ = inference_engine.predict_single_subject(data, properties, roi_size, sw_batch_size, overlap)
        
        # Save result
        logger.info("Saving segmentation result...")
        save_segmentation(segmentation_mask, properties, args.output)
        
        logger.info("Inference completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to do inference on the subject: {e}")
        sys.exit(1)  # Exit with error code 1

if __name__ == "__main__":
    main()