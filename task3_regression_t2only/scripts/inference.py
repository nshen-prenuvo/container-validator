#!/usr/bin/env python3
"""
FOMO25 Challenge - Task 3: Regression
Modular inference system using ViT mean pooling model
"""
import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
import logging
import json
import sys
import os
import torch
import torch.nn.functional as F

# Add the MIM-Med3D code path
sys.path.append('/app/MIM-Med3D/code')

from models.vit_pooling import ViTMeanPooling
from experiments.sl.reg_main import ViTRegressionHead

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fill_nan_with_zero(array):
    """Fill NaN values with 0 in the array"""
    array = np.array(array, dtype=float)  # force numeric
    return np.nan_to_num(array, nan=0.0)


class BrainAgeModel(torch.nn.Module):
    """Complete brain age prediction model with ViT backbone and regression head"""
    
    def __init__(self, model_config: dict, output_size: int = 1, dropout_rate: float = 0.1):
        """
        Initialize the brain age prediction model
        
        Args:
            model_config: Dictionary containing ViT backbone configuration
            output_size: Output size for regression (default: 1)
            dropout_rate: Dropout rate for regression head
        """
        super().__init__()
        
        # Initialize ViT backbone with mean pooling
        self.backbone = ViTMeanPooling(**model_config)
        logger.info("ViT backbone created successfully")
        
        # Get hidden size from config
        hidden_size = model_config.get('hidden_size', 768)
        
        # Add regression head
        self.regressor = ViTRegressionHead(
            hidden_size=hidden_size,
            output_size=output_size,
            dropout_rate=dropout_rate
        )
        logger.info("Regression head created successfully")
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize regression head weights
        for module in self.regressor.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        logger.info("Model weights initialized")
    
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (B, C, H, W, D)
            
        Returns:
            Regression output of shape (B, output_size)
        """
        # Get pooled embeddings from ViT backbone
        pooled_embeddings = self.backbone(x)
        
        # Pass through regression head
        predictions = self.regressor(pooled_embeddings)
        
        return predictions
    
    def load_checkpoint(self, checkpoint_path: str, device: str = 'cpu'):
        """
        Load model weights from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load weights on
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load state dict
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
        
        logger.info("Checkpoint loaded successfully")
        
        return missing_keys, unexpected_keys


class FOMORegressionInferenceEngine:
    """Modular inference engine for FOMO25 Task 3 regression using T2 modality only"""
    
    def __init__(self, model_config: dict, checkpoint_path: str, output_size: int = 1, device: str = 'auto'):
        """
        Initialize the inference engine
        
        Args:
            model_config: Dictionary containing model configuration
            output_size: Output size for regression (default: 1)
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            checkpoint_path: Path to model checkpoint file (optional)
        """
        self.model_config = model_config
        self.output_size = output_size
        self.checkpoint_path = checkpoint_path
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load and initialize the brain age prediction model"""
        logger.info("Loading brain age prediction model...")
        
        # Create model instance
        model = BrainAgeModel(
            model_config=self.model_config,
            output_size=self.output_size,
            dropout_rate=0.1
        )
        

        missing_keys, unexpected_keys = model.load_checkpoint(self.checkpoint_path, self.device)
        logger.info("Model loaded with pretrained weights")

        # Move model to device
        model.to(self.device)
        model.eval()
        
        logger.info("Brain age prediction model loaded successfully")
        return model
    
    def preprocess_data(self, data_array: np.ndarray) -> torch.Tensor:
        """
        Preprocess input data for inference
        
        Args:
            data_array: Input data array from preprocessing (T2 modality only)
            
        Returns:
            T2 tensor ready for model input
        """
        # Extract only T2 modality
        t2_data = data_array[:, 1:2]  # T2 is at index 1
        logger.info(f"Extracted T2 modality, shape: {t2_data.shape}")
        
        # Fill NaN values and convert to tensor
        t2_data = fill_nan_with_zero(t2_data)
        
        t2_tensor = torch.from_numpy(t2_data).float()
        
        # Add batch dimension if not present
        if t2_tensor.dim() == 4:  # (C, H, W, D)
            t2_tensor = t2_tensor.unsqueeze(0)  # Add batch dimension
        
        logger.info(f"Final T2 tensor shape: {t2_tensor.shape}")
        
        return t2_tensor.to(self.device)
    
    def run_inference(self, t2_tensor: torch.Tensor) -> np.ndarray:
        """
        Run inference on T2 tensor
        
        Args:
            t2_tensor: Preprocessed T2 input tensor
            
        Returns:
            T2 predictions as numpy array
        """
        logger.info("Running regression inference on T2 modality...")
        
        with torch.no_grad():
            # Forward pass through the model for T2
            t2_predictions = self.model(t2_tensor)
            logger.info(f"T2 predictions: {t2_predictions.cpu().numpy()[0]}")
            
            # Convert to numpy array
            pred_values = t2_predictions.cpu().numpy()
            
            return pred_values
    
    def predict_single_subject(self, data_array: np.ndarray, properties: dict = None) -> tuple:
        """
        Run complete inference pipeline for a single subject
        
        Args:
            data_array: Preprocessed data array
            properties: Properties dictionary with metadata
            
        Returns:
            Tuple of (regression_predictions, properties_with_predictions)
        """
        # Preprocess data for T2 modality only
        t2_tensor = self.preprocess_data(data_array)
        
        # Run inference on T2 modality
        pred_values = self.run_inference(t2_tensor)
        
        # Add T2 predictions to properties for saving
        if properties is None:
            properties = {}
        properties['t2_predictions'] = pred_values
        
        return pred_values, properties

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

def create_model_config(config_path: str) -> dict:
    """Create ViT mean pooling model configuration for FOMO25 Task 3 regression"""
    if not config_path or not os.path.exists(config_path):
        raise ValueError(f"Model configuration file not found: {config_path}")
    
    logger.info(f"Loading model configuration from {config_path}")
    try:
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Extract model configuration
        if 'model_config' in config_data:
            model_config = config_data['model_config']
            # Convert lists to tuples for img_size if present
            if 'img_size' in model_config and isinstance(model_config['img_size'], list):
                model_config['img_size'] = tuple(model_config['img_size'])
            return model_config
        else:
            raise ValueError("No 'model_config' section found in configuration file")
            
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        raise

def save_regression_results(regression_predictions: np.ndarray,
                           t2_predictions: np.ndarray,
                           output_path: str,
                           subject_name: str = None):
    """
    Save regression results including T2 predictions and brain age prediction
    
    Args:
        regression_predictions: Regression predictions (brain age) from T2 modality
        t2_predictions: T2 modality predictions
        output_path: Output file path
        subject_name: Subject name for multi-subject outputs
    """
    # Determine output path
    if subject_name and len(subject_name) > 0:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / f"{subject_name}_brain_age.txt"
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        final_path = output_path
    
    # Get the brain age prediction (first prediction)
    brain_age = regression_predictions[0, 0]
    t2_pred = t2_predictions[0, 0]
    
    # Save only the T2-based brain age prediction
    with open(final_path, 'w') as f:
        f.write(f"{brain_age:.1f}")
    
    # Log all predictions for monitoring/debugging
    logger.info(f"Brain age prediction (T2-based) saved to {final_path}: {brain_age:.1f} years")
    logger.info(f"T2 prediction: {t2_pred:.1f} years")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FOMO25 Task 3 Brain Age Regression Inference")
    
    # Input directory containing preprocessed .npy files
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Path to directory containing preprocessed .npy files")
    
    # Output path for regression results
    parser.add_argument("--output", type=str, required=True, 
                       help="Path to save brain age prediction results (or directory for multiple subjects)")
    
    # Model configuration
    parser.add_argument("--config", type=str, required=True,
                       help="Path to model configuration JSON file")
    
    # Model checkpoint
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint file (optional)")
    
    parser.add_argument("--device", type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help="Device to run inference on")
    
    # Regression parameters
    parser.add_argument("--output_size", type=int, default=1,
                       help="Output size for regression (default: 1)")
    
    return parser.parse_args()

def main():
    """Main execution function."""
    logger.info("Starting brain age regression inference...")
    
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
    device = args.device if args.device != 'auto' else inference_config.get('device', 'auto')
    output_size = args.output_size if args.output_size != 1 else inference_config.get('output_size', 1)
    
    # Get checkpoint path from config or command line
    checkpoint_path = args.checkpoint if args.checkpoint else inference_config.get('checkpoints')
    
    # Log checkpoint information
    if checkpoint_path:
        logger.info(f"Using checkpoint: {checkpoint_path}")
        # Check if checkpoint file exists
        if not os.path.exists(checkpoint_path):
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            sys.exit(1)
    else:
        logger.warning("No checkpoint provided - model will use randomly initialized weights")
    
    # Initialize inference engine
    inference_engine = FOMORegressionInferenceEngine(
        model_config=model_config,
        output_size=output_size,
        device=device,
        checkpoint_path=checkpoint_path
    )
    
    logger.info("Processing single subject")
    
    try:
        # Run inference
        logger.info("Running brain age regression inference on T2 modality...")
        regression_predictions, properties = inference_engine.predict_single_subject(data, properties)
        
        # Extract individual predictions from properties
        t2_predictions = properties.get('t2_predictions')
        
        if t2_predictions is None:
            logger.error("T2 modality predictions not found in properties")
            sys.exit(1)
        
        # Save result
        logger.info("Saving brain age prediction results (from T2 modality)...")
        save_regression_results(regression_predictions, t2_predictions, args.output)
        
        logger.info("Brain age regression inference completed successfully!")
        
        # Output the brain age prediction to stdout for easy parsing
        brain_age = regression_predictions[0, 0]
        t2_pred = t2_predictions[0, 0]
        
        print(f"T2 Prediction: {t2_pred:.1f}")
        print(f"Brain Age Prediction: {brain_age:.1f}")
        
    except Exception as e:
        logger.error(f"Failed to do brain age regression inference on the subject: {e}")
        sys.exit(1)  # Exit with error code 1

if __name__ == "__main__":
    main()