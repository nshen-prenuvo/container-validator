#!/usr/bin/env python3
"""
FOMO25 Challenge - Task 1: Classification
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
from experiments.sl.class_main import ViTClassificationHead

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fill_nan_with_zero(array):
    """Fill NaN values with 0 in the array"""
    array = np.array(array, dtype=float)  # force numeric
    return np.nan_to_num(array, nan=0.0)


class BrainClassificationModel(torch.nn.Module):
    """Complete brain classification model with ViT backbone and classification head"""
    
    def __init__(self, model_config: dict, num_classes: int = 2, dropout_rate: float = 0.1):
        """
        Initialize the brain classification model
        
        Args:
            model_config: Dictionary containing ViT backbone configuration
            num_classes: Number of classes for classification
            dropout_rate: Dropout rate for classification head
        """
        super().__init__()
        
        # Initialize ViT backbone with mean pooling
        self.backbone = ViTMeanPooling(**model_config)
        logger.info("ViT backbone created successfully")
        
        # Get hidden size from config
        hidden_size = model_config.get('hidden_size', 768)
        
        # Add classification head
        self.classifier = ViTClassificationHead(
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
        logger.info("Classification head created successfully")
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        # Initialize classification head weights
        for module in self.classifier.modules():
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
            Classification logits of shape (B, num_classes)
        """
        # Get pooled embeddings from ViT backbone
        pooled_embeddings = self.backbone(x)
        
        # Pass through classification head
        logits = self.classifier(pooled_embeddings)
        
        return logits
    
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


class FOMOClassificationInferenceEngine:
    """Modular inference engine for FOMO25 Task 1 classification"""
    
    def __init__(self, model_config: dict, num_classes: int = 2, device: str = 'auto', checkpoint_path: str = None):
        """
        Initialize the inference engine
        
        Args:
            model_config: Dictionary containing model configuration
            num_classes: Number of classes for classification
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            checkpoint_path: Path to model checkpoint file (optional)
        """
        self.model_config = model_config
        self.num_classes = num_classes
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
        """Load and initialize the brain classification model"""
        logger.info("Loading brain classification model...")
        
        # Create model instance
        model = BrainClassificationModel(
            model_config=self.model_config,
            num_classes=self.num_classes,
            dropout_rate=0.1
        )
        
        # Load checkpoint if provided
        if self.checkpoint_path:
            try:
                model.load_checkpoint(self.checkpoint_path, self.device)
                # missing_keys, unexpected_keys = model.load_checkpoint(self.checkpoint_path, self.device)
                logger.info("Model loaded with pretrained weights")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
                logger.info("Using randomly initialized weights")
        else:
            logger.info("No checkpoint provided, using randomly initialized weights")
        
        # Move model to device
        model.to(self.device)
        model.eval()
        
        logger.info("Brain classification model loaded successfully")
        return model
    
    def preprocess_data(self, data_array: np.ndarray) -> tuple:
        """
        Preprocess input data for inference
        
        Args:
            data_array: Input data array from preprocessing (FLAIR and DWI modalities)
            
        Returns:
            Tuple of (flair_tensor, dwi_tensor) ready for model input
        """
        # Extract both FLAIR and DWI modalities
        dwi_data = data_array[:, 0:1]  # DWI is at index 0
        flair_data = data_array[:, 1:2]  # FLAIR is at index 1
        logger.info(f"Extracted DWI modality, shape: {dwi_data.shape}")
        logger.info(f"Extracted FLAIR modality, shape: {flair_data.shape}")
        
        # Fill NaN values and convert to tensors
        dwi_data = fill_nan_with_zero(dwi_data)
        flair_data = fill_nan_with_zero(flair_data)
        
        dwi_tensor = torch.from_numpy(dwi_data).float()
        flair_tensor = torch.from_numpy(flair_data).float()
        
        # Add batch dimension if not present
        if dwi_tensor.dim() == 4:  # (C, H, W, D)
            dwi_tensor = dwi_tensor.unsqueeze(0)  # Add batch dimension
            flair_tensor = flair_tensor.unsqueeze(0)  # Add batch dimension
        
        logger.info(f"Final DWI tensor shape: {dwi_tensor.shape}")
        logger.info(f"Final FLAIR tensor shape: {flair_tensor.shape}")
        
        return dwi_tensor.to(self.device), flair_tensor.to(self.device)
    
    def run_inference(self, dwi_tensor: torch.Tensor, flair_tensor: torch.Tensor) -> tuple:
        """
        Run inference on both DWI and FLAIR tensors and average the predictions
        
        Args:
            dwi_tensor: Preprocessed DWI input tensor
            flair_tensor: Preprocessed FLAIR input tensor
            
        Returns:
            Tuple of (averaged_predictions, dwi_predictions, flair_predictions) as numpy arrays
        """
        logger.info("Running classification inference on both DWI and FLAIR modalities...")
        
        with torch.no_grad():
            # Forward pass through the model for DWI
            dwi_logits = self.model(dwi_tensor)
            dwi_probabilities = torch.softmax(dwi_logits, dim=1)
            dwi_predicted_class = torch.argmax(dwi_probabilities, dim=1)
            logger.info(f"DWI predictions - Class: {dwi_predicted_class.cpu().numpy()[0]}, Probabilities: {dwi_probabilities.cpu().numpy()[0]}")
            
            # Forward pass through the model for FLAIR
            flair_logits = self.model(flair_tensor)
            flair_probabilities = torch.softmax(flair_logits, dim=1)
            flair_predicted_class = torch.argmax(flair_probabilities, dim=1)
            logger.info(f"FLAIR predictions - Class: {flair_predicted_class.cpu().numpy()[0]}, Probabilities: {flair_probabilities.cpu().numpy()[0]}")
            
            # Average the probabilities from both modalities
            averaged_probabilities = (dwi_probabilities + flair_probabilities) / 2.0
            averaged_predicted_class = torch.argmax(averaged_probabilities, dim=1)
            logger.info(f"Averaged predictions - Class: {averaged_predicted_class.cpu().numpy()[0]}, Probabilities: {averaged_probabilities.cpu().numpy()[0]}")
            
            # Convert to numpy arrays
            pred_class = averaged_predicted_class.cpu().numpy()
            pred_probs = averaged_probabilities.cpu().numpy()
            dwi_probs = dwi_probabilities.cpu().numpy()
            flair_probs = flair_probabilities.cpu().numpy()
            
            return pred_class, pred_probs, dwi_probs, flair_probs
    
    def predict_single_subject(self, data_array: np.ndarray, properties: dict = None) -> tuple:
        """
        Run complete inference pipeline for a single subject
        
        Args:
            data_array: Preprocessed data array
            properties: Properties dictionary with metadata
            
        Returns:
            Tuple of (predicted_class, prediction_probabilities, properties_with_predictions)
        """
        # Preprocess data for both modalities
        dwi_tensor, flair_tensor = self.preprocess_data(data_array)
        
        # Run inference on both modalities
        pred_class, pred_probs, dwi_probs, flair_probs = self.run_inference(dwi_tensor, flair_tensor)
        
        # Add individual predictions to properties for saving
        if properties is None:
            properties = {}
        properties['dwi_predictions'] = dwi_probs
        properties['flair_predictions'] = flair_probs
        
        return pred_class, pred_probs, properties

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
    """Create ViT mean pooling model configuration for FOMO25 Task 1 classification"""
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

def save_classification_results(prediction_probabilities: np.ndarray,
                              output_path: str,
                              subject_name: str = None):
    """
    Save classification results as a single probability value
    
    Args:
        prediction_probabilities: Prediction probabilities for all classes
        output_path: Output file path
        subject_name: Subject name for multi-subject outputs
    """
    # Determine output path
    if subject_name and len(subject_name) > 0:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / f"{subject_name}_classification.txt"
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        final_path = output_path
    
    # Always output the probability of the second class (index 1)
    second_class_probability = prediction_probabilities[0, 1]
    
    # Save only the single probability value
    with open(final_path, 'w') as f:
        f.write(f"{second_class_probability:.3f}")
    
    logger.info(f"Classification probability saved to {final_path}: {second_class_probability:.3f}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FOMO25 Task 1 Classification Inference")
    
    # Input directory containing preprocessed .npy files
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Path to directory containing preprocessed .npy files")
    
    # Output path for classification results
    parser.add_argument("--output", type=str, required=True, 
                       help="Path to save classification results (or directory for multiple subjects)")
    
    # Model configuration
    parser.add_argument("--config", type=str, required=True,
                       help="Path to model configuration JSON file")
    
    # Model checkpoint
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint file (optional)")
    
    parser.add_argument("--device", type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help="Device to run inference on")
    
    # Classification parameters
    parser.add_argument("--num_classes", type=int, default=2,
                       help="Number of classes for classification")
    
    return parser.parse_args()

def main():
    """Main execution function."""
    logger.info("Starting classification inference...")
    
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
    num_classes = args.num_classes if args.num_classes != 2 else inference_config.get('num_classes', 2)
    
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
    inference_engine = FOMOClassificationInferenceEngine(
        model_config=model_config,
        num_classes=num_classes,
        device=device,
        checkpoint_path=checkpoint_path
    )
    
    logger.info("Processing single subject")
    
    try:
        # Run inference
        logger.info("Running classification inference on both DWI and FLAIR modalities...")
        predicted_class, prediction_probabilities, properties = inference_engine.predict_single_subject(data, properties)
        
        # Extract individual predictions from properties
        dwi_predictions = properties.get('dwi_predictions')
        flair_predictions = properties.get('flair_predictions')
        
        if dwi_predictions is None or flair_predictions is None:
            logger.error("Individual modality predictions not found in properties")
            sys.exit(1)
        
        # Save result
        logger.info("Saving classification results (averaged from DWI and FLAIR)...")
        save_classification_results(prediction_probabilities, args.output)
        
        logger.info("Classification inference completed successfully!")
        
        # Output the classification results to stdout for easy parsing
        predicted_class_value = predicted_class[0]
        second_class_probability = prediction_probabilities[0, 1]
        dwi_second_class_prob = dwi_predictions[0, 1]
        flair_second_class_prob = flair_predictions[0, 1]
        
        print(f"DWI Prediction - Class: {np.argmax(dwi_predictions[0])}, Probability: {dwi_second_class_prob:.3f}")
        print(f"FLAIR Prediction - Class: {np.argmax(flair_predictions[0])}, Probability: {flair_second_class_prob:.3f}")
        print(f"Averaged Prediction - Class: {predicted_class_value}, Probability: {second_class_probability:.3f}")
        
    except Exception as e:
        logger.error(f"Failed to do classification inference on the subject: {e}")
        sys.exit(1)  # Exit with error code 1

if __name__ == "__main__":
    main()