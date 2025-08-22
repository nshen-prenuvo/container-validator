#!/usr/bin/env python3
"""
FOMO25 Challenge - Direct Preprocessing Script for Brain Age Regression
Uses yucca.preprocess_case_for_inference directly on T1 and T2 .nii files
"""
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
import logging
import pickle
import os
import json
from yucca.functional.preprocessing import preprocess_case_for_inference
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments for preprocessing."""
    parser = argparse.ArgumentParser(description="FOMO25 Direct Preprocessing Pipeline for Brain Age Regression")
    
    parser.add_argument("--t1", type=str, required=True, 
                       help="Path to T1-weighted image")
    parser.add_argument("--t2", type=str, required=True, 
                       help="Path to T2-weighted image")
    parser.add_argument("--output", type=str, required=True, 
                       help="Path to output directory for preprocessed data")
    parser.add_argument("--config", type=str, default="/app/model_config.json",
                       help="Path to model configuration file")
    
    return parser.parse_args()

def load_and_validate_images(t1_path, t2_path):
    """
    Load and validate T1 and T2 input images.
    
    Args:
        t1_path: Path to T1-weighted file
        t2_path: Path to T2-weighted file
    
    Returns:
        tuple: (images list, modality_names list)
    """
    images = []
    modality_names = []
    
    # Load T1 (modality index 0)
    try:
        t1_img = nib.load(t1_path)
        images.append(t1_img)
        modality_names.append("t1")
        logger.info(f"Successfully loaded T1 image: {t1_img.shape}")
    except Exception as e:
        raise ValueError(f"Failed to load T1 image: {e}")
    
    # Load T2 (modality index 1)
    try:
        t2_img = nib.load(t2_path)
        images.append(t2_img)
        modality_names.append("t2")
        logger.info(f"Successfully loaded T2 image: {t2_img.shape}")
    except Exception as e:
        raise ValueError(f"Failed to load T2 image: {e}")
    
    # Validate that both images have the same dimensions
    if t1_img.shape != t2_img.shape:
        raise ValueError(f"T1 and T2 images must have the same dimensions. T1: {t1_img.shape}, T2: {t2_img.shape}")
    
    logger.info(f"Loaded {len(images)} modalities: {modality_names}")
    return images, modality_names

def apply_preprocessing(images, config):
    """
    Apply preprocessing using yucca.preprocess_case_for_inference.
    
    Args:
        images: List of nibabel image objects (T1 and T2)
        config: Configuration dictionary from model_config.json
    
    Returns:
        tuple: (preprocessed_data, properties)
    """
    
    # Get preprocessing parameters from config, with defaults
    preproc_config = config.get("preprocessing_config", {})
    
    target_spacing = tuple(preproc_config.get("target_spacing", [1.0, 1.0, 1.0]))
    target_orientation = preproc_config.get("target_orientation", "RAS")
    patch_size = tuple(preproc_config.get("patch_size", [96, 96, 96]))
    crop_to_nonzero = preproc_config.get("crop_to_nonzero", True)
    keep_aspect_ratio = preproc_config.get("keep_aspect_ratio", True)
    normalization = preproc_config.get("normalization", "volume_wise_znorm")

    # Normalization operations for each modality (T1 and T2)
    norm_op = [normalization] * len(images)
    
    logger.info(f"Preprocessing parameters:")
    logger.info(f"  Target spacing: {target_spacing}")
    logger.info(f"  Target orientation: {target_orientation}")
    logger.info(f"  Patch size: {patch_size}")
    logger.info(f"  Crop to nonzero: {crop_to_nonzero}")
    logger.info(f"  Keep aspect ratio: {keep_aspect_ratio}")
    logger.info(f"  Normalization: {normalization}")
    
    try:
        # Apply preprocessing
        preprocessed_data, properties = preprocess_case_for_inference(
            crop_to_nonzero=crop_to_nonzero,
            images=images,
            intensities=None,  # Use default intensity normalization
            normalization_scheme=norm_op,
            patch_size=patch_size,
            target_size=None,  # We use target_spacing instead
            target_spacing=target_spacing,
            target_orientation=target_orientation,
            allow_missing_modalities=False,
            keep_aspect_ratio=keep_aspect_ratio,
            transpose_forward=[0, 1, 2],  # Standard transpose order
        )
        
        logger.info("Preprocessing completed successfully")
        return preprocessed_data, properties
        
    except Exception as e:
        raise RuntimeError(f"Preprocessing failed: {e}")

def save_preprocessed_data(preprocessed_data, properties, output_dir, modality_names, config=None):
    """
    Save preprocessed data as .npy and .pkl files.
    
    Args:
        preprocessed_data: List of preprocessed numpy arrays (T1 and T2)
        properties: Preprocessing properties dictionary
        output_dir: Output directory path
        modality_names: List of modality names
        config: Configuration dictionary from model_config.json
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert PyTorch tensors to numpy arrays if needed
    converted_data = []
    for data in preprocessed_data:
        if hasattr(data, 'cpu') and hasattr(data, 'numpy'):
            converted_data.append(data.cpu().numpy())
        else:
            converted_data.append(data)
    
    # Save preprocessed data as .npy file
    # Stack all modalities into a single array
    data_array = np.array(converted_data, dtype=object)
    npy_path = output_path / "preprocessed_data.npy"
    np.save(npy_path, data_array)
    logger.info(f"Saved preprocessed data to {npy_path}")
    
    # Save properties as .pkl file
    pkl_path = output_path / "preprocessing_properties.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(properties, f)
    logger.info(f"Saved preprocessing properties to {pkl_path}")
    
    # Save individual modality files as nifti files for easier access
    for i, (data, modality) in enumerate(zip(converted_data, modality_names)):
        # Convert back to NIfTI format using preprocessing properties
        modality_path = output_path / f"{modality}.nii.gz"
        
        # Create NIfTI image with proper header information
        # Use the properties from preprocessing to set spacing and orientation
        if 'spacing_after_resampling' in properties:
            spacing = properties['spacing_after_resampling']
        else:
            # Get spacing from config or use default
            if config and "preprocessing_config" in config:
                spacing = tuple(config["preprocessing_config"].get("target_spacing", [1.0, 1.0, 1.0]))
            else:
                spacing = (1.0, 1.0, 1.0)  # Default to 1mm isotropic
            
        # Create affine matrix based on spacing
        affine = np.eye(4)
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1]
        affine[2, 2] = spacing[2]
        
        # Create NIfTI image
        nii_img = nib.Nifti1Image(data, affine)
        
        # Save as compressed NIfTI
        nib.save(nii_img, modality_path)
        logger.info(f"Saved {modality} modality to {modality_path}")
    
    # Save metadata about the preprocessing
    if config and "preprocessing_config" in config:
        preproc_config = config["preprocessing_config"]
        metadata = {
            "modalities": modality_names,
            "num_modalities": len(modality_names),
            "target_spacing": preproc_config.get("target_spacing", [1.0, 1.0, 1.0]),
            "crop_to_nonzero": preproc_config.get("crop_to_nonzero", True),
            "keep_aspect_ratio": preproc_config.get("keep_aspect_ratio", True),
            "normalization": preproc_config.get("normalization", "volume_wise_znorm"),
            "task": "brain_age_regression"
        }
    else:
        metadata = {
            "modalities": modality_names,
            "num_modalities": len(modality_names),
            "target_spacing": [1.0, 1.0, 1.0],
            "crop_to_nonzero": True,
            "keep_aspect_ratio": True,
            "normalization": "volume_wise_znorm",
            "task": "brain_age_regression"
        }
    
    metadata_path = output_path / "preprocessing_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"Saved preprocessing metadata to {metadata_path}")

def main():
    """Main preprocessing function for brain age regression."""
    args = parse_args()
    
    logger.info("Starting preprocessing for brain age regression task")
    logger.info(f"T1 image: {args.t1}")
    logger.info(f"T2 image: {args.t2}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Config file: {args.config}")

    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
        
    # Load and validate input images
    try:
        images, modality_names = load_and_validate_images(args.t1, args.t2)
    except Exception as e:
        logger.error(f"Failed to load images: {e}")
        sys.exit(1)
    
    # Apply preprocessing
    try:
        preprocessed_data, properties = apply_preprocessing(images, config)
    except Exception as e:
        logger.error(f"Failed to apply preprocessing: {e}")
        sys.exit(1)
    
    # Save preprocessed data
    try:
        save_preprocessed_data(
            preprocessed_data, properties, args.output, modality_names, config
        )
    except Exception as e:
        logger.error(f"Failed to save preprocessed data: {e}")
        sys.exit(1)
    
    # Save affine to file after directory is created
    try:
        affine_path = Path(args.output) / "affine.npy"
        np.save(affine_path, images[0].affine)
        logger.info(f"Saved affine matrix to {affine_path}")
    except Exception as e:
        logger.warning(f"Failed to save affine matrix: {e}")
    
    logger.info("Preprocessing completed successfully for brain age regression task")


if __name__ == "__main__":
    main() 