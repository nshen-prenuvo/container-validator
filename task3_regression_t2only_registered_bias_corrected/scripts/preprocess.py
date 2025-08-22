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
import ants

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments for preprocessing."""
    parser = argparse.ArgumentParser(description="FOMO25 Direct Preprocessing Pipeline for Brain Age Regression")
    
    parser.add_argument("--t1", type=str, required=True, 
                       help="Path to T1-weighted image (required for compatibility but not used)")
    parser.add_argument("--t2", type=str, required=True, 
                       help="Path to T2-weighted image")
    parser.add_argument("--output", type=str, required=True, 
                       help="Path to output directory for preprocessed data")
    parser.add_argument("--config", type=str, default="/app/model_config.json",
                       help="Path to model configuration file")
    
    return parser.parse_args()

def load_and_validate_images(t2_path):
    """
    Load and validate T2 input image (after registration).
    
    Args:
        t2_path: Path to registered T2-weighted file
    
    Returns:
        tuple: (images list, modality_names list)
    """
    images = []
    modality_names = []
    
    # Load T2 (modality index 0)
    try:
        t2_img = nib.load(t2_path)
        images.append(t2_img)
        modality_names.append("t2")
        logger.info(f"Successfully loaded registered T2 image: {t2_img.shape}")
    except Exception as e:
        raise ValueError(f"Failed to load registered T2 image: {e}")
    
    logger.info(f"Loaded {len(images)} modalities: {modality_names}")
    return images, modality_names

def apply_preprocessing(images, config):
    """
    Apply preprocessing using yucca.preprocess_case_for_inference.
    
    Args:
        images: List of nibabel image objects (T2 only)
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

    # Normalization operations for T2 modality only
    norm_op = [normalization]
    
    logger.info(f"Preprocessing parameters:")
    logger.info(f"  Target spacing: {target_spacing}")
    logger.info(f"  Target orientation: {target_orientation}")
    logger.info(f"  Patch size: {patch_size}")
    logger.info(f"  Crop to nonzero: {crop_to_nonzero}")
    logger.info(f"  Keep aspect ratio: {keep_aspect_ratio}")
    logger.info(f"  Normalization: {normalization}")
    logger.info(f"  Processing T2 modality only")
    
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
        preprocessed_data: List of preprocessed numpy arrays (T2 only)
        properties: Preprocessing properties dictionary
        output_dir: Output directory path
        modality_names: List of modality names (T2 only)
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
            "task": "brain_age_regression_t2only"
        }
    else:
        metadata = {
            "modalities": modality_names,
            "num_modalities": len(modality_names),
            "target_spacing": [1.0, 1.0, 1.0],
            "crop_to_nonzero": True,
            "keep_aspect_ratio": True,
            "normalization": "volume_wise_znorm",
            "task": "brain_age_regression_t2only"
        }
    
    metadata_path = output_path / "preprocessing_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"Saved preprocessing metadata to {metadata_path}")

def register_t2_image(t2_path, output_dir):
    """
    Register T2 image to MNI template using ANTs.
    
    Args:
        t2_path: Path to T2-weighted image
        output_dir: Output directory for intermediate files
    
    Returns:
        str: Path to registered T2 image
    """
    logger.info("Starting T2 registration to MNI template")
    
    # Define template paths (same as in reference script)
    template_path = '/app/mni_t2_template/images/mni_icbm152_t2_tal_nlin_sym_09c.nii.gz'
    template_mask_path = '/app/mni_t2_template/labels/mni_icbm152_t1_tal_nlin_sym_09c_mask.nii.gz'
    
    # Check if template files exist
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template file not found: {template_path}")
    if not os.path.exists(template_mask_path):
        raise FileNotFoundError(f"Template mask file not found: {template_mask_path}")
    
    try:
        # Load template and mask
        template = ants.image_read(template_path)
        template_mask = ants.image_read(template_mask_path)
        
        # Apply mask to template
        template = template * template_mask
        
        # Load T2 image
        t2_image = ants.image_read(t2_path)
        
        logger.info("Running ANTs registration (SyN transform)")
        
        # Run registration
        mytx = ants.registration(fixed=template, moving=t2_image, type_of_transform='SyN')
        
        # Apply transforms
        registered_image = ants.apply_transforms(
            fixed=template, 
            moving=t2_image, 
            transformlist=mytx['fwdtransforms'], 
            interpolator='linear'
        )
        
        # Remove background noise
        dilated_mask = ants.morphology(template_mask, operation='dilate', radius=3, mtype='binary')
        registered_image = registered_image * dilated_mask
        
        # Save registered image
        registered_path = os.path.join(output_dir, "t2_registered.nii.gz")
        ants.image_write(registered_image, registered_path)
        
        logger.info(f"T2 registration completed. Saved to: {registered_path}")
        return registered_path
        
    except Exception as e:
        raise RuntimeError(f"T2 registration failed: {e}")

def remove_background_noise(ants_image, ants_mask):
    """Remove background noise using dilated mask."""
    dilated_mask = ants.morphology(ants_mask, operation='dilate', radius=3, mtype='binary')
    image_in_mask = ants_image * dilated_mask
    return image_in_mask

def main():
    """Main preprocessing function for brain age regression."""
    args = parse_args()
    
    logger.info("Starting preprocessing for brain age regression task")
    logger.info(f"T1 image: {args.t1} (not used in processing)")
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
    
    # Create intermediate directory for registered T2
    intermediate_dir = os.path.join(args.output, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)
    
    # Register T2 image to MNI template
    try:
        registered_t2_path = register_t2_image(args.t2, intermediate_dir)
        logger.info(f"T2 registration completed: {registered_t2_path}")
    except Exception as e:
        logger.error(f"Failed to register T2 image: {e}")
        sys.exit(1)
        
    # Load and validate input images (now using registered T2 only)
    try:
        images, modality_names = load_and_validate_images(registered_t2_path)
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