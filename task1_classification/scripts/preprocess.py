#!/usr/bin/env python3
"""
FOMO25 Challenge - Direct Preprocessing Script
Uses yucca.preprocess_case_for_inference directly on input .nii files
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments for preprocessing."""
    parser = argparse.ArgumentParser(description="FOMO25 Direct Preprocessing Pipeline")
    
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
                       help="Path to output directory for preprocessed data")
    parser.add_argument("--config", type=str, default="/app/model_config.json",
                       help="Path to model configuration file")
    
    return parser.parse_args()

def load_and_validate_images(flair_path, adc_path, dwi_b1000_path, t2s_path, swi_path):
    """
    Load and validate all input images.
    
    Args:
        flair_path: Path to FLAIR file
        adc_path: Path to ADC file
        dwi_b1000_path: Path to DWI b1000 file
        t2s_path: Path to T2* file (optional)
        swi_path: Path to SWI file (optional)
    
    Returns:
        tuple: (images list, modality_names list)
    """
    images = []
    modality_names = []
    
    # Load DWI b1000 (modality index 0)
    try:
        dwi_img = nib.load(dwi_b1000_path)
        images.append(dwi_img)
        modality_names.append("dwi_b1000")
    except Exception as e:
        raise ValueError(f"Failed to load DWI b1000 image: {e}")
    
    # Load FLAIR (modality index 1)
    try:
        flair_img = nib.load(flair_path)
        images.append(flair_img)
        modality_names.append("flair")
    except Exception as e:
        raise ValueError(f"Failed to load FLAIR image: {e}")
    
    # Load ADC (modality index 2)
    try:
        adc_img = nib.load(adc_path)
        images.append(adc_img)
        modality_names.append("adc")
    except Exception as e:
        raise ValueError(f"Failed to load ADC image: {e}")
    
    # Load T2* or SWI (modality index 3)
    if t2s_path:
        try:
            t2s_img = nib.load(t2s_path)
            images.append(t2s_img)
            modality_names.append("t2s")
        except Exception as e:
            raise ValueError(f"Failed to load T2* image: {e}")
    elif swi_path:
        try:
            swi_img = nib.load(swi_path)
            images.append(swi_img)
            modality_names.append("swi")
        except Exception as e:
            raise ValueError(f"Failed to load SWI image: {e}")
    else:
        raise ValueError("Either T2* or SWI must be provided")
    
    return images, modality_names

def apply_preprocessing(images, config):
    """
    Apply preprocessing using yucca.preprocess_case_for_inference.
    
    Args:
        images: List of nibabel image objects
        config: Configuration dictionary from model_config.json
    
    Returns:
        tuple: (preprocessed_data, properties)
    """
    
    # Get preprocessing parameters from config, with defaults
    preproc_config = config["preprocessing_config"]
    
    target_spacing = tuple(preproc_config["target_spacing"])
    target_orientation = preproc_config["target_orientation"]
    patch_size = tuple(preproc_config["patch_size"])
    crop_to_nonzero = preproc_config["crop_to_nonzero"]
    keep_aspect_ratio = preproc_config["keep_aspect_ratio"]
    normalization = preproc_config["normalization"]

    # Normalization operations for each modality
    norm_op = [normalization] * len(images)
    
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
        
        return preprocessed_data, properties
        
    except Exception as e:
        raise RuntimeError(f"Preprocessing failed: {e}")

def save_preprocessed_data(preprocessed_data, properties, output_dir, modality_names, config=None):
    """
    Save preprocessed data as .npy and .pkl files.
    
    Args:
        preprocessed_data: List of preprocessed numpy arrays
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
    
    # Save properties as .pkl file
    pkl_path = output_path / "preprocessing_properties.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(properties, f)
    
    # Save individual modality files as nifti filesfor easier access
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
    
    # Save metadata about the preprocessing
    if config and "preprocessing_config" in config:
        preproc_config = config["preprocessing_config"]
        metadata = {
            "modalities": modality_names,
            "num_modalities": len(modality_names),
            "target_spacing": preproc_config.get("target_spacing", [1.0, 1.0, 1.0]),
            "crop_to_nonzero": preproc_config.get("crop_to_nonzero", True),
            "keep_aspect_ratio": preproc_config.get("keep_aspect_ratio", True),
            "normalization": preproc_config.get("normalization", "volume_wise_znorm")
        }
    else:
        metadata = {
            "modalities": modality_names,
            "num_modalities": len(modality_names),
            "target_spacing": [1.0, 1.0, 1.0],
            "crop_to_nonzero": True,
            "keep_aspect_ratio": True,
            "normalization": "volume_wise_znorm"
        }
    
    metadata_path = output_path / "preprocessing_metadata.pkl"
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

def main():
    """Main preprocessing function."""
    args = parse_args()
    
    # Validate that at least one of t2s or swi is provided
    if args.t2s is None and args.swi is None:
        parser.error("At least one of --t2s or --swi must be provided")
    

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    # Load and validate input images
    images, modality_names = load_and_validate_images(
        args.flair, args.adc, args.dwi_b1000, args.t2s, args.swi
    )
    
    # Apply preprocessing
    preprocessed_data, properties = apply_preprocessing(images, config)
    
    # Save preprocessed data
    save_preprocessed_data(
        preprocessed_data, properties, args.output, modality_names, config
    )
    
    # Save affine to file after directory is created
    affine_path = Path(args.output) / "affine.npy"
    np.save(affine_path, images[0].affine)


if __name__ == "__main__":
    main() 