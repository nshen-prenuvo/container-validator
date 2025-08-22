# Multi-Environment Apptainer Container for FOMO25 Segmentation

This Apptainer container supports both preprocessing and inference stages using different Python environments:

- **Preprocessing**: Python 3.12.0 via pyenv with environment name `fomo25-3.12`
- **Inference**: Python 3.8 via conda with environment name `pytorch_p38`

## Architecture

The container uses a hybrid approach:
1. **pyenv** for preprocessing environment (Python 3.12.0, environment: `fomo25-3.12`)
2. **Miniconda** for inference environment (Python 3.8, environment: `pytorch_p38`)

This allows you to use different Python versions and dependency sets for different stages of your pipeline.

## Files

- `Apptainer.def`: Main container definition
- `preprocess.py`: Preprocessing script (runs in pyenv environment)
- `predict.py`: Inference script (runs in conda environment)
- `run_pipeline.py`: Complete pipeline runner
- `preprocess_requirements.txt`: Dependencies for preprocessing
- `environment.yml`: Conda environment for inference
- `requirements.txt`: Original requirements (kept for compatibility)

## Building the Container

```bash
apptainer build --fakeroot segmentation.sif Apptainer.def
```

## Usage Options

### 1. Complete Pipeline (Recommended)

Run both preprocessing and inference in sequence:

```bash
apptainer run --bind /path/to/input:/input:ro \
    --bind /path/to/output:/output \
    --nv \
    segmentation.sif \
    --input /input/raw_data \
    --output /output/segmentation.nii.gz
```

### 2. Inference Only (Default)

Run only the inference stage with preprocessed data:

```bash
apptainer run --bind /path/to/input:/input:ro \
    --bind /path/to/output:/output \
    --nv \
    segmentation.sif \
    --flair /input/flair.nii.gz \
    --dwi_b1000 /input/dwi.nii.gz \
    --t2s /input/t2s.nii.gz \
    --output /output/segmentation.nii.gz
```

### 3. Preprocessing Only

Run only the preprocessing stage:

```bash
apptainer run --app preprocess \
    --bind /path/to/input:/input:ro \
    --bind /path/to/output:/output \
    segmentation.sif \
    --input /input/raw_data \
    --output /output/preprocessed_data
```

## Environment Details

### Preprocessing Environment (pyenv + Python 3.12.0)
- **Environment Name**: `fomo25-3.12`
- **Location**: `/root/.pyenv/versions/fomo25-3.12/`
- **Key libraries**: nibabel, scikit-image, SimpleITK, OpenCV
- **Use case**: Image preprocessing, normalization, registration

### Inference Environment (conda + Python 3.8)
- **Environment Name**: `pytorch_p38`
- **Location**: `/opt/conda/envs/pytorch_p38/`
- **Key libraries**: PyTorch, MONAI, TensorBoard, scikit-learn
- **Use case**: Deep learning inference, model prediction

## Customization

### Adding Dependencies

1. **For preprocessing**: Edit `preprocess_requirements.txt`
2. **For inference**: Edit `environment.yml`

### Modifying Python Versions

1. **Preprocessing Python version**: Change `pyenv install 3.12.0` in `Apptainer.def`
2. **Inference Python version**: Change `python=3.8` in `environment.yml`

### Adding New Scripts

1. Add the script to the `%files` section in `Apptainer.def`
2. Make it executable in the `%post` section
3. Update the help documentation

## Troubleshooting

### Common Issues

1. **Build fails**: Ensure you have `--fakeroot` capability or build as root
2. **Environment not found**: Check that the environment paths are correct
3. **Permission denied**: Ensure scripts are executable (`chmod +x`)

### Debugging

To debug environment issues, you can shell into the container:

```bash
apptainer shell segmentation.sif
```

Then manually activate environments:
```bash
# For preprocessing environment
eval "$(pyenv init -)"
pyenv activate fomo25-3.12

# For inference environment
source /opt/conda/bin/activate pytorch_p38
```

## Performance Considerations

- The container is larger due to multiple Python installations
- Consider using multi-stage builds for production
- GPU support requires CUDA-enabled base images
- Memory usage will be higher with both environments loaded

## Security Notes

- The container runs as root inside (typical for Apptainer)
- Bind mounts should use `:ro` for input directories
- Consider security implications of running as root 