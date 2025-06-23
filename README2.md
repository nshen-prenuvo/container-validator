# FOMO25 Challenge Container Validation

This repository contains the official container validation tool for the FOMO25 Challenge, which investigates the few-shot generalization properties of foundation models for brain MRI data analysis.

Please note: This repository will be continually refined, so check back occasionally to get the latest updates.

## Table of Contents
- [Overview](#overview)
<!-- - [Validation Workflow](#validation-workflow) -->
- [Prerequisites](#prerequisites)
  - [Install Apptainer](#install-apptainer)
  - [Install Required Python Libraries](#install-required-python-libraries)
- [1. Prepare Required Files](#1-prepare-required-files)
  - [Task 1: Infarct Detection]()
    - [predict.py](#requirementstxt)
    - [Apptainer.def](#apptainerdef)
  - [Task 2: Meningioma Segmentation]()
    - [predict.py](#requirementstxt)
    - [Apptainer.def](#apptainerdef)
  - [Task 3: Brain Age Estimation]()
    - [predict.py](#requirementstxt)
    - [Apptainer.def](#apptainerdef)
- [2. Integrate Validation with Your Project](#2-integrate-validation-with-your-project)
  - [Validation Components](#validation-components)
  - [Project Directory Structure](#project-directory-structure)
  - [Setup Validation Environment](#setup-validation-environment)
- [3. Build Your Container](#3-build-your-container)
- [4. Run Validation](#4-run-validation)
  - [Validation Process](#validation-process)
  - [Interpreting Validation Results](#interpreting-validation-results)
- [5. Post-Validation Steps](#5-post-validation-steps)
- [Troubleshooting Guide](#troubleshooting-guide)
- [FAQ](#faq)
- [Getting Help](#getting-help)

## Overview

The FOMO25 Challenge requires participants to submit their models as containerized solutions. This containerization approach ensures that your model can run in the evaluation environment exactly as it does on your own system, with all dependencies properly packaged. The container creates a standardized, isolated environment where your model can operate regardless of the host system configuration.

This repository provides a validation tool that performs a critical "sanity check" on your container before official submission. The purpose of this validation is not to evaluate how well your model performs, but rather to verify that it meets the technical requirements needed for proper execution in the challenge environment. Many submissions are rejected due to technical issues that could have been caught beforehand, wasting valuable time and effort.

The validation specifically checks:

- Container structure and the presence of required files.
- Execution permissions and script functionality.
- Proper handling of input/output paths and NIfTI medical image files (a format commonly used for storing neuroimaging data).
- Container's ability to run in the expected environment.

By running this validation locally, you can identify and fix technical issues early, ensuring your submission can be properly evaluated on its scientific merits rather than being rejected due to implementation problems.

## Validation Workflow

The validation process follows this general workflow:

1. You prepare the required files for your container
2. You set up the validation environment
3. You build your container
4. You run the validation tool against your container
5. If validation fails, you debug and fix issues
6. When validation passes, your container is ready for submission

<div align="center">
  <img src="imgs/workflow-diagram-v2.svg" width="70%" alt="FOMO25 Container Validation Workflow">
</div>

## Prerequisites

Before beginning the container validation process, ensure you have installed all necessary tools and dependencies.

### Install Apptainer

You need to install Apptainer (formerly Singularity) to build and run your container. Apptainer primarily supports Linux environments (Ubuntu, Debian, etc). If using MacOS or Windows, you'll need to use virtualization tools (Docker, Virtual Machines, or WSL2).

Installation instructions by platform:
- [Install in Linux (Ubuntu, Debian, Fedora, ...)](https://apptainer.org/docs/admin/main/installation.html#install-from-pre-built-packages)
- [Install in MacOS](https://apptainer.org/docs/admin/main/installation.html#mac)
- [Install in Windows](https://apptainer.org/docs/admin/main/installation.html#windows)

Verify your Apptainer installation with:

```bash
apptainer --version
```



### Install Required Python Libraries

You need these Python libraries in your local environment for generating synthetic test data and calculating metrics (these are used by the validation scripts outside the container):

```bash
pip install nibabel numpy pandas scikit-learn tqdm
```




















## 1. Task Specific Requirements

You must prepare the following files for your submission (all these files are **mandatory**). Your container **must** have the following internal structure:

```
/
├── app/              # Your application code
│   └── predict.py    # Main inference script (REQUIRED)
├── input/            # Mounted input directory (DO NOT include in container)
├── output/           # Mounted output directory (DO NOT include in container)
└── ...               # Other system files
```

Important notes:
- Your predict.py file must be located at `/app/predict.py`
- The input and output directories are mounted at runtime and should not be included in your container


### Task 1: Infarct Detection

**Input**: T2 FLAIR, DWI (b-value 1000), ADC, and either T2* or SWI images.
**Output**: A text file (.txt) with the probability that an infarct is present. Single probability value (eg. 0.750)

Required arguments for `predict.py`:
- `--flair`: Path to T2 FLAIR image
- `--adc`: Path to ADC image
- `--dwi_b1000`: Path to DWI b1000 image
- `--t2s`: Path to T2* image (optional, can be replaced with SWI)
- `--swi`: Path to SWI image (optional, can be replaced with T2*)
- `--output`: Path to save output .txt file with probability

**Example usage**:
```bash
python predict.py \
  --flair /path/to/flair.nii.gz \
  --adc /path/to/adc.nii.gz \
  --dwi_b1000 /path/to/dwi_b1000.nii.gz \
  --t2s /path/to/t2s.nii.gz \
  --swi /path/to/swi.nii.gz \
  --output /path/to/output.txt
```
### Apptainer.def
<!-- Here i'd like to explain how the apptainer def file should look for this task and what has to be modified -->
Your `Apptainer.def` file should be structured as follows. Remember to include the necessary dependencies for your model, such as PyTorch, NumPy, and any other libraries you use in `predict.py`. Ensure that the script is executable and that it correctly handles the input and output paths specified in the command line arguments.

```apptainer
Bootstrap: docker

# Use any docker image as a base (see https://hub.docker.com/)
# If using GPU, consider using a CUDA-enabled base image
From: python:3.11-slim

%labels
    Author Your Name Here
    Version v1.0.0
    Description FOMO25 Infarct Classification Submission

%environment
    export PYTHONUNBUFFERED=1
    export LC_ALL=C.UTF-8

%files
    # Copy your files to the container (predict.py, requirements.txt, model weigths, ...) - ADD YOUR MODEL FILES HERE:
    
    ./predict.py /app/predict.py
    ./requirements.txt /app/requirements.txt

%post
    mkdir -p /input /output /app
    
    # Install system dependencies if needed
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        && rm -rf /var/lib/apt/lists/*
    
    # Install Python dependencies
    pip install --no-cache-dir -U pip setuptools wheel
    pip install --no-cache-dir -r /app/requirements.txt
    
    # Make predict.py executable
    chmod +x /app/predict.py

%runscript
    exec python /app/predict.py "$@"

%help
    Build: apptainer build --fakeroot classification.sif Apptainer.def
    
    Usage: apptainer run --bind /input:/input:ro \
            --bind /output:/output \
            --nv \
            my_classification_container.sif \
           --flair /input/flair.nii.gz \
           --adc /input/adc.nii.gz \
           --dwi_b1000 /input/dwi.nii.gz \
           --output /output/prediction.txt
```


### Task 2: Meningioma Segmentation

**Input**: T2 FLAIR, DWI (b-value 1000), and either T2* or SWI images.
**Output**: A NIfTI file (.nii.gz) containing the binary segmentation mask of the meningioma. It should have the same dimensions and affine as the input images.

Required arguments for `predict.py`:
- `--flair`: Path to T2 FLAIR image
- `--dwi_b1000`: Path to DWI b1000 image
- `--t2s`: Path to T2* image (optional, can be replaced with SWI)
- `--swi`: Path to SWI image (optional, can be replaced with T2*)
- `--output`: Path to save segmentation NIfTI file

**Example usage**:
```bash
python predict.py \
  --flair /path/to/flair.nii.gz \
  --dwi_b1000 /path/to/dwi_b1000.nii.gz \
  --t2s /path/to/t2s.nii.gz \
  --swi /path/to/swi.nii.gz \
  --output /path/to/output.nii.gz
```
### Apptainer.def
<!-- Here i'd like to explain how the apptainer def file should look for this task and what has to be modified -->

Your `Apptainer.def` file should be structured as follows. Remember to include the necessary dependencies for your model, such as PyTorch, NumPy, and any other libraries you use in `predict.py`. Ensure that the script is executable and that it correctly handles the input and output paths specified in the command line arguments.

```apptainer
Bootstrap: docker

# Use any docker image as a base (see https://hub.docker.com/)
# If using GPU, consider using a CUDA-enabled base image
From: python:3.11-slim

%labels
    Author Your Name Here
    Version v1.0.0
    Description FOMO25 Infarct Classification Submission

%environment
    export PYTHONUNBUFFERED=1
    export LC_ALL=C.UTF-8

%files
    # Copy your files to the container (predict.py, requirements.txt, model weigths, ...) - ADD YOUR MODEL FILES HERE:
    
    ./predict.py /app/predict.py
    ./requirements.txt /app/requirements.txt

%post
    mkdir -p /input /output /app
    
    # Install system dependencies if needed
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        && rm -rf /var/lib/apt/lists/*
    
    # Install Python dependencies
    pip install --no-cache-dir -U pip setuptools wheel
    pip install --no-cache-dir -r /app/requirements.txt
    
    # Make predict.py executable
    chmod +x /app/predict.py

%runscript
    exec python /app/predict.py "$@"

%help
    Build: apptainer build --fakeroot classification.sif Apptainer.def
    
    Usage: apptainer run --bind /input:/input:ro \
            --bind /output:/output \
            --nv \
            my_classification_container.sif \
           --flair /input/flair.nii.gz \
           --adc /input/adc.nii.gz \
           --dwi_b1000 /input/dwi.nii.gz \
           --output /output/prediction.txt
```

### Task 3: Brain Age Estimation

### predict.py
**Input**: T1-weighted and T2-weighted images.
**Output**: A text file (.txt) containing the predicted brain age in years.

Required arguments for `predict.py`:
- `--t1`: Path to T1-weighted image
- `--t2`: Path to T2-weighted image
- `--output`: Path to save output .txt file with predicted brain age. Single value (eg. 35.5)

**Example usage**:
```bash
python predict.py \
  --t1 /path/to/t1.nii.gz \
  --t2 /path/to/t2.nii.gz \
  --output /path/to/output.txt
```

### Apptainer.def
<!-- Here i'd like to explain how the apptainer def file should look for this task and what has to be modified -->
Your `Apptainer.def` file should be structured as follows. Remember to include the necessary dependencies for your model, such as PyTorch, NumPy, and any other libraries you use in `predict.py`. Ensure that the script is executable and that it correctly handles the input and output paths specified in the command line arguments.

```apptainer
Bootstrap: docker

# Use any docker image as a base (see https://hub.docker.com/)
# If using GPU, consider using a CUDA-enabled base image
From: python:3.11-slim

%labels
    Author Your Name Here
    Version v1.0.0
    Description FOMO25 Infarct Classification Submission

%environment
    export PYTHONUNBUFFERED=1
    export LC_ALL=C.UTF-8

%files
    # Copy your files to the container (predict.py, requirements.txt, model weigths, ...) - ADD YOUR MODEL FILES HERE:
    
    ./predict.py /app/predict.py
    ./requirements.txt /app/requirements.txt

%post
    mkdir -p /input /output /app
    
    # Install system dependencies if needed
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        && rm -rf /var/lib/apt/lists/*
    
    # Install Python dependencies
    pip install --no-cache-dir -U pip setuptools wheel
    pip install --no-cache-dir -r /app/requirements.txt
    
    # Make predict.py executable
    chmod +x /app/predict.py

%runscript
    exec python /app/predict.py "$@"

%help
    Build: apptainer build --fakeroot classification.sif Apptainer.def
    
    Usage: apptainer run --bind /input:/input:ro \
            --bind /output:/output \
            --nv \
            my_classification_container.sif \
           --flair /input/flair.nii.gz \
           --adc /input/adc.nii.gz \
           --dwi_b1000 /input/dwi.nii.gz \
           --output /output/prediction.txt
```




## 2. Integrate Validation with Your Project

### Validation Components

The validation tool includes these key components that will test your container:

- `validate_container.sh`: Main validation script that orchestrates the testing process
- `compute_metrics.py`: Calculates performance metrics on your model's output
- `test_data_generator.py`: Creates synthetic NIfTI test images for validation

You don't need to modify these files, but understanding their purpose helps troubleshoot validation issues.

### Project Directory Structure

Set up your project with the following recommended structure to easily integrate the validation tool:

```
your-project/
├── src/                  # Your model code and implementation
│   ├── predict.py        # Main inference script (will be copied to container)
│   └── ...               # Other model files
├── requirements.txt      # Dependencies for your model
├── Apptainer.def         # Container definition file
├── validation/           # Validation tool directory (clone from this repo)
│   ├── validate_container.sh
│   ├── compute_metrics.py
│   ├── test_data_generator.py
│   └── ...
├── test/                 # Default directories for validation data
│   ├── input/            # Test inputs (empty, will be populated during validation)
│   └── output/           # Test outputs (empty, will be populated during validation)
└── container_config.yml  # Validation configuration
```

### Setup Validation Environment

1. **Clone the validation repository into your project**

```bash 
git clone https://github.com/pablorocg/fomo25-sanity-check-pipeline.git validation
```

2. **Copy configuration template**

```bash 
cp validation/container_config.template.yml ./container_config.yml
```

3. **Create necessary directories**

```bash 
mkdir -p test/input test/output
```

4. **Configure validation settings**
Edit `container_config.yml` to match your project's specific needs:

```yaml
# Container settings
container:
  name: "your-model-name"   # Give your container a meaningful name
  command: "apptainer"      # Use "apptainer" or "singularity" based on your installation

# Directory paths
directories:
  input: "test/input"       # Relative path to test input directory
  output: "test/output"     # Relative path to test output directory
  containers: "."           # Location where your container image is stored

# Validation settings
validate:
  gpu: true                 # Set to false if not using GPU for testing
  generate_data: true       # Creates synthetic test data
  compute_metrics: true     # Calculate performance metrics
  save_report: true         # Generate validation report
  result_file: "validation_result.json"  # Report output location
```

## 3. Build Your Container

Build your container using the Apptainer.def file you prepared in step 1:

```bash
apptainer build /path/to/save/your/container.sif Apptainer.def
```

This command creates a `.sif` container file that encapsulates your model and all its dependencies.

## 4. Run Validation

### Validation Process

Once your container is built, run the validation tool to ensure it will work correctly in the evaluation environment:

```bash
./validation/validate_container.sh --path /path/to/your-container.sif
```

Or if you've configured a custom `container_config.yml`:

```bash
./validation/validate_container.sh --config container_config.yml
```

The validation process will:
1. Generate synthetic NIfTI test data (if configured)
2. Run your container against this test data
3. Evaluate the output format and basic functionality
4. Generate a validation report in `validation_result.json`

### Interpreting Validation Results

The validation tool produces a detailed report with information about:
- Container structure verification
- Execution success/failure
- Output format correctness
- Basic performance metrics

Review this report carefully to identify any issues that need to be addressed.

## 5. Post-Validation Steps

Once your container passes validation:

1. **Review the validation report** one final time to ensure there are no warnings or issues
2. **Test with representative data** if possible, to confirm your model performs as expected
3. **Submit your container** to the FOMO25 Challenge platform following the submission guidelines on the main challenge website
4. **Track your submission status** on the challenge platform for any feedback or issues

## Troubleshooting Guide

Common validation errors and their solutions:

| Error | Possible Cause | Solution |
|-------|---------------|----------|
| Missing predict.py | Script not at the correct path | Ensure predict.py is at `/app/predict.py` in the container |
| Permission denied | Script not executable | Add `chmod +x /app/predict.py` to your Apptainer.def %post section |
| Dependency errors | Missing packages | Check that all required packages are in requirements.txt and properly installed |
| Input/output errors | Incorrect path handling | Verify your script correctly uses the paths provided via command-line arguments |
| Memory errors | Model too large for available resources | Optimize your model or check GPU memory usage |
| NIfTI format errors | Metadata not preserved | Ensure you're using the input image's affine and header for the output |

For more complex issues, check the validation logs and container build logs for detailed error messages.


## FAQ

**Q: Do I need to include training code in my submission?**  
A: No, only the inference code is required. The evaluation will only run your `predict.py` script.

**Q: Can I use frameworks other than PyTorch?**  
A: Yes, you can use any framework as long as it's included in your container. Make sure to specify all dependencies in your `Apptainer.def` file.

**Q: How do I handle GPU support?**  
A: The validation script will test GPU support if available. Include GPU-compatible versions of your libraries if your model uses GPU acceleration.

**Q: Can I test with my own data?**  
A: Yes, place your test data in the input directory defined in `container_config.yml`.


## Getting Help

If you encounter issues not covered in this documentation:

- Check the [main FOMO25 Challenge website](https://fomo25.github.io/) for additional resources
- Post questions by [creating an issue](https://github.com/pablorocg/fomo25-sanity-check-pipeline/issues/new) in the repository
- Contact the challenge organizers at fomo25@di.ku.dk

For Apptainer-specific issues, refer to the [official Apptainer documentation](https://apptainer.org/docs/user/latest/).