# FOMO25 Challenge Submission Guide

This repository contains the guide for preparing submitting containers for the FOMO25 Challenge.
Please note: This repository will be continually refined, so check back occasionally to get the latest updates.

## Table of Contents
- [Overview](#overview)

- [1. Prerequisites](#prerequisites)
  - [Install Apptainer](#install-apptainer)
  <!-- - [Install Required Python Libraries](#install-required-python-libraries) -->

- [2. Prepare Required Files](#1-prepare-required-files)

  - [Task 1: Infarct Detection]()
    - [Inference file (predict.py)](#requirementstxt)
    - [Container Definition file (Apptainer.def)](#apptainerdef)

  - [Task 2: Meningioma Segmentation]()
    - [Inference file (predict.py)](#requirementstxt)
    - [Container Definition file (Apptainer.def)](#apptainerdef)

  - [Task 3: Brain Age Estimation]()
    - [Inference file (predict.py)](#requirementstxt)
    - [Container Definition file (Apptainer.def)](#apptainerdef)

- [3. Build Your Container](#3-build-your-container)

- [4. Run Validation](#4-run-validation)
  - [Validation Process](#validation-process)
  - [Interpreting Validation Results](#interpreting-validation-results)

<!-- - [FAQ](#faq) -->
- [Getting Help](#getting-help)

## Overview

The FOMO25 Challenge requires participants to submit their models as containerized solutions. This containerization approach ensures that your model can run in the evaluation environment exactly as it does on your own system, with all dependencies properly packaged. The container creates a standardized, isolated environment where your model can operate regardless of the host system configuration.

## Prerequisites

Before beginning the container validation process, ensure you have installed all necessary tools and dependencies.

### Install Apptainer

You need to install Apptainer (formerly Singularity) to build and run your container. Apptainer primarily supports Linux environments (Ubuntu, Debian, etc). If using MacOS or Windows, you'll need to use virtualization tools (Docker, Virtual Machines, or WSL2).

Installation instructions by platform:
- [Install in Linux (Ubuntu, Debian, Fedora, ...)](https://apptainer.org/docs/admin/main/installation.html#install-from-pre-built-packages)
- [Install in MacOS](https://apptainer.org/docs/admin/main/installation.html#mac)
- [Install in Windows](https://apptainer.org/docs/admin/main/installation.html#windows)

Once you have installed it, verify your Apptainer installation with:

```bash
apptainer --version
```





## 1. Task Specific Requirements

To participate in the FOMO25 Challenge, you must prepare a container that meets specific requirements for each downstream task. Each task has its own input and output specifications, in order to ensure that your evaluation is successful, you must follow the guidelines below. 

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

This task requires you to classify the presence of an infarct(s) in brain MRI images (binary classification). 



### predict.py
This script will be executed inside your container to perform the classification. It should take the required input images, process them, and output a single probability value indicating the presence of an infarct.

**Input**: T2 FLAIR, DWI (b-value 1000), ADC, and either T2* or SWI images.You will alway receive all four images, but you can use only the ones you need for your model.
**Output**: A text file (.txt) with the probability that an infarct is present. Single probability value (eg. 0.750).

Your `predict.py` script should handle the following command-line arguments:
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
This file defines how your container is built and what dependencies it includes. It specifies the base image, environment variables, files to include, and the command to run when the container starts. It should be structured as follows. Remember to include the necessary dependencies for your model, such as PyTorch, NumPy, and any other libraries you use in `predict.py`. Ensure that the script is executable and that it correctly handles the input and output paths specified in the command line arguments.

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
    Build: apptainer build --fakeroot my_task1_container.sif Apptainer.def
    
    Usage: apptainer run --bind /input:/input:ro --bind /output:/output \
            --nv my_task1_container.sif \
            --flair /input/flair.nii.gz \
            --adc /input/adc.nii.gz \
            --dwi_b1000 /input/dwi.nii.gz \
            --t2s /input/t2s.nii.gz \
            --output /output/prediction.txt
```


### Task 2: Meningioma Segmentation

This task requires you to segment meningiomas in brain MRI images. The output should be a binary mask indicating the presence of meningioma in the images.
### predict.py
This script will be executed inside your container to perform the segmentation. It should take the required input images, process them, and output a binary segmentation mask.



**Input**: T2 FLAIR, DWI (b-value 1000), and either T2* or SWI images.
**Output**: A NIfTI file (.nii.gz) containing the binary segmentation mask of the meningioma. It should have the same dimensions and affine as the input images.

Your `predict.py` script should handle the following command-line arguments:
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
    To use this container you will need to build it first and then run it with the appropriate arguments.

    1. Build: apptainer build --fakeroot my_task2_container.sif Apptainer.def
    
    2. Usage: apptainer run --bind /input:/input:ro \
            --bind /output:/output \
            --nv \
            my_task2_container.sif \
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
    Build: apptainer build --fakeroot my_task3_container.sif Apptainer.def
    
    Usage: apptainer run --bind /input:/input:ro \
            --bind /output:/output \
            --nv \
            my_task3_container.sif \
           --flair /input/flair.nii.gz \
           --adc /input/adc.nii.gz \
           --dwi_b1000 /input/dwi.nii.gz \
           --output /output/prediction.txt
```




## 3. Build Your Container

Build your container using the Apptainer.def file you prepared in step 2:

```bash
apptainer build --fakeroot /path/to/save/your/container.sif path/to/Apptainer.def
```

This command creates a `.sif` container file that encapsulates your model and all its dependencies.








## 4. Run Validation

Once your container is built, run the validation tool to ensure it will work correctly in the evaluation environment. This validation process will check that your container meets the requirements for each task and that it can process the input data correctly. You can use the fake data provided in the `fake_data/fomo25` directory to test your container.

Arguments for the validation tool:
- `--task {task1, task2, task3}`: Specify the task you are validating (task1=classification, task2=segmentation, task3=regression).
- `--container`: Path to your container file (.sif)
- `--apptainer-cmd`: Command to run Apptainer (default is `apptainer`).
- `--data-dir`: Path to the data directory containing preprocessed subjects
- `--output-dir`: Path to the directory where the output will be saved.
- `--validate-env-only`: Only validate the container environment (skip predictions).
- `--skip-gpu-check`: Skip GPU availability check during environment validation.


```bash
# Command to run full validation on task 1 (infarct detection)
python main.py --task task1 --container /path/to/your/container/for/task1.sif --data-dir fake_data/fomo25/fomo-task1-val/ --output-dir output/task1/ 

# Command to run validation on task 2 (meningioma segmentation)
python main.py --task task2 --container /path/to/your/container/for/task1.sif --data-dir fake_data/fomo25/fomo-task2-val/ --output-dir output/task2/

# Command to run validation on task 3 (brain age estimation)
python main.py --task task3 --container /path/to/your/container/for/task1.sif --data-dir fake_data/fomo25/fomo-task3-val/ --output-dir output/task3/


# Command to only validate the environment (without running the model)
python main.py --task task1 --container /path/to/your/container/for/task1.sif --data-dir fake_data/fomo25/fomo-task1-val/ --output-dir output/task1/ --validate-env-only

# Command to validate the environment using CPU only (skip GPU checks)
python main.py --task task1 --container /path/to/your/container/for/task1.sif --data-dir fake_data/fomo25/fomo-task1-val/ --output-dir output/task1/ --validate-env-only --skip-gpu-check
```




## Getting Help

If you encounter issues not covered in this documentation:

- Check the [main FOMO25 Challenge website](https://fomo25.github.io/) for additional resources
- Post questions by [creating an issue](https://github.com/pablorocg/fomo25-sanity-check-pipeline/issues/new) in the repository
- Contact the challenge organizers at fomo25@di.ku.dk

For Apptainer-specific issues, refer to the [official Apptainer documentation](https://apptainer.org/docs/user/latest/).