# FOMO25 Challenge Container Validator

A streamlined validation framework for neuroimaging models using Apptainer/Singularity containers. This toolkit helps FOMO25 Challenge participants build, test, and validate their model submissions to ensure they meet all requirements.

## 📋 Overview

This validation framework helps you:
- Build Apptainer/Singularity containers without requiring Docker
- Test model structure and compatibility
- Validate GPU and CPU execution
- Verify input/output interfaces work correctly
- Ensure your submission will pass the official evaluation

## ⚙️ Requirements

### System Prerequisites
- Apptainer (preferred) or Singularity
- Python 3.x
- Sudo privileges for container building
- NVIDIA GPU with drivers (optional, CPU fallback available)

### Model Requirements
Your neuroimaging model must:
1. Include a `predict.py` script in the `/app` directory
2. Accept input from the `/input` directory (mounted read-only)
3. Write outputs to the `/output` directory
4. Process NIfTI format input files (`.nii` or `.nii.gz`)
5. Handle both GPU and CPU execution
6. List dependencies in `requirements.txt`

## 🚀 Quick Start

```bash
# Build a container
./build.sh -n my-container

# Validate the container
./do_validate_container.sh -n my-container
```

## 📚 Usage Options

### Building Containers with build.sh

```
Usage: ./build.sh [options]
Options:
  -n, --name NAME      Container name (default: fomo25-container)
  -d, --def FILE       Definition file (default: ./Apptainer.def)
  -o, --output DIR     Output directory for containers (default: ./apptainer-images)
  -c, --config FILE    Config file path (default: ./config.yml)
  --cmd PATH           Custom Apptainer/Singularity command path
  -h, --help           Show this help
```

### Validating Containers with do_validate_container.sh

```
Usage: ./do_validate_container.sh [options]
Options:
  -n, --name NAME      Container name (default: fomo25-container)
  -p, --path PATH      Container path (overrides name)
  -i, --input DIR      Input directory (default: ./test/input)
  -o, --output DIR     Output directory (default: ./test/output)
  -c, --config FILE    Config file path (default: ./container_config.yml)
  --no-gpu             Disable GPU support
  --result FILE        Specify output JSON file for results
  --cmd PATH           Custom Apptainer/Singularity command path
  -h, --help           Show this help
```

### Examples:

```bash
# Build with a custom name
./build.sh -n custom-model

# Build with custom definition file
./build.sh -d custom.def -n my-model

# Validate with specific input/output dirs
./do_validate_container.sh -n my-model -i /path/to/inputs -o /path/to/results

# Validate without GPU support
./do_validate_container.sh -n my-model --no-gpu
```

## 🔍 Validation Process

### Structure Tests
The validator checks:
- Container existence and accessibility
- Presence of required files (`/app/predict.py`)
- Basic command execution capability
- GPU support detection

### Runtime Tests
For inference testing, the validator:
1. Generates synthetic test data if needed
2. Mounts input/output directories
3. Runs prediction with performance monitoring
4. Verifies output files were generated
5. Computes evaluation metrics

## 🛠️ Repository Structure

```
.
├── Apptainer.def          # Container definition template
├── build.sh               # Main build script
├── do_build_container.sh  # Alternative build script
├── do_validate_container.sh # Validation script
├── container_config.yml   # Configuration template
├── requirements.txt       # Python dependencies
├── src/                   # Example source code directory
│   ├── predict.py         # Required prediction script template
│   └── ...                # Other supporting code
├── validation/            # Validation tools
│   ├── compute_metrics.py # Metrics computation
│   └── test_data_generator.py # Test data generation
└── submission-guide.md    # Detailed submission guidelines
```

## 📊 Output & Metrics

After validation, the results are saved to a JSON file (default: `validation_result.json`). This includes:

- Status: `PASSED` or `FAILED`
- Detailed check results (container structure, GPU support, etc.)
- Errors and warnings
- Performance metrics

If metrics computation is enabled, detailed segmentation metrics are saved to:
- `test/output/results/metrics_results.json`
- `test/output/results/metrics_results.csv`

## 🔧 Troubleshooting

### Container Build Issues
- Verify Apptainer/Singularity is installed
- Check your `requirements.txt` for compatibility issues
- Ensure you have sudo privileges

### Validation Failures
- Container not found: Build it first with build.sh
- predict.py not found: Ensure it exists at `/app/predict.py` in your container
- No output files: Make sure your model writes to the `/output` directory
- GPU not detected: Install NVIDIA drivers or use `--no-gpu`

## 📝 Note for Challenge Participants

This validation framework ensures your model will function correctly in the FOMO25 Challenge environment. Pass all validation checks to confirm your submission will be evaluated properly.

A successful validation confirms:
1. Your container can be built
2. Your model can run inference
3. I/O paths are correctly configured
4. Output format meets requirements

For more detailed information about challenge requirements and submission guidelines, please refer to the included `submission-guide.md` document.

## 📄 License

This framework is provided for use in the FOMO25 Challenge.