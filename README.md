# FOMO25 Container Validator

For instructions on
1. how to prepare your code for submission, please see [Preparing your model for submission](https://www.synapse.org/Synapse:syn64895667/wiki/633093).
2. how to submit your container, please see [Submission instructions](https://www.synapse.org/Synapse:syn64895667/wiki/632983).

## Installation

We recommend using a virtual environment to avoid dependency conflicts:

1. Install Apptainer
You need to install Apptainer (formerly Singularity) to build and run your container. Installation instructions by platform:
- [Install in Linux (Ubuntu, Debian, Fedora, ...)](https://apptainer.org/docs/admin/main/installation.html#install-from-pre-built-packages)
- [Install in MacOS](https://apptainer.org/docs/admin/main/installation.html#mac)
- [Install in Windows](https://apptainer.org/docs/admin/main/installation.html#windows)

Once you have installed it, verify your Apptainer installation with:

```bash
apptainer --version
```
2. Install python requirements

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```
Make sure you have Python 3.8+ and pip installed.

## How To Run Validation

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

