#!/bin/bash
# Command to only validate the environment (without running the model)
## IMPORTANT: run inside pyenv fomo25-3.12
python main.py \
    --task task1 \
    --container task1_classification/task1_classification.sif \
    --data-dir fake_data/fomo25/fomo-task1-val/ \
    --output-dir output/task1/ # \
    # --validate-env-only
