#!/bin/bash
# Command to only validate the environment (without running the model)
## IMPORTANT: run inside pyenv fomo25-3.12
python main.py \
    --task task3 \
    --container task3_regression_t2only_registered_bias_corrected/task3_regression_t2only_registered_bias_corrected.sif \
    --data-dir fake_data/fomo25/fomo-task3-val/ \
    --output-dir output/task3/ # \
    # --validate-env-only
