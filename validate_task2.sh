#!/bin/bash
# Command to only validate the environment (without running the model)
## IMPORTANT: run inside pyenv fomo25-3.12
python main.py \
    --task task2 \
    --container task2_segmentation/task2_segmentation.sif \
    --data-dir fake_data/fomo25/fomo-task2-val/ \
    --output-dir output/task2/ # \
    # --validate-env-only
