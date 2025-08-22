#!/bin/bash

cd /home/ubuntu/fomo25_challenge/pipelines/pretrain_mim_med3d/container-validator/task2_segmentation

# Build the container
apptainer build \
    --fakeroot \
    task2_segmentation.sif \
    Apptainer.def
