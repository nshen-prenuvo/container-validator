#!/bin/bash

cd /home/ubuntu/fomo25_challenge/pipelines/pretrain_mim_med3d/container-validator/task1_classification

# Build the container
apptainer build \
    --fakeroot \
    task1_classification.sif \
    Apptainer.def
