#!/bin/bash

cd /home/ubuntu/fomo25_challenge/pipelines/pretrain_mim_med3d/container-validator/task3_regression

# Build the container
apptainer build \
    --fakeroot \
    task3_regression.sif \
    Apptainer.def
