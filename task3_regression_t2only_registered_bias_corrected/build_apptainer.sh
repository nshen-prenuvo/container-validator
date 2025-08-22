#!/bin/bash

# Build the container
apptainer build \
    --fakeroot \
    task3_regression_t2only_registered_bias_corrected.sif \
    Apptainer.def
