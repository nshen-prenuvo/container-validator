#!/bin/bash

# Build the container
apptainer build \
    --fakeroot \
    task3_regression_t2only_registered.sif \
    Apptainer.def
