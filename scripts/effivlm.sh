#!/bin/bash
# ==============================================================================
# VTC-Bench EffiVLM Evaluation Wrapper
# 
# This script is a wrapper to call the EffiVLM-Bench evaluation suite.
# It aligns with the parameter usage of dart.sh.
#
# Usage: 
#   bash scripts/effivlm.sh [IMAGE_RATIO] [REDUCTION_RATIO]
#
# Arguments:
#   IMAGE_RATIO     : Integer factor for physical image resizing (1=Original).
#   REDUCTION_RATIO : Float (0.0-1.0) for token pruning (e.g., 0.75).
# ==============================================================================
# --- 1. Default Configurations ---
IMAGE_RATIO=${1:-"1"}           # Default: 1 (No physical resize)
REDUCTION_RATIO=${2:-"0.75"}    # Default: 0.75 pruning ratio
# --- 2. Print Configuration ---
echo "--------------------------------------------------------"
echo "  VTC-Bench (EffiVLM) Evaluation Wrapper"
echo "--------------------------------------------------------"
echo "  Image Resize Ratio : $IMAGE_RATIO"
echo "  Pruning Ratio      : $REDUCTION_RATIO"
echo "--------------------------------------------------------"
# --- 3. Execute the underlying EffiVLM script ---
# We call the script using its path relative to the VTC-Bench root
bash external/EffiVLM-Bench/run_example.sh "$IMAGE_RATIO" "$REDUCTION_RATIO"