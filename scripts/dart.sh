#!/bin/bash
# ==============================================================================
# VTC-Bench Evaluation Suite
# 
# This script manages the evaluation of DART (Dynamic Autoregressive 
# Token-reduction) and native Qwen2-VL models. It allows flexible control 
# over model architecture, image resolution, and pruning intensity.
#
# Usage: 
#   bash dart.sh [USE_DART] [IMAGE_RATIO] [REDUCTION_RATIO]
#
# Arguments:
#   USE_DART        : 'true' to use DART-optimized model, 'false' for native.
#   IMAGE_RATIO     : Integer factor for physical image resizing (1=Original).
#   REDUCTION_RATIO : Float (0.0-1.0) for DART token pruning (e.g., 0.778).
# ==============================================================================

# --- Default Configurations ---
USE_DART=${1:-"true"}           # Default: true (Enable DART)
IMAGE_RATIO=${2:-"1"}           # Default: 1 (No physical resize)
REDUCTION_RATIO=${3:-"0.75"}   # Default: 0.778 pruning ratio

# --- Input Validation ---
if [[ "$USE_DART" != "true" && "$USE_DART" != "false" ]]; then
  echo "[Error] The first argument must be 'true' or 'false'."
  echo "Usage: bash Downsample.sh [true|false] [image_ratio] [reduction_ratio]"
  exit 1
fi

# --- Export Environment Variables ---
# These variables will be consumed by the Python backend 
# (modeling_qwen2_vl_self.py and qwen2_vl.py)
export DART_DOWNSAMPLE=$USE_DART
export VTC_IMAGE_RATIO=$IMAGE_RATIO
export PYTHONPATH=$PYTHONPATH:$(pwd)/external/DART/Qwen2-VL
# --- Print Configuration ---
echo "--------------------------------------------------------"
echo "  VTC-Bench Evaluation Configuration"
echo "--------------------------------------------------------"
echo "  Model Architecture : $( [ "$USE_DART" == "true" ] && echo "DART (Enabled)" || echo "Native Qwen2-VL" )"
echo "  Image Resize Ratio : $VTC_IMAGE_RATIO"
echo "  DART Pruning Ratio : $( [ "$USE_DART" == "true" ] && echo "$REDUCTION_RATIO" || echo "N/A" )"
echo "--------------------------------------------------------"

# --- Construct Experiment ID for Organization ---
if [[ "$USE_DART" == "true" ]]; then
    EXP_ID="DART_IR${IMAGE_RATIO}_RR${REDUCTION_RATIO}"
else
    EXP_ID="NATIVE_IR${IMAGE_RATIO}"
fi

# --- Execute Evaluation ---
# Ensure the path to the underlying evaluation script is correct
bash external/DART/Qwen2-VL/eval_scripts/lmms_eval.sh "$USE_DART" "$REDUCTION_RATIO" "$EXP_ID"