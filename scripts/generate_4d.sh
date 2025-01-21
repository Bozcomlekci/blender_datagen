#!/bin/bash

# This is required, otherwise BlenderProc does not allow external imports
export OUTSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT_BUT_IN_RUN_SCRIPT=1

# Script: 4dgenerate.sh
python 4dgenerate.py \
  --samples_folder samples/take \
  --split_name train \
  --cam_settings cam_settings/monocular.json \
  --light_settings light_settings/settings.json \
  --fourd_fpt 1