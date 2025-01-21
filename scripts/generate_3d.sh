#!/bin/bash

# This is required, otherwise BlenderProc does not allow external imports
export OUTSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT_BUT_IN_RUN_SCRIPT=1

# An example script for rendering sample objects that are stored inside samples folder
python generate.py \
  --samples_folder samples/CloSe-Di \
  --num_views 18 \
  --split_name images \
  --cam_settings cam_settings/3dcustom.json \
  --light_settings light_settings/basic.json \
  --cam_file_settings cam_file_settings/basic.json \
  --data_settings data_settings/basic.json \


# For rendering mask ids
python generate.py \
  --samples_folder samples/CloSe-Di \
  --num_views 18 \
  --split_name object_mask \
  --cam_settings cam_settings/3dcustom.json \
  --light_settings light_settings/basic.json \
  --cam_file_settings cam_file_settings/basic.json \
  --data_settings data_settings/basic.json \
  --render_mask_ids