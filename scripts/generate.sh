#!/bin/bash

# This is required, otherwise BlenderProc does not allow external imports
export OUTSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT_BUT_IN_RUN_SCRIPT=1

# Example runner for CloSe-D with mask
python generate.py --samples_folder samples/CloSe-Di --num_views 60 --split_name images --cam_settings cam_settings/circular.json --light_settings light_settings/settings.json --cam_file_settings cam_file_settings/settings.json --data_settings data_settings/basic.json
python generate.py --samples_folder samples/CloSe-Di --num_views 60 --split_name object_mask --cam_settings cam_settings/circular.json --light_settings light_settings/settings.json --cam_file_settings cam_file_settings/settings.json --data_settings data_settings/basic.json --render_mask_ids 

# Example runner for full CloSe-D data generation
python generate.py --samples_folder samples/CloSe-Di --num_views 60 --split_name train \
    --cam_settings cam_settings/spherical.json --light_settings light_settings/settings.json \
    --cam_file_settings cam_file_settings/train.json --data_settings data_settings/settings.json \
    --output_path blender_data --config_name closedi
python generate.py --samples_folder samples/CloSe-Di --num_views 10 --split_name test \
    --cam_settings cam_settings/spherical.json --light_settings light_settings/settings.json \
    --cam_file_settings cam_file_settings/test.json --data_settings data_settings/settings.json \
    --output_path blender_data --config_name closedi

