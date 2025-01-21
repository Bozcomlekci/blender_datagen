#!/bin/bash

# This is required, otherwise BlenderProc does not allow external imports
export OUTSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT_BUT_IN_RUN_SCRIPT=1

# Example runner for synthetic data (multiple objs in a folder rendered together) generation
python generate.py --samples_folder assets/examples --num_views 20 --split_name images --cam_settings cam_settings/multi.json --light_settings light_settings/multi.json --cam_file_settings cam_file_settings/settings.json --data_settings data_settings/multi.json
python generate.py --samples_folder assets/examples --num_views 20 --split_name object_mask --cam_settings cam_settings/multi.json --light_settings light_settings/multi.json --cam_file_settings cam_file_settings/settings.json --data_settings data_settings/multi.json --render_mask_ids



