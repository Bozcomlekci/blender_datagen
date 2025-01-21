import random
import sys, os, shutil
import json
import bpy
import mathutils
import numpy as np
import glob
from PIL import Image
from generator import render_data
import argparse
from tqdm import tqdm

# Initialize argparser
parser = argparse.ArgumentParser()
parser.add_argument('--samples_folder', type=str, default='samples1')
parser.add_argument('--num_views', type=int, default=10)
parser.add_argument('--split_name', type=str, default='train')
parser.add_argument('--cam_settings', type=str, default='cam_settings/3dcustom.json')
parser.add_argument('--light_settings', type=str, default='light_settings/settings.json')
parser.add_argument('--data_settings', type=str, default='data_settings/settings.json')
parser.add_argument('--cam_file_settings', type=str, default='cam_file_settings/settings.json')
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--render_mask_ids', action='store_true', help='render part ids with alpha based on the color of the object part')
parser.add_argument('--render_silhouettes', action='store_true', help='render silhouette masks instead of segmentation masks')
parser.add_argument('--config_name', type=str, default=None)
parser.add_argument('--nested_folders', action='store_true', help='Structure the output in nested folders')
args = parser.parse_args()

samples_folder = args.samples_folder
data_settings = json.load(open(args.data_settings, 'r'))
data_settings = dict(data_settings)
render_folder_together = data_settings["render_folder_together"]
data_settings["render_silhouettes"] = args.render_silhouettes

if render_folder_together:
    # collect folders and exclude files
    path_list = glob.glob(f"{samples_folder}/*/") # get all folders
    path_list = [path[:-1] for path in path_list if not os.path.isfile(path)]
else:
    data_types = data_settings["data_types"]
    path_list = []
    # Recursively search subfolders
    obj_files = glob.glob(f"{samples_folder}/**/*.obj", recursive=True) if "obj" in data_types else []
    ply_files = glob.glob(f"{samples_folder}/**/*.ply", recursive=True) if "ply" in data_types else []
    npz_files = glob.glob(f"{samples_folder}/**/*.npz", recursive=True) if "npz" in data_types else []
    for files in [obj_files, ply_files, npz_files]:
        files.sort()
        path_list.extend(files)
    
i = 0
path_list.sort()

for file_path in tqdm(path_list):
    if args.nested_folders and args.config_name == None and not render_folder_together and (args.output_path != None):
        # Add scan's relative path to the output_path
        root_file_path = os.path.dirname(file_path)
        relative_scan_path = os.path.relpath(root_file_path, samples_folder)
        scan_out_path =  os.path.join(args.output_path, relative_scan_path)
    else:
        scan_out_path = args.output_path
    
    # Make data settings immutable
    render_data(file_path=file_path, split_name=args.split_name,
                cam_settings=args.cam_settings, num_views=args.num_views,
                light_settings=args.light_settings, cam_file_settings=args.cam_file_settings,
                data_settings=data_settings,
                output_path=scan_out_path,
                render_ids=args.render_mask_ids,
                config_name=args.config_name
    )

    
