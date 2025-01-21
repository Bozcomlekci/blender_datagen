import bpy
import json
import os
import numpy as np
import glob
from PIL import Image
import numpy as np
import open3d as o3d
from pathlib import Path
import shutil
import torch
import pickle as pkl
import trimesh
import math
from assets.constants import CLOSED_CLOTHING_LABELS, COLOR_PALETTE, CLOSED_CLOTHING_LABELS_NEAREST

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def transform_save(out_data, output_path, split_name, cam_file_name=None):
    if cam_file_name is not None:
        transform_path = os.path.join(output_path, f'{cam_file_name}.json')
    else:
        transform_path = os.path.join(output_path, f'transforms_{split_name}.json')
    # check if file exists
    if os.path.exists(transform_path):
        # if exists then load and add out_data to it
        with open(transform_path, 'r') as in_file:
            in_data = json.load(in_file)
        in_data.update(out_data)
        out_data = in_data
    with open(transform_path, 'w') as out_file:
        json.dump(out_data, out_file, indent=4)

def rgb_to_alpha(imgs_folder, data_settings=None):
    imgs_path = glob.glob(imgs_folder + '/*')
    img = np.array(Image.open(imgs_path[0]))
    alpha_img = np.zeros((img.shape[0], img.shape[1]))
    # For each unique color in the image, create an alpha mask with integer values from 0 to 255
    if data_settings is not None and data_settings['unique_colors'] is not None:
        unique_colors = data_settings['unique_colors']
        unique_colors.insert(0, [0, 0, 0, 255])
    else:
        unique_colors = []
        for img_path in imgs_path:
            img = np.array(Image.open(img_path))
            for color in np.unique(img.reshape(-1, img.shape[2]), axis=0):
                if list(color) not in unique_colors:
                    unique_colors.append(list(color))
    mask_vals = [i for i in range(len(unique_colors))]
    # Copy old images to another directory and replace the color with the mask value
    mask_vis = Path(imgs_folder).parent / 'mask_vis'
    shutil.copytree(imgs_folder, mask_vis, dirs_exist_ok=True)
    for img_path in imgs_path:
        img = np.array(Image.open(img_path))
        # alpha_img = np.zeros((img.shape[0], img.shape[1]))
        for i, color in enumerate(unique_colors):
            # Find all pixels with the color in img with shape (H, W, 4)
            mask = np.all(img == color, axis=2)
            alpha_img[mask] = mask_vals[i]
        alpha_img_int = alpha_img.astype(np.uint8)
        alpha_img_int = Image.fromarray(alpha_img_int)
        alpha_img_int.save(img_path)
    # Save the mask val to color mapping to a json file
    maskint_to_color = {mask_vals[i]: np.array(unique_colors, dtype=np.int64)[i].tolist() for i in range(len(mask_vals))}
    parent_imgs_folder = Path(imgs_folder).parent
    # Add the maskint_to_color to the metadata.json file, don't overwrite the existing metadata
    if data_settings is not None and data_settings['include_metadata']:
        with open(os.path.join(parent_imgs_folder, 'metadata.json'), 'w') as out_file:
            json.dump({'maskint_to_color': maskint_to_color}, out_file, indent=4)

def replace_noisy_labels(noisy_labels, unique_labels, unique_counts, noisy_threshold=0.005):
    labels = np.copy(noisy_labels)
    # if the ratio of the labels to the total number of labels is less than the threshold, replace the label with the replacement label
    # replacement label is derived from the CLOSED_CLOTHING_LABELS_NEAREST
    num_labels = len(labels)
    unique_base_labels = unique_labels[unique_counts / num_labels > noisy_threshold]
    for label in unique_labels:
        count = np.sum(labels == label)
        if count / num_labels  < noisy_threshold:
            # Find the nearest label
            nearest_labels = CLOSED_CLOTHING_LABELS_NEAREST[label] #returns a proximity list of labels
            # choose the closest present label
            for nearest_label in nearest_labels:
                if nearest_label in unique_base_labels:
                    break
            labels[labels == label] = nearest_label
    unique_labels, unique_counts = np.unique(labels, axis=0, return_counts=True)
    return labels, unique_labels, unique_counts

def npz_to_ply(npz_file_path, VColorMask=False, color_mode='rgb', suppress_noised_labels=True, data_settings=None):
    '''
    Convert npz file to ply file using numpy and open3d
    '''
    npz_data = np.load(npz_file_path)
    # files: ['points', 'labels', 'faces', 'colors', 'scale', 'pose', 'betas', 'trans', 'coap_body_part', 'garments', 'normals', 'canon_pose']
    faces = npz_data['faces']
    points = npz_data['points']
    if VColorMask: # Mask rendering
        labels = np.array(npz_data['labels'], dtype=np.int32)
        unique_labels, unique_counts  = np.unique(labels, axis=0, return_counts=True)
        try:
            labels_to_colors = np.load('assets/color_palette.npy')
        except:
            labels_to_colors = np.array(COLOR_PALETTE)
        if suppress_noised_labels:
            labels, unique_labels, unique_counts = replace_noisy_labels(npz_data['labels'], unique_labels, unique_counts)
        labels_to_colors = np.array(labels_to_colors).round(decimals=3)
        # append the alpha channel to the colors
        labels_to_colors = np.concatenate((labels_to_colors, np.ones((len(labels_to_colors), 1))), axis=1)
        colors = labels_to_colors[labels]
        unique_colors = np.unique(colors, axis=0, return_counts=False)
        print(f'Unique labels: {unique_labels}')
        # Print the names of the unique labels
        for unique_label, unique_count in zip(unique_labels, unique_counts):
            print(f'Present label {unique_label}: {CLOSED_CLOTHING_LABELS[unique_label]} with {unique_count} occurences') 
        unique_colors = [np.rint(np.array(unique_colors)*255).astype(np.uint8).tolist() for unique_colors in unique_colors]
        labels_to_colors = [np.rint(np.array(labels_to_colors)*255).astype(np.uint8).tolist() for labels_to_colors in labels_to_colors]
        data_settings['unique_colors'], data_settings['labels_to_colors'] = unique_colors, labels_to_colors
    else: # Color rendering
        colors = npz_data['colors']
        if np.max(colors) > 1:
            colors = colors / 255.0
        if color_mode == 'bgr':
            colors = colors[:, [2, 1, 0]]
        elif color_mode == 'rgb':
            colors = colors
        else:
            raise ValueError('Invalid color mode. Choose either bgr or rgb')
       
    # Create trimesh mesh
    mesh = trimesh.Trimesh(vertices=points, faces=faces, vertex_colors=colors, process=False, maintain_order=True)
    ply_path = npz_file_path.replace('.npz', '.ply')
    
    root_dir = Path(ply_path).parent.parent
    if VColorMask:
        colors = np.round(colors, decimals=3)
        parent_dir = Path(ply_path).parent.name + '_gt_plys'
    else:
        parent_dir = Path(ply_path).parent.name + '_plys'
    file_name = Path(ply_path).name
    os.makedirs(os.path.join(root_dir, parent_dir), exist_ok=True)
    ply_path = os.path.join(root_dir, parent_dir , file_name)
    mesh.export(ply_path)

    if VColorMask:
        # Save the unique labels and their counts to a json file
        parent_folder, scan_name = Path(npz_file_path).parent, Path(npz_file_path).stem
        with open(os.path.join(parent_folder, f'{scan_name}_metadata.json'), 'w') as out_file:
            json.dump({str(CLOSED_CLOTHING_LABELS[unique_label]): int(unique_count) for unique_label, unique_count in zip(unique_labels, unique_counts)}, out_file, indent=4)
    return ply_path

def apply_transformations(mesh: o3d.geometry.TriangleMesh, rotation, scale=1, translation=None, center=None) -> o3d.geometry.TriangleMesh:
    '''
    Apply transformations to the mesh
    First scale the mesh, then translate it, then rotate it
    '''
    # Scale the mesh
    if center is None:
        mesh.scale(scale, center=mesh.get_center())
    else:
        mesh.scale(scale, center=center)
    # Center the mesh
    # print(f'Center of the mesh: {mesh.get_center()} with type {type(mesh.get_center())} and shape {mesh.get_center().shape}')
    if translation is not None:
        mesh.translate(translation)
    else:
        mesh.translate(-mesh.get_center())
    # Rotate
    R = mesh.get_rotation_matrix_from_xyz(rotation)
    # if center is None:
    mesh.rotate(R, center=(0, 0, 0))
    # else:  
    #     mesh.rotate(R, center=center)
    return mesh        

def render_background(foreground_img_dir, bg_color=(255, 255, 255)):
    """
    Copy the foreground images to the specified background
    """
    # Find all foreground images
    fg_path = Path(foreground_img_dir)
    img_paths = glob.glob(foreground_img_dir + '/*')
    
    transparent_img_dir = str(fg_path.parent / fg_path.name) + '_transparent'
    os.makedirs(transparent_img_dir, exist_ok=True)
    for foreground_img in img_paths:
        name = Path(foreground_img).name
        shutil.copy(foreground_img, transparent_img_dir + '/' + name)
    # Paste the foreground images to a specified bg
    for img_path in img_paths:
        # Create a new image with the specified background color
        img_size = Image.open(img_path).size
        bg = Image.new('RGBA', img_size, bg_color)
        fg = Image.open(img_path).convert('RGBA')
        bg.paste(fg, (0, 0), fg)
        bg.convert('RGB').save(img_path)

def include_textured_submeshes(output_path):
    return NotImplementedError
    # Include textured submeshes as well 
    colored_scan_path = os.path.join(output_path, 'points3d.ply')
    # Load using open3d 
    mesh = o3d.io.read_triangle_mesh(colored_scan_path)
    submeshes = glob.glob(os.path.join(output_path, 'objs') + '/*')
    for submesh in submeshes:
        # TO DO 
        pass

def include_metadata(output_path, scan_path, **kwargs):
    # Include metadata present in the scan folder in the output folder
    samples_folder, scan_name = Path(scan_path).parent, Path(scan_path).stem
    metadata_scan = os.path.join(samples_folder, f'{scan_name}_metadata.json')
    render_metadata = os.path.join(output_path, 'metadata.json')
    metadata = {}
    if os.path.exists(render_metadata):
        metadata_render = json.load(open(render_metadata, 'r'))
        metadata.update(metadata_render)
    if os.path.exists(metadata_scan):
        metadata_scan = json.load(open(metadata_scan, 'r'))
        metadata.update(metadata_scan)
    for key, value in kwargs.items():
        metadata.update({key: value})
    json.dump(metadata, open(render_metadata, 'w'), indent=4)

def trimesh_center_unit_scale(scan_path):
     # * normalize the scan
    scan_mesh = trimesh.load(scan_path, process=False, maintain_order=True)
    total_size = (scan_mesh.bounds[1] - scan_mesh.bounds[0]).max()
    centers = (scan_mesh.bounds[1] + scan_mesh.bounds[0]) / 2

    scan_mesh.apply_translation(-centers)
    scan_mesh.apply_scale(1 / total_size)
    # Overwrite the scan
    scan_mesh.export(scan_path)
    return centers, 1 / total_size

def get_center_tri(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """
    Get the center of the mesh, (min+max)/2 in Trimesh
    """
    centers = (mesh.bounds[1] + mesh.bounds[0]) / 2
    return centers

def assert_views(num_views, cam_settings):
    """
    Assert the number of views is valid
    """
    # calculate the number of views based on the camera settings
    if 'cam_config' in cam_settings:
        if type(cam_settings['cam_config']['img_per_angle']) is int:
            img_per_angle = cam_settings['cam_config']['img_per_angle']
            num_polar_angles = len(cam_settings['cam_config']['polar_angles'])
            nviews = img_per_angle * num_polar_angles
        elif type(cam_settings['cam_config']['img_per_angle']) is list:
            img_per_angle = cam_settings['cam_config']['img_per_angle']
            num_polar_angles = len(cam_settings['cam_config']['polar_angles'][0]) 
            nviews = sum([nimg * num_polar_angles for nimg in img_per_angle])   
        else:
            return NotImplementedError(f"Number of views is not specified")   
    else:
        if num_views is None:
            return NotImplementedError(f"Number of views is not specified")
        nviews = num_views
    return nviews
    

