import sys, os
import json
import bpy
import mathutils
import numpy as np
from scene_light import setLight_sun, setLight_ambient, shadowThreshold, invisibleGround
from utils.blender_utils import get_scale, get_center, lookAt, get_min_max
from utils.blender_utils import renderImage, blenderInit, readMesh, setCamera, std_of_mesh, setCamera_orthographic, parent_obj_to_camera
from utils.utils import listify_matrix, transform_save, rgb_to_alpha, render_background
from utils.utils import include_textured_submeshes, include_metadata
from utils.utils import assert_views
from utils.blender_utils import include_file, export_transformation_matrix
from math import radians
from pathlib import Path
from cam_path import *
import copy
from types import MappingProxyType

def render_data(
                file_path, num_views=60, split_name='train', cam_settings='cam_settings/3dcustom.json',
                light_settings='light_settings/settings.json', cam_file_settings=None,
                num_samples=200, output_path=None,
                data_settings= None,
                render_ids=False,
                config_name=None,
                ):
    data_config = copy.deepcopy(data_settings)
    data_settings = MappingProxyType(data_settings)
    file_name = file_path.split("/")[-1].split(".")[0]
    config = cam_settings.split("/")[-1].split(".")[0]
    cam_settings = json.load(open(cam_settings, 'r'))
    radius = cam_settings['radius']
    resolution = cam_settings['resolution']
    camera_type = cam_settings['camera_type']
    focalLength = cam_settings['focalLength']
    if camera_type == 'orthographic':
        orthographic_window_size = cam_settings['orthographic_window_size']

    nviews = assert_views(num_views, cam_settings)
    imgRes_x = resolution
    imgRes_y = resolution
    
    use_GPU = True
    light_settings = json.load(open(light_settings, 'r'))

    if cam_file_settings is not None:
        cam_file_settings = json.load(open(cam_file_settings, 'r'))
        cam_file_name = cam_file_settings['file_name']
        img_extension = cam_file_settings['img_extension']
        include_img_extension = cam_file_settings['include_img_extension']
        include_intrinsics_details = cam_file_settings['include_intrinsics_details']
    else:
        cam_file_name = None
        img_extension = 'png'
        include_img_extension = False
        include_intrinsics_details = False

    if data_config is not None and 'forward_facing' in data_config and data_config['forward_facing']:
        data_config['rotation_euler'][0] += data_config['forward_facing'][0] 
        data_config['rotation_euler'][1] += data_config['forward_facing'][1]
        data_config['rotation_euler'][2] += data_config['forward_facing'][2]
    if config_name is None:
        config_name = f'{config}_{nviews}'
    if output_path is None:
        output_path = os.path.join('nvs_data', config_name, file_name)
    else:
        output_path = os.path.join(output_path, config_name, file_name)
    fp = bpy.path.abspath(os.path.join(output_path, split_name))
    if not os.path.exists(fp):
        os.makedirs(fp)
        
    if render_ids:
        blenderInit(imgRes_x, imgRes_y, num_samples, use_GPU, clean_scene=True, mask_render=render_ids)
    
        # Render vertex colored mesh
        mesh, scale_factor = readMesh(file_path, VColorMask=render_ids, data_settings=data_config)
    else:
        blenderInit(imgRes_x, imgRes_y, num_samples, use_GPU, clean_scene=True, render_background=light_settings['render_background'])
        mesh = readMesh(file_path, data_settings=data_config)

        bpy.ops.object.shade_smooth()

        if 'shadow_catcher' in light_settings and light_settings['shadow_catcher']:
            # set invisible plane (shadow catcher)
            mesh_min, mesh_max = get_min_max(mesh, dim=2)
            ground = invisibleGround(shadowBrightness=0.9, location=[0,0,mesh_min])

        # Set lighting
        if light_settings['use_sun']:
            lightAngle = light_settings['light_angle']
            strength = light_settings['sun_strength']
            shadowSoftness = light_settings['shadow_softness']
            sun = setLight_sun(lightAngle, strength, shadowSoftness)

        # set ambient light
        color = light_settings['ambient_light_color']
        setLight_ambient(color=color)

        # set gray shadow to completely white with a threshold (optional but recommended)
        if light_settings['alpha_threshold'] or light_settings['interpolation_mode']:
            shadowThreshold(alphaThreshold=light_settings['alpha_threshold'], interpolationMode=light_settings['interpolation_mode'])


    # Scale of the whole mesh is precalculated before mesh partitioning for mask rendering, so no need to recalculate
    if not render_ids:
        if 'distance_by_std' in data_settings and data_settings['distance_by_std']:
            #Note that the true radius will be relative as it is normalized by the std of the mesh for rendering purposes
            scale_factor = std_of_mesh(mesh)
        else:
            scale_factor = 1 / get_scale(mesh) # get scale of the mesh to decide camera distance from the object
           
    # Set shortcut for bpy scene
    scene = bpy.context.scene

    # Set camera
    cam_distance = radius*scale_factor 
    camLocation =  (0, cam_distance, 0) 
    lookAtLocation = (0,0,0) # get_center(mesh)
    cam = setCamera(camLocation, focalLength=focalLength, lookAtLocation=lookAtLocation)
    cam = scene.objects['Camera']
    # Data to store in JSON file
    out_data = {
        'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
    }

    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty
    stepsize = 360.0 / nviews

    out_data['frames'] = []

    if include_intrinsics_details:
        # Obtain the instrinsics from the camera_angle_x
        out_data['fl_x'] = 0.5 * imgRes_x / np.tan(0.5 * bpy.data.objects['Camera'].data.angle_x)
        out_data['fl_y'] = 0.5 * imgRes_y / np.tan(0.5 * bpy.data.objects['Camera'].data.angle_x)

        out_data['cx'] = imgRes_x/2
        out_data['cy'] = imgRes_y/2
        out_data['w'] = imgRes_x
        out_data['h'] = imgRes_y
    
    for i in range(0, nviews):
        scene.render.filepath = os.path.join(fp, ("{:04d}".format(i)))
        outputPath = scene.render.filepath + f'.{img_extension}'
        renderImage(cam, outputPath=outputPath) 

        img_file_path = split_name + '/' + ("{:04d}".format(i)) + f'.{img_extension}' if include_img_extension else split_name + '/' + ("{:04d}".format(i))
        frame_data = {
            'file_path': img_file_path,
            'rotation': radians(stepsize),
            'transform_matrix': listify_matrix(cam.matrix_world)
        }
        out_data['frames'].append(frame_data)
        next_cam(config, b_empty, i, cam_settings, stepsize, head_offset=scale_factor)
            
    # Reset scene camera
    b_empty.rotation_euler = (0, 0, 0) # Reset rotation back to the original state for sanity

    if render_ids:
        rgb_to_alpha(os.path.join(output_path, split_name), data_settings=data_config)
        if data_config['include_objs']:
            obj_path = os.path.join(output_path, 'objs')
            if not os.path.exists(obj_path):
                os.makedirs(obj_path)
            # Include each objects' ply in the output folder
            include_file(obj_path, file_path, export_objs=True)
    else:
        transform_save(out_data, output_path, split_name, cam_file_name=cam_file_name)
        if data_config['include_file']:
            include_file(output_path, file_path)
            # include_textured_submeshes(output_path)
        if light_settings['render_background']:
            render_background(foreground_img_dir=os.path.join(output_path, split_name))
    if data_config['include_metadata']:       
        include_metadata(output_path, file_path, **data_config)
        export_transformation_matrix(output_path, file_path, data_config)
    
         
    