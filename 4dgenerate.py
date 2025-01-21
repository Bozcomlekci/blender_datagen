import sys, os
import json
import bpy
import mathutils
import numpy as np
from scene_light import setLight_sun, setLight_ambient, shadowThreshold, invisibleGround
from utils.blender_utils import renderImage, blenderInit, readMesh, setCamera, std_of_mesh, setCamera_orthographic, parent_obj_to_camera
from utils.utils import listify_matrix, transform_save
from math import radians
import glob
import argparse
from cam_path import cube8corners



def init_blender(imgRes_x, imgRes_y, num_samples, light_settings, use_GPU=True, clean_scene=True):
    blenderInit(imgRes_x, imgRes_y, num_samples, use_GPU, clean_scene=clean_scene)

    # Set lighting
    bpy.ops.object.shade_smooth()
    light_settings = json.load(open(light_settings, 'r'))
    if light_settings['use_sun']:
        lightAngle = light_settings['light_angle']
        strength = light_settings['sun_strength']
        shadowSoftness = light_settings['shadow_softness']
        sun = setLight_sun(lightAngle, strength, shadowSoftness)
        # set ambient light
        color = light_settings['ambient_light_color']
        setLight_ambient(color=color)

    # set gray shadow to completely white with a threshold (optional but recommended)
    shadowThreshold(alphaThreshold=light_settings['alpha_threshold'], interpolationMode=light_settings['interpolation_mode'])


def init_cam_trajectory(file_path, radius, camera_type, focalLength, views, timespan, data_settings, orthographic_window_size=1024):
    # Calculate scale factor with respect to the sample mesh
    mesh = readMesh(file_path, data_settings=data_settings) 
    scale_factor = std_of_mesh(mesh)
    # remove mesh from scene
    bpy.data.objects.remove(mesh, do_unlink=True)
        
    scene = bpy.context.scene
    camLocation =  (0,radius*scale_factor,0)
    cam = setCamera(camLocation, focalLength=focalLength)
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

    stepsize = 360 / views

    out_data['frames'] = []
    return cam, b_empty, out_data, scene, stepsize

def render_sequence(samples_folder, split_name='train', 
                    cam_settings='cam_settings/custom.json', light_settings='light_settings/settings.json', data_settings='data_settings/settings.json',
                    num_samples=200, output_path=None, fourd_fpt=0, time_span=1.0):
    

    config = os.path.basename(cam_settings).split('.')[0]
    cam_settings = json.load(open(cam_settings, 'r'))
    data_settings = json.load(open(data_settings, 'r'))
    imgRes_x = cam_settings['resolution']
    imgRes_y = cam_settings['resolution']
    file_name = os.path.basename(samples_folder)

    if output_path is None:
        output_path = os.path.join('nvs_data', f'{file_name}_{config}_fpt{fourd_fpt}')
    fp = bpy.path.abspath(os.path.join(output_path, split_name))
    if not os.path.exists(fp):
        os.makedirs(fp)

    file_types = data_settings['data_types']
    path_list = []
    
    for file_type in file_types:
        # Retrieve subfolders that contain a sequence of ply files <samples_folder>/<subfolder>/*.ply
        files = glob.glob(f"{samples_folder}/**/*.{file_type}", recursive=True)
        if files:
            # define sorting criterion
            def key(s):
                return int(os.path.basename(s).split('.')[-2].split('f')[-1])
            files.sort(key=key)
            # keep consecutive files only
            path_list = []
            end_index = 0
            for i in range(len(files)-1):
                if key(files[i+1]) - key(files[i]) == 1:
                    path_list.append(files[i])
                    end_index = i+1
                else:
                    break
            print(f"Sequence till {end_index}th file")
        else:
            print(f"No {file_type} files found in {samples_folder}")
    if not path_list:
        raise FileNotFoundError(f"No subfolder containing a sequence of ply files found in {samples_folder}")

    init_blender(imgRes_x, imgRes_y, num_samples, light_settings)
    cam, b_empty, out_data, scene, stepsize = init_cam_trajectory(path_list[0], cam_settings['radius'], cam_settings['camera_type'], cam_settings['focalLength'],
                        views=len(path_list)*fourd_fpt, timespan=time_span, data_settings=data_settings)
    time_step = time_span / len(path_list)
    for scan_idx, file_path in enumerate(path_list):
        mesh = readMesh(file_path, data_settings=data_settings) # Automatically inserted into scene
        for i in range(0, fourd_fpt):
            sample_idx = scan_idx*fourd_fpt + i
            if config == 'spherical':
                # Rotate camera around the object spherically
                rot = np.random.uniform(0, 1, size=3) * (1,-2*np.pi,2*np.pi)
                rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
                b_empty.rotation_euler = rot
            elif config == 'circular':
                # Rotate camera around the object in a circle
                b_empty.rotation_euler[2] += radians(stepsize)
            elif config == '4dcustom':
                b_empty.rotation_euler[2] += radians(stepsize)
                b_empty.rotation_euler[0] += radians(np.sin(sample_idx * stepsize * 2 * np.pi))
                #b_empty.rotation_euler[1] += radians(np.cos(sample_idx * stepsize * 2 * np.pi)) * 0.1
            elif config == 'cube8corners':
                cube8corners(b_empty, i)
            elif config == 'frontbackside':
                if i == 0:
                    b_empty.rotation_euler[2] = 0
                elif i == 1:
                    b_empty.rotation_euler[2] += radians(90)
                elif i == 2:
                    b_empty.rotation_euler[2] += radians(90)
                else:
                    NotImplementedError
            elif config == 'monocular':
                pass
            else:
                # Invalid config 
                NotImplementedError
            scene.render.filepath = os.path.join(fp, ("{:04d}".format(sample_idx)))
            outputPath = scene.render.filepath + '.png'
            renderImage(cam, outputPath=outputPath) 

            frame_data = {
                'file_path': split_name + '/' + ("{:04d}".format(sample_idx)),
                'rotation': radians(stepsize),
                'transform_matrix': listify_matrix(cam.matrix_world),
                'time': scan_idx * time_step
            }
            out_data['frames'].append(frame_data)
        # remove mesh from scene
        bpy.data.objects.remove(mesh, do_unlink=True)
    transform_save(out_data, output_path, split_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples_folder', type=str, default='samples1')
    parser.add_argument('--split_name', type=str, default='train')
    parser.add_argument('--data_settings', type=str, default='data_settings/settings.json')
    parser.add_argument('--cam_settings', type=str, default='cam_settings/4dcustom.json')
    parser.add_argument('--light_settings', type=str, default='light_settings/settings.json')
    parser.add_argument('--fourd_fpt', type=int, default=0) # frame per timestamp
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--time_span', type=float, default=1.0)
    parser.add_argument('--num_samples', type=int, default=200)
    args = parser.parse_args()

    render_sequence(samples_folder=args.samples_folder, split_name=args.split_name,
                    cam_settings=args.cam_settings, light_settings=args.light_settings, data_settings=args.data_settings,
                    output_path=args.output_path, fourd_fpt=args.fourd_fpt, time_span=args.time_span)
    