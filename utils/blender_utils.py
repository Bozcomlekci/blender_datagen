import bpy
import numpy as np
import bmesh
import random
import os
import mathutils
from math import *
from pathlib import Path
import json

from utils.bprocutil import set_world_background_color, colorize_objects_for_instance_segmentation, colorize_object
from utils.bprocutil import render_segmap
from blenderproc.python.utility.BlenderUtility import get_all_blender_mesh_objects
from blenderproc.python.material import MaterialLoaderUtility
from blenderproc.python.renderer import RendererUtility
from blenderproc.python.utility.Utility import Utility, UndoAfterExecution

from utils.utils import npz_to_ply
from assets.constants import COLOR_PALETTE, COLOR_PALETTE_0_1
from utils.numeric import tuple_uint8_2_tuple_float
import open3d
import trimesh
from mathutils import Vector

# https://github.com/HTDerekLiu/BlenderToolbox/blob/master/BlenderToolBox

def blenderInit(resolution_x, resolution_y, numSamples = 200, exposure = 1.5, use_GPU = True,
				 resolution_percentage = 100, clean_scene = False, format = 'PNG', color_depth = '8',
				 max_bounces = 6, mask_render = False, render_background=False):
	# clear all
	bpy.ops.wm.read_homefile()
	if clean_scene:
		bpy.ops.object.select_all(action = 'SELECT')
		bpy.ops.object.delete()

	if mask_render:
		RendererUtility.render_init()
		# bpy.context.scene.render.engine = 'BLENDER_EEVEE'
        # the amount of samples must be one and there can not be any noise threshold
		RendererUtility.set_max_amount_of_samples(1)
		RendererUtility.set_noise_threshold(0)
		RendererUtility.set_denoiser(None)
		RendererUtility.set_light_bounces(0, 0, 0, 0, 0, 0, 0)
		bpy.data.scenes["Scene"].view_settings.view_transform = 'Raw'
		
		# Remove anti-aliasing
		bpy.context.scene.display.render_aa = 'OFF'
		bpy.context.scene.display.viewport_aa = 'OFF'
		bpy.context.scene.display.shading.color_type = 'VERTEX'
		# Remove shadows, cavity, and denoising
		bpy.context.scene.display.shading.show_shadows = False
		bpy.context.scene.display.shading.show_cavity = False
		bpy.context.scene.cycles.filter_width = 0.0
		bpy.context.scene.cycles.use_denoising = False
		bpy.context.scene.cycles.use_adaptive_sampling = False

		bpy.context.scene.render.resolution_x = resolution_x 
		bpy.context.scene.render.resolution_y = resolution_y
		RendererUtility.set_output_format("PNG", 16)
	else:
		# use cycle
		bpy.context.scene.render.engine = 'CYCLES'
		bpy.context.scene.render.resolution_x = resolution_x 
		bpy.context.scene.render.resolution_y = resolution_y 
		
		bpy.context.scene.cycles.samples = numSamples 
		bpy.context.scene.cycles.max_bounces = max_bounces
		bpy.context.scene.render.resolution_percentage = resolution_percentage
		if format == 'PNG':
			bpy.context.scene.render.image_settings.color_mode = 'RGBA'
		else:
			bpy.context.scene.render.image_settings.color_mode = 'RGB'
		bpy.context.scene.render.image_settings.color_depth = str(color_depth)
		bpy.context.scene.cycles.film_exposure = exposure
		bpy.data.scenes["Scene"].render.film_transparent = True # if not render_background else False
		bpy.data.scenes[0].view_layers[0]['cycles']['use_denoising'] = 0
		

	# set devices
	cyclePref  = bpy.context.preferences.addons['cycles'].preferences
	for dev in cyclePref.devices:
		print("using rendering device", dev.name, ":", dev.use)
	if use_GPU:
		# https://blender.stackexchange.com/questions/104651/selecting-gpu-with-python-script
		bpy.context.scene.cycles.device = "GPU"
		# Set the device_type
		bpy.context.preferences.addons[
			"cycles"
		].preferences.compute_device_type = "CUDA" # or "OPENCL"

		# get_devices() to let Blender detects GPU device
		bpy.context.preferences.addons["cycles"].preferences.get_devices()
		print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
		for d in bpy.context.preferences.addons["cycles"].preferences.devices:
			d["use"] = 1 # Using all devices, include GPU and CPU
			# print(d["name"], d["use"])
	else:
		bpy.context.scene.cycles.device = "CPU"
	print("cycles rendering with:", bpy.context.scene.cycles.device)
	return 0


def setMat_VColor(mesh, meshVColor):
	mat = bpy.data.materials.new('MeshMaterial')
	mesh.data.materials.append(mat)
	mesh.active_material = mat
	mat.use_nodes = True
	tree = mat.node_tree

	# read vertex attribute
	tree.nodes.new('ShaderNodeAttribute')
	tree.nodes[-1].attribute_name = "Col"
	HSVNode = tree.nodes.new('ShaderNodeHueSaturation')
	tree.links.new(tree.nodes["Attribute"].outputs['Color'], HSVNode.inputs['Color'])
	HSVNode.inputs['Saturation'].default_value = meshVColor.S
	HSVNode.inputs['Value'].default_value = meshVColor.V
	HSVNode.inputs['Hue'].default_value = meshVColor.H
	HSVNode.location.x -= 200

	# set color brightness/contrast
	BCNode = tree.nodes.new('ShaderNodeBrightContrast')
	BCNode.inputs['Bright'].default_value = meshVColor.B
	BCNode.inputs['Contrast'].default_value = meshVColor.C
	BCNode.location.x -= 400

	# set principled BSDF
	tree.nodes["Principled BSDF"].inputs['Roughness'].default_value = 1.0
	try:
		tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = 0
	except:
		# ValueError: bpy_struct: item.attr = val: sequence expected at dimension 1, not 'float'
		tree.nodes["Principled BSDF"].inputs['Sheen Tint'].default_value = (0.0, 0.0, 0.0, 0.0)
	tree.links.new(HSVNode.outputs['Color'], BCNode.inputs['Color'])
	tree.links.new(BCNode.outputs['Color'], tree.nodes['Principled BSDF'].inputs['Base Color'])

def setTexture(mesh, texture_image_path=None):
	# Ensure the object has a material slot
	if not mesh.data.materials:
		# Create a new material and assign it to the object
		mat = bpy.data.materials.new(name="Material")
		mesh.data.materials.append(mat)
	else:
		# Use the existing material
		mat = mesh.data.materials[0]

	# Enable 'Use Nodes' for the material
	mat.use_nodes = True

	# Get the material's node tree
	nodes = mat.node_tree.nodes
	links = mat.node_tree.links

	# Clear existing nodes
	for node in nodes:
		nodes.remove(node)

	# Add necessary nodes
	output_node = nodes.new(type="ShaderNodeOutputMaterial")
	output_node.location = (400, 0)

	principled_bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
	principled_bsdf.location = (0, 0)

	image_texture_node = nodes.new(type="ShaderNodeTexImage")
	image_texture_node.location = (-400, 0)

	# Load the texture image
	try:
		parent_dir = os.path.dirname(texture_image_path)
		if os.path.exists(texture_image_path):
			image = bpy.data.images.load(texture_image_path)
		elif os.path.exists(os.path.join(parent_dir, 'material0.jpeg')):
			image = bpy.data.images.load(os.path.join(parent_dir, 'material0.jpeg'))
		else:
			image = bpy.data.images.load(texture_image_path.replace('.jpg', '.jpeg'))
		image_texture_node.image = image
	except:
		print("Texture image not found")
	# Link the nodes
	links.new(image_texture_node.outputs["Color"], principled_bsdf.inputs["Base Color"])
	links.new(principled_bsdf.outputs["BSDF"], output_node.inputs["Surface"])

	# Assign the material to the object
	mesh.active_material = mat

def readMesh(filePath, location=(0.0, 0.0, 0.0), rotation_euler=(0, 0, 0), scale=(1, 1, 1), VColorMask = False,
			 data_settings = None):
	_, extension = os.path.splitext(filePath)
	if data_settings is not None:
		rotation_euler = data_settings['rotation_euler']
		scale = data_settings['scale']
		location = data_settings['location']
	if extension == '.ply' or extension == '.PLY':
		mesh = readPLY(filePath, location, rotation_euler, scale, data_settings = data_settings)
		if VColorMask:
			# Precalculate the scale of the mesh before partititioning for mask calculation
			scale_factor = std_of_mesh(mesh) if 'distance_by_std' in data_settings and data_settings['distance_by_std'] else 1 / get_scale(mesh)
			setMask_VColor(mesh, data_settings = data_settings)
		else:
			meshVColor = colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
			setMat_VColor(mesh, meshVColor)
	elif extension == '.obj' or extension == '.OBJ':
		if VColorMask:
			ply_path = filePath.replace('.obj', '.ply') # Corresponding vertex colored ply should be available
			mesh, scale_factor = readMesh(ply_path, location, rotation_euler, scale, VColorMask, data_settings = data_settings)
		else:
			mesh = readOBJ(filePath, location, rotation_euler, scale, data_settings=data_settings)
			setTexture(mesh, texture_image_path=filePath.replace('.obj', '.jpg'))
	elif extension == '.npz' or extension == '.NPZ':
		if VColorMask:
			ply_path = npz_to_ply(filePath, VColorMask, color_mode=data_settings['color_mode'], data_settings = data_settings)
			mesh, scale_factor = readMesh(ply_path, location, rotation_euler, scale, VColorMask, data_settings=data_settings)
		else:
			ply_path = npz_to_ply(filePath, VColorMask, color_mode=data_settings['color_mode'])
			mesh = readMesh(ply_path, location, rotation_euler, scale, VColorMask, data_settings = data_settings)
		# Remove the temporary ply file
		os.remove(ply_path)
	elif data_settings is not None and data_settings['render_folder_together']:
		delete_all_objects()
		import_objects_from_folder(filePath, data_settings)
		if VColorMask:
			data_settings['labels_to_colors'], data_settings['unique_colors'] = assign_colors_to_objects()
		file_name = Path(filePath).name
		parent = Path(filePath).parent
		output_path = os.path.join(parent, file_name + '.ply')
		merged_mesh = merge_into_single_mesh(output_path, out=True)
		if VColorMask:
			mesh, scale_factor = readMesh(output_path, location, rotation_euler, scale, VColorMask, data_settings = data_settings)
			bpy.data.objects.remove(merged_mesh)
		else:
			dummy = readMesh(output_path, location, rotation_euler, scale, VColorMask, data_settings = data_settings)
			# Remove dummy mesh
			bpy.data.objects.remove(dummy)
			mesh = merged_mesh
	else:
		raise TypeError("only support .ply, .obj, and .stl for now")
	if VColorMask:
		return mesh, scale_factor
	else:
		bpy.context.view_layer.objects.active = mesh
		bpy.ops.object.shade_flat() # default flat shading
		return mesh

	
def readOBJ(filePath, location, rotation_euler, scale, normalize=True, data_settings = None):
	x = rotation_euler[0] * 1.0 / 180.0 * np.pi 
	y = rotation_euler[1] * 1.0 / 180.0 * np.pi 
	z = rotation_euler[2] * 1.0 / 180.0 * np.pi 
	angle = (x,y,z)

	prev = []
	for ii in range(len(list(bpy.data.objects))):
		prev.append(bpy.data.objects[ii].name)
	bpy.ops.import_scene.obj(filepath=filePath, split_mode='OFF')
	after = []
	for ii in range(len(list(bpy.data.objects))):
		after.append(bpy.data.objects[ii].name)
	name = list(set(after) - set(prev))[0]
	mesh = bpy.data.objects[name]

	if normalize:
		mesh.rotation_euler = angle
		bpy.context.view_layer.update()
		unit_scale = get_scale(mesh)
		scale = (scale[0] * unit_scale, scale[1] * unit_scale, scale[2] * unit_scale)
		mesh.scale = scale
		bpy.context.view_layer.update()
		center, _ = get_center(mesh)
		location = (location[0] - center[0], location[1] - center[1], location[2] - center[2])
		mesh.location = location
		data_settings['center'] = list(center)
		data_settings['unit_scale'] = unit_scale
	else:
		mesh.location = location
		mesh.rotation_euler = angle
		mesh.scale = scale
	bpy.context.view_layer.update()
	
	return mesh 

def readPLY(filePath, location, rotation_euler, scale, normalize=True, data_settings = None):
	# example input types:
	# - location = (0.5, -0.5, 0)
	# - rotation_euler = (90, 0, 0)
	# - scale = (1,1,1)
	x = rotation_euler[0] * 1.0 / 180.0 * np.pi 
	y = rotation_euler[1] * 1.0 / 180.0 * np.pi 
	z = rotation_euler[2] * 1.0 / 180.0 * np.pi 
	angle = (x,y,z)

	prev = []
	for ii in range(len(list(bpy.data.objects))):
		prev.append(bpy.data.objects[ii].name)
	
	if filePath is None: # import a dummy cube
		bpy.ops.mesh.primitive_cube_add(size=2)
	else:
		try:
			bpy.ops.import_mesh.ply(filepath=filePath)
		except:
			bpy.ops.wm.ply_import(filepath=filePath)

	after = []
	for ii in range(len(list(bpy.data.objects))):
		after.append(bpy.data.objects[ii].name)
	name = list(set(after) - set(prev))[0]
	# filePath = filePath.rstrip(os.sep) 
	# name = os.path.basename(filePath)
	# name = name.replace('.ply', '')
	mesh = bpy.data.objects[name]
	# print(list(bpy.data.objects))
	# mesh = bpy.data.objects[-1]


	if normalize:
		mesh.rotation_euler = angle
		bpy.context.view_layer.update()
		unit_scale = get_scale(mesh)
		scale = (scale[0] * unit_scale, scale[1] * unit_scale, scale[2] * unit_scale)
		mesh.scale = scale
		bpy.context.view_layer.update()
		center, _ = get_center(mesh)
		location = (location[0] - center[0], location[1] - center[1], location[2] - center[2])
		mesh.location = location
		data_settings['center'] = list(center)
		data_settings['unit_scale'] = unit_scale
	else:
		mesh.location = location
		mesh.rotation_euler = angle
		mesh.scale = scale
	bpy.context.view_layer.update()
	return mesh 

def rotate_objs(objs, data_settings, normalize=True):
	location=data_settings['location']; rotation_euler=data_settings['rotation_euler']; scale=data_settings['scale']
	center=data_settings['center']; unit_scale=data_settings['unit_scale']
	x = rotation_euler[0] * 1.0 / 180.0 * np.pi 
	y = rotation_euler[1] * 1.0 / 180.0 * np.pi 
	z = rotation_euler[2] * 1.0 / 180.0 * np.pi 
	angle = (x,y,z)
	# rotate objects
	if normalize:
		for obj in objs:
			obj.rotation_euler = angle
			bpy.context.view_layer.update()
			scale = (scale[0] * unit_scale, scale[1] * unit_scale, scale[2] * unit_scale)
			obj.scale = scale
			bpy.context.view_layer.update()
			location = (location[0] - center[0], location[1] - center[1], location[2] - center[2])
			obj.location = location
	else:
		for obj in objs:
			obj.location = location
			obj.rotation_euler = angle
			obj.scale = scale
	return objs

def setCamera(camLocation, lookAtLocation = (0,0,0), focalLength = 35):
	# initialize camera
	bpy.ops.object.camera_add(location = camLocation) # name 'Camera'
	cam = bpy.context.object
	cam.data.lens = focalLength
	loc = mathutils.Vector(lookAtLocation)
	lookAt(cam, loc)
	return cam

def setCamera_orthographic(camLocation, top, bottom, left, right, lookAtLocation = (0,0,0)):
	# scale the resolution y using resolution x
	assert(abs(left-right)>0)
	assert(abs(top-bottom)>0)
	aspectRatio = abs(right - left)*1.0 / abs(top - bottom)
	bpy.context.scene.render.resolution_y = int(bpy.context.scene.render.resolution_x / aspectRatio)

	bpy.ops.object.camera_add(location = camLocation)
	cam = bpy.context.object
	bpy.context.object.data.type = 'ORTHO'
	cam.data.ortho_scale = abs(left-right)
	loc = mathutils.Vector(lookAtLocation)
	lookAt(cam, loc)
	return cam

def lookAt(camera, point):
	direction = point - camera.location
	rotQuat = direction.to_track_quat('-Z', 'Y')
	camera.rotation_euler = rotQuat.to_euler()

def std_of_mesh(mesh):
	# Given a mesh which is a bpy.data.object, return the std of the mesh vertex coordinates
	# https://devtalk.blender.org/t/position-of-selected-point-edge-face/9680
	# select the object
	mesh.select_set(True)
	# set the object as the active object
	bpy.context.view_layer.objects.active = mesh
	bpy.ops.object.mode_set(mode='EDIT') 
	bm = bmesh.from_edit_mesh(mesh.data)

	points = []
	for v in bm.verts:
		if (v.select == True):
			obMat = mesh.matrix_world
			points.append(obMat @ v.co)
	#Here, we multiply the vertex coordinate by the object's world matrix, in case the object is
	# transformed.
	#It is important to put the obMat before the v.co
	# the @ (matrix multiplication) operator is NOT commutative.
	# calculate the std along the x, y, z axis
	points = np.array(points)
	stdx = np.std(points[:,0])
	stdy = np.std(points[:,1])
	stdz = np.std(points[:,2])
	std = max(stdx, stdy, stdz) # stdz (height) is max often times
	bpy.ops.object.mode_set(mode='OBJECT')
	return std

def renderImage(camera, outputPath='test.png'):
    bpy.data.scenes['Scene'].render.filepath = outputPath
    bpy.data.scenes['Scene'].camera = camera
    bpy.ops.render.render(write_still = True)

def split_mesh(obj, unique_colors):
	color_data = obj.data.vertex_colors.active.data
	def discriminant(face):
		vertex_colors = []
		for i, loop_index in enumerate(face.loop_indices):
			vertex_color = color_data[loop_index].color 
			vertex_colors.append(tuple([c for c in vertex_color]))
		# Check if the dominant color of the face is the same as the color
		dominant = max(set(vertex_colors), key=vertex_colors.count)
		# Round the tuple dominant
		dominant = tuple([round(c, 3) for c in dominant])
		# try:
		# Sometimes the dominant color is not in the unique colors
		tag = unique_colors[dominant] # list(unique_colors.keys())[0][0]
		return tag
	accumulator = {}

	for polygon in obj.data.polygons:
		tag = discriminant(polygon)

		acc = accumulator.get(tag)
		if acc is None:
			acc = {
				"tag":tag,
				"verts":[],
				"faces":[],
				"materials":[],
				"vertMap": {}
			}
			accumulator[tag] = acc

		face = []
		for vi in polygon.vertices:
			vi2 = acc["vertMap"].get(vi)
			if vi2 is None:
				verts = acc["verts"]
				vi2 = len( verts )
				acc["vertMap"][vi] = vi2
				verts.append(obj.data.vertices[vi].co)
			face.append(vi2)
		acc["faces"].append(face)
		acc["materials"].append(polygon.material_index)

	rval = []
	for tag,acc in accumulator.items():
		print(tag)

		mesh = bpy.data.meshes.new(tag)
		#print(acc)
		mesh.from_pydata(acc["verts"], [], acc["faces"])
		mesh.validate(verbose=True)

		for mat in obj.data.materials:
			mesh.materials.append(mat)

		for i in range(len(mesh.polygons)):
			mesh.polygons[i].material_index = acc["materials"][i]

		obj = bpy.data.objects.new(tag, mesh)
		#bpy.context.scene.objects.link(obj)
		bpy.context.collection.objects.link(obj)
		rval.append(obj)

	return rval

def setMask_VColor(mesh, data_settings=None):
	
	if data_settings['render_silhouettes']:
		pass
	else:
		# Extract all unique colors from the object
		l2c = {i: tuple_uint8_2_tuple_float(c) for i, c in enumerate(data_settings['labels_to_colors'])}
		c2l = {c: l for l, c in l2c.items()}
		if data_settings is not None and data_settings['unique_colors'] is not None:
			unique_colors = {tuple_uint8_2_tuple_float(c): f'color_{c2l[tuple_uint8_2_tuple_float(c)]}' for c in data_settings['unique_colors']}
		else:
			colors = mesh.data.vertex_colors["Col"].data
			unique_colors = set()
			for poly in mesh.data.polygons:
				for i, loop_index in enumerate(poly.loop_indices):
					vertex_color = colors[loop_index].color 
					unique_colors.add(tuple([c for c in vertex_color]))
			unique_colors = {color: f'color_{c2l[color]}' for color in unique_colors}
		print("Unique colors:", unique_colors, '\n', 'Number of unique colors:', len(unique_colors))

		objs = split_mesh(mesh, unique_colors=unique_colors)
		if data_settings is not None and not data_settings['render_folder_together']:
			rotate_objs(objs, data_settings=data_settings)
		# Remove the original object
		bpy.data.objects.remove(mesh)

	# Get objects with meshes (i.e. not lights or cameras)
	objs_with_mats = get_all_blender_mesh_objects()

	# Set some category ids for loaded objects 
	for j, obj in enumerate(objs_with_mats):
		obj["category_id"] = j+1
		if isinstance(obj["category_id"], float) or isinstance(obj["category_id"], int):
			Utility.insert_keyframe(obj, "[\"" + "category_id" + "\"]", None)

	if data_settings['render_silhouettes']:
		# White foreground and black background
		rgb_unique_colors = [(1,1,1), (0,0,0)]
	else:
		# Sort the objects by the name of the object
		objs_with_mats = sorted(objs_with_mats, key=lambda x: x.name)
		unique_colors = sorted(unique_colors.items(), key=lambda x: x[1])
		unique_colors = dict(unique_colors)
		unique_colors[(0.0, 0.0, 0.0, 1.0)] = 'color_0' # add black color for the background
		rgb_unique_colors = [color[:3] for color in unique_colors.keys()]

	if len(rgb_unique_colors) is not len(objs_with_mats)+1:
		AssertionError("Number of unique colors does not match the number of objects")
		# diff = len(objs_with_mats) - len(rgb_unique_colors)
		# rgb_unique_colors.extend([(0, 0, 0.1) * (i+1) for i in range(diff)])

	colorize_objects_for_instance_segmentation(objs_with_mats, rgb_unique_colors, use_alpha_channel=False)

	bpy.context.scene.cycles.filter_width = 0.0
	bpy.data.scenes["Scene"].render.filter_size = 0.0

class colorObj(object):
    def __init__(self, RGBA, \
    H = 0.5, S = 1.0, V = 1.0,\
    B = 0.0, C = 0.0):
        self.H = H # hue
        self.S = S # saturation
        self.V = V # value
        self.RGBA = RGBA
        self.B = B # brightness
        self.C = C # contrast

# Function to check file extension (modify if needed)
def is_valid_object_file(filepath):
  return filepath.lower().endswith((".obj", ".fbx", ".blend"))

def delete_all_objects():
    # if object is mesh type, delete it
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            bpy.data.objects.remove(obj)

def import_objects_from_folder(folder_path, data_settings):
    # Loop through files in the folder
    location = data_settings['location']
    rotation_euler = data_settings['rotation_euler']
    scale = data_settings['scale']
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        
        # Check if valid object file
        if is_valid_object_file(filepath):
            print(f"Importing object from: {filepath}")
            # Import the object using the appropriate operator based on extension
            if filepath.lower().endswith(".obj"):
                obj = readOBJ(filepath, location=location, rotation_euler=rotation_euler, scale=scale, data_settings=data_settings, normalize=False)
                setTexture(obj, texture_image_path=filepath.replace('.obj', '.jpg'))
            elif filepath.lower().endswith(".ply"):
                obj = readPLY(filepath, location=location, rotation_euler=rotation_euler, scale=scale, data_settings=data_settings, normalize=False)
                meshVColor = colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
                setMat_VColor(obj, meshVColor)
            # Add more import operators for other extensions (e.g., bpy.ops.import_mesh.blend for .blend)
            else:
                print(f"Unsupported file format: {filepath}")

    #Deselect all
    bpy.ops.object.select_all(action='DESELECT')

    #Mesh objects
    mesh_objs = [m for m in bpy.context.scene.objects if m.type == 'MESH']

    for mesh_obj in mesh_objs:
        #Select all mesh objects
        mesh_obj.select_set(state=True)
        #Makes one active
        bpy.context.view_layer.objects.active = mesh_obj
    
def merge_into_single_mesh(output_path, out=False):
	# bpy context for Blender operations
	context = bpy.context
	# Select all objects in the scene
	bpy.ops.object.select_all(action='SELECT')
	# Join all selected objects into a single object
	bpy.ops.object.join()
	joined_mesh = context.active_object
	# Scale the object to fit within the unit cube
	# bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
	# Export the merged object as a PLY file
	# Preserve colors and normals
	if out:
		bpy.ops.export_mesh.ply(filepath=output_path, use_colors=True, use_normals=True)
	# Delete the objects except the merged object
	for obj in bpy.data.objects:
		if obj != joined_mesh and obj.type == 'MESH':
			bpy.data.objects.remove(obj)
	# Deselect all objects
	# bpy.ops.object.select_all(action='DESELECT')
	print("Merged all objects into a single mesh")
	return joined_mesh


def assign_colors_to_objects():
	def generate_random_color():
		return (random.random(), random.random(), random.random())

	unique_colors = []
	for i, obj in enumerate(bpy.data.objects):
		# Create a new material with random color
		if obj.type != 'MESH':
			continue

		if obj.mode == 'EDIT':
			bpy.ops.object.mode_set()

		if i >= len(COLOR_PALETTE_0_1):
			# Define color
			color = generate_random_color()
		else:
			color = COLOR_PALETTE_0_1[i]
			unique_colors.append(COLOR_PALETTE[i] + [255])
		print(f"Assigning color {color} to object: {obj.name} with index {i}")

		# colorize_object(obj, color, False)

		#property uchar red, green, blue, alpha should be present in the ply file
		# Convert the Color attribute to Byte Color under the Attributes
		rgba_color = (color[0], color[1], color[2], 1)
		if 'Col' not in obj.data.vertex_colors:
			obj.data.vertex_colors.new(name='Col')
			obj.data.update()
			# Edit mode to assign the color
			obj.data.vertex_colors['Col'].active = True
			for poly in obj.data.polygons:
				for loop_index in poly.loop_indices:
					obj.data.vertex_colors['Col'].data[loop_index].color = rgba_color
		else:
			for poly in obj.data.polygons:
				for loop_index in poly.loop_indices:
					obj.data.vertex_colors['Col'].data[loop_index].color = rgba_color
		# # Remove any existing materials (optional)
		# if len(obj.data.materials) > 1:
		#     for slot in obj.material_slots:
		#         bpy.data.materials.remove(slot.material)
		#         obj.data.materials.remove(slot)
	labels_to_colors = [ item + [255] for item in COLOR_PALETTE]
	return labels_to_colors, unique_colors

		
					
def scene_objects_to_ply(output_path, file_name='points3d.ply', export_objs=False):
	# Unselect all objects
	bpy.ops.object.select_all(action='DESELECT')
	if export_objs:
		for obj in bpy.data.objects:
			obj.select_set(False)
			# disable object visibility
			obj.hide_render = True
		# Loop through objects in the scene
		for obj in bpy.data.objects:
			# Check if the object is a mesh
			if obj.type == 'MESH':
				# Select the object
				obj.select_set(True)
				obj.hide_render = False
				# Export the object as a PLY file
				bpy.ops.export_mesh.ply(filepath=output_path + f'/{obj.name}.ply', use_colors=True, use_normals=True, use_selection=True)
				# Deselect the object
				obj.select_set(False)
	else:
		# Select all objects in the scene
		bpy.ops.object.select_all(action='SELECT')
		# Export the merged object as a PLY file
		bpy.ops.export_mesh.ply(filepath=output_path + f'/{file_name}', use_colors=True, use_normals=True)
		# Deselect all objects
		bpy.ops.object.select_all(action='DESELECT')


def include_file(output_path, file_path, export_objs=False):
	# Remove the plane added by shadow catcher if exists bpy.ops.mesh.primitive_plane_add(location = location, size = groundSize)
	if bpy.data.objects.get("Plane"):
		bpy.data.objects.remove(bpy.data.objects["Plane"])
	if export_objs:
		scene_objects_to_ply(output_path, export_objs=export_objs)
	else:
		scene_objects_to_ply(output_path)

def export_transformation_matrix(output_path, file_path, data_settings):
	# Cube at the origin
	delete_all_objects()
	rotation_euler = data_settings['rotation_euler']
	scale = data_settings['scale']
	location = data_settings['location']
	mesh = readPLY(None, location, rotation_euler, scale, data_settings=data_settings) # Reads a dummy object for extracting transformation matrix
	transform_matrix = np.array(mesh.matrix_world, dtype=np.float32).tolist()
	# open metadata.json file and write the transformation matrix
	metadata = json.load(open(output_path + '/metadata.json', 'r')) if os.path.exists(output_path + '/metadata.json') else {}
	metadata['transformation_matrix'] = transform_matrix
	with open(output_path + '/metadata.json', 'w') as f:
		json.dump(metadata, f, indent=4)

def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty
			
def get_center(o):
	local_bbox_center = 0.125 * sum((Vector(b) for b in o.bound_box), Vector())
	global_bbox_center = o.matrix_world @ local_bbox_center
	return global_bbox_center, local_bbox_center

def get_scale(o):
	s = 1/max(o.dimensions.to_tuple())
	return s

def get_min_max(o, dim=2):
	# Using the bounding box of the object, find the min and max of the object in the specified dimension
	bbox = o.bound_box
	bbox = [o.matrix_world @ Vector(b) for b in bbox]
	o_min = min([b[dim] for b in bbox])
	o_max =	max([b[dim] for b in bbox])
	return o_min, o_max
