import bpy
import numpy as np

# https://github.com/HTDerekLiu/BlenderToolbox/blob/master/BlenderToolBox

def setLight_sun(rotation_euler, strength, shadow_soft_size = 0.05):
    # https://github.com/HTDerekLiu/BlenderToolbox/blob/master/BlenderToolBox/setLight_sun.py
    x = rotation_euler[0] * 1.0 / 180.0 * np.pi 
    y = rotation_euler[1] * 1.0 / 180.0 * np.pi 
    z = rotation_euler[2] * 1.0 / 180.0 * np.pi 
    angle = (x,y,z)
    bpy.ops.object.light_add(type = 'SUN', rotation = angle)
    lamp = bpy.data.lights['Sun']
    lamp.use_nodes = True
    # lamp.shadow_soft_size = shadow_soft_size # this is for older blender 2.8
    lamp.angle = shadow_soft_size

    lamp.node_tree.nodes["Emission"].inputs['Strength'].default_value = strength
    return lamp


def setLight_ambient(color = (0,0,0,1)):
    # https://github.com/HTDerekLiu/BlenderToolbox/blob/master/BlenderToolBox/setLight_sun.py
	bpy.data.scenes[0].world.use_nodes = True
	bpy.data.scenes[0].world.node_tree.nodes["Background"].inputs['Color'].default_value = color
      

def shadowThreshold(alphaThreshold, interpolationMode = 'CARDINAL'):
    # https://github.com/HTDerekLiu/BlenderToolbox/blob/master/BlenderToolBox/shadowThreshold.py
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    RAMP = tree.nodes.new('CompositorNodeValToRGB')
    RAMP.color_ramp.elements[0].color[3] = 0
    RAMP.color_ramp.elements[0].position = alphaThreshold
    RAMP.color_ramp.interpolation = interpolationMode

    REND = tree.nodes["Render Layers"]
    OUT = tree.nodes["Composite"]
    tree.links.new(REND.outputs[1], RAMP.inputs[0])
    tree.links.new(RAMP.outputs[1], OUT.inputs[1])

def invisibleGround(location = (0,0,0), groundSize = 100, shadowBrightness = 0.7):
	# https://github.com/HTDerekLiu/BlenderToolbox/blob/master/BlenderToolBox/invisibleGround.py
	# initialize a ground for shadow
	bpy.context.scene.cycles.film_transparent = True
	obj = bpy.ops.mesh.primitive_plane_add(location = location, size = groundSize)
	try:
		bpy.context.object.is_shadow_catcher = True # for blender 3.X
	except:
		bpy.context.object.cycles.is_shadow_catcher = True # for blender 2.X

	# # set material
	ground = bpy.context.object
	mat = bpy.data.materials.new('MeshMaterial')
	ground.data.materials.append(mat)
	mat.use_nodes = True
	tree = mat.node_tree
	tree.nodes["Principled BSDF"].inputs['Transmission'].default_value = shadowBrightness
	return ground

