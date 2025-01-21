# Data Settings

Any data specific argument is specified files stored in this folde

```
{
    "color_mode": "rgb", # rgb or bgr for vertex colors
    "include_file": true, # include the transformed data file (mesh) together with the outputted renderings
    "render_folder_together": false, # whether you have multiple objects inside subfolders of the samples folder that needs to be rendered together or not 
    "data_types": ["ply", "npz"], # only render the specified the data files (obj, ply or npz) inside the samples folder, npz expects certain structure
    "location": [0.0, 0.0, 0.0],  # determines translation applied to the inputted mesh
    "rotation_euler": [90,0,180], # determines rotation applied to the inputted mesh
    "scale": [1, 1, 1], # determines rotation applied to the inputted mesh
    "include_objs": false, # whether to include the individual objects in the outputted directory 
    "include_metadata": true, # include any kind of extra metadata that leads to the outputted rendering files
    "forward_facing": false # apply an extra rotation to make the mesh face forward. Possible arguments: [x,y,z] or false
}
```
