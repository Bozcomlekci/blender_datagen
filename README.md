<h1 align="center", style="font-family: 'Comic Sans MS', cursive; color: #FF4500;">👁️ BLENDER DATA GENERATION 👁️</h1>

![](assets/teaser.gif)

This repository allows you to automatically render your 3D objects in Blender without explicitly accessing the Blender interface. It enables you to render your objects of different types in different camera configurations.

The outputs can be used in Novel View Synthesis tasks that use Blender type of cameras. You can prepare data for your NeRF and Gaussian Splatting models with this repository.

<h2 align="center", style="font-family: 'Comic Sans MS', cursive; color: #FF4500;">🔧 SETUP 🔨</h1>

Make sure that Blender software above version 3 is installed in your system. If it is not, either [install Blender](https://www.blender.org/) in your machine or download the Blender packages and refer to the packages in your python call.

To install the python environment via conda:

```python
conda env create -f environment.yml
conda activate blender4
```

If you get an error with BlenderProc while running `generate.py`, set `OUTSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT_BUT_IN_RUN_SCRIPT=1`

To download an example mesh file that you can render, run:

```
mkdir samples
wget --no-check-certificate -P samples/CloSe-Di https://github.com/anticdimi/CloSe/raw/main/assets/demo/demo_scan.npz
```

<h2 align="center", style="font-family: 'Comic Sans MS', cursive; color: #FF4500;">⚙️ HOW TO USE ⚙️</h1>

Use `generate.py` that uses `generator.py to `.  An example cmd code can be:

```
python generate.py --samples_folder samples --num_views 10 --cam_settings cam_settings/3dcustom.json --light_settings light_settings/settings.json --split_name train
```

An example vscode config file is also provided. Change folder `--samples_folder` with your main folder containing other obj file folders. The structure of the folder should be like:

```
<samples_folder>
├── <obj_name>   
│   ├── <obj_name>.obj  
│   ├── <material>.jpeg   
│   ├── <material>.mtl
└── ...
```

Running the generation script will yield a folder with the structure:

```
nvs_data
├── <config>_<num_views>
│  ├── <obj_name> 
│  │  ├── <split_name>
│  │  │  ├── 0000.png
│  │  │  ├── 0001.png
│  │  │  ├── ...
│  │  ├── <cam_file_name>.json   
└── ...
```

This procedure generates NVS data with invisible background. There are many ways to alter your renderings for your use case. For example, to make the background white, you can also play with the `render_background` parameter in `light_settings/settings.json`

To change the light settings, change `light_settings/settings.json` file and  provide with desired lighting. [Further instructions...](light_settings/README.md)

To change the camera settings, change `cam_settings/3dcustom.json` or `cam_settings/4dcustom.json` file and  provide with desired camera setting. [Further instructions...](cam_settings/README.md)

To impose specific settings tailored to your data, made changes to the data config files under `data_settings` folder. [Further instructions...](data_settings/README.md)

To alter how your camera intrinsics and extrinsics are saved, make . [Further instructions...](cam_file_settings/README.md)

See [scripts](scripts/README.md) directory for example scripts to run

![](assets/multi.gif)

## References

The resources in the creation of this work include:

```
@article{Denninger2023, 
    doi = {10.21105/joss.04901},
    url = {https://doi.org/10.21105/joss.04901},
    year = {2023},
    publisher = {The Open Journal}, 
    volume = {8},
    number = {82},
    pages = {4901}, 
    author = {Maximilian Denninger and Dominik Winkelbauer and Martin Sundermeyer and Wout Boerdijk and Markus Knauer and Klaus H. Strobl and Matthias Humt and Rudolph Triebel},
    title = {BlenderProc2: A Procedural Pipeline for Photorealistic Rendering}, 
    journal = {Journal of Open Source Software}
} 

@software{Liu_BlenderToolbox_2018,
  author = {Liu, Hsueh-Ti Derek},
  month = {12},
  title = {{Blender Toolbox}},
  url = {https://github.com/HTDerekLiu/BlenderToolbox},
  year = {2018}
}

@inproceedings{antic2024close,
    title = {{CloSe}: A {3D} Clothing Segmentation Dataset and Model},
    author = {Antić, Dimitrije and Tiwari, Garvita and Ozcomlekci, Batuhan  and Marin, Riccardo  and Pons-Moll, Gerard},
    booktitle = {International Conference on 3D Vision (3DV)},
    month = {March},
    year = {2024},
}


```
