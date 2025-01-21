# Cam File Settings

For any Blender data, there needs to be a "transforms.json" like file that contains camera intrinsics and extrinsics of the camera. Cam file settings determines the specifications of this camera file.

```
{
    "file_name": "transforms_train", # For example transforms_train for training examples and transforms_test for test examples
    "img_extension": "png", # png or jpg
    "include_img_extension": false, # Include img extension in the paths inside the camera file while refering to the images
    "include_intrinsics_details": true # Include the details of the intrinsics in the camera file
}
```
