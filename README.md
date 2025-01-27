## DeePSD: Automatic deep skinning and pose space deformation for 3D garment animation

<img src="https://github.com/DualBones25Sig/DualBones/fig/teaser.jpeg">

Paper(coming soon)| 

## Abstract
>We present a dual bone-based approach for realistic garment simulation across diverse poses and clothing styles. Our approach introduces an automatic bone generation technique, a geometrically consistent method that generates virtual bones alongside body joints, thereby improving garment deformation for both loose-fitting and tight-fitting apparel. By leveraging these generated dual bones, we enhanced the Linear Blend Skinning (LBS) module and incorporated a Post-Bone Translation (PBT) stage to minimize training errors. We utilize efficient supervised networks to compute both coarse and fine deformations: coarse deformations refine the movement of loose-fitting regions, while fine deformations apply vertex-level corrections for improved accuracy. This approach can improve prediction speed significantly while maintaining high accuracy. Experimental results demonstrate that our approach achieves low RMSE across multiple test.

<a href="blank.com">Anonymous for evaluation</a>

## Data
The dataset used on this work is <a href="https://drive.google.com/file/d/1U-VnR4warO3qlD5Fcp81Thd50XLbEwp0/view?usp=sharing">HANFU</a>.

### Preprocessing
Dual bone will be processed and cached in the specified folder in config before the model starts training.

We use open3d to visualize the preprocessing results. You can turn it off in ./data/utils/clothMesh.py


## Train
Just run 'train.py' . You can specify the config file via "--config"

## predict
Just run 'predict.py' . You can specify the config file via "--config" and predict folder via "--folder"

## Dataset Structure

**data**

| ***motion_folder***   Describe one motion

| |  **transform.npz**   *Describes the body information of this motion*

| |  ***garment_name*.npz**   *Describes the garment vertices of this motion*

| **body_info.npz**    Describes the body information in bind pose

| ***garment_name*_info.npz**    Describes the cloth information in bind pose

## Citation
```
Anonymous review stage
```
