## MRI Image Super Resolution through Filter Learning Based on Surrounding Gradient Information in 3D Space

 - Implemation of 3D MRI Super-resolution with linear regression that minimizing MSE.

 - Paper link (Korean): https://doi.org/10.9717/kmms.2020.24.2.178

> **Abstract** : Three-dimensional high-resolution magnetic resonance imaging (MRI) provides fine-level anatomical information for disease diagnosis. However, there is a limitation in obtaining high resolution due to the long scan time for wide spatial coverage. Therefore, in order to obtain a clear high-resolution(HR) image in a wide spatial coverage, a super-resolution technology that converts a low-resolution(LR) MRI image into a high-resolution is required. In this paper, we propose a super-resolution technique through filter learning based on information on the surrounding gradient information in 3D space from 3D MRI images. In the learning step, the gradient features of each voxel are computed through eigen-decomposition from 3D patch. Based on these features, we get the learned filters that minimize the difference of intensity between pairs of LR and HR images for similar features. In test step, the gradient feature of the patch is obtained for each voxel, and the filter is applied by selecting a filter corresponding to the feature closest to it. As a result of learning 100 T1 brain MRI images of HCP which is publicly opened, we showed that the performance improved by up to about 11% compared to the traditional interpolation method.



## Requirments

 - Both Linux and Windows are supported.

 - You can use CUDA at least 11GB vRAM for fast computing. (We used RTX 2080 Classic Ti)

 - You need at least 10GB RAM to allocate Q and V matrix.

 - Package Required: numpy, cupy, numba, skimage, nibabel



## Prepare datasets

 - We got young-adult T1-weighted masked MRI brain Dataset from 'Human Connectome Project' (https://www.humanconnectome.org/study/hcp-young-adult)

 - We used 100 images to train, 50 images to estimate metrics.

 - Store your HR train data to 'train' folder, HR test data to 'test' folder. In test stage, data will be downscaled to estimate HR image.



## Folder Structure Example
```
.
├── train
|   ├── T1w_acpc_dc_restore_brain_id1.nii.gz
|   └── T1w_acpc_dc_restore_brain_id2.nii.gz
├── test
|   ├── T1w_acpc_dc_restore_brain_id3.nii.gz
|   └── T1w_acpc_dc_restore_brain_id4.nii.gz
├── result
|   └── 110521
|       ├── T1w_acpc_dc_restore_brain_id3.nii.gz
|       ├── T1w_acpc_dc_restore_brain_id4.nii.gz
├── arrays
|   ├── h_2.npy
|   ├── Qfactor_str2
|   └── Qfactor_coh2
├── train.py
├── test.py
├── crop_black.py
├── filter_constant.py
├── filter_func.py
├── get_lar.py
├── hashtable.py
├── matrix_compute.py
└── util.py
```


## Running
 - Run following commmand to start training filters.

```
git clone https://github.com/Snailpong/MRI_Super_Resolution_3D_Gradient_Filter_Learning.git
cd MRI_Super_Resolution_3D_Gradient_Filter_Learning
python train.py
```
 - Run following commmand to get upscaled images.

```
python test.py
```


## Result Visualization
![Result](https://user-images.githubusercontent.com/11583179/109902732-3349f180-7cde-11eb-8762-31f2a2fd60a2.png)


## Code References
  - https://github.com/JalaliLabUCLA/Jalali-Lab-Implementation-of-RAISR
  - https://github.com/movehand/raisr



## License
GNU General Public License 3.0 License
