## MRI Image Super Resolution through Filter Learning Based on Surrounding Gradient Information in 3D Space

 - Implemation of 3D MRI Super-resolution with machine learning that minimizing MSE.

 - Papers will be updated.

  

## Requirments

 - Both Linux and Windows are supported.

 - You need NVIDIA GPU at least 11GB DRAM for fast computing. (We used RTX 2080 Classic Ti)

 - To use GPU, You should install CUDA.

 - You need at least 10GB RAM to allocate

 - Package Required：numpy, cupy, numba, skimage, nibabel



## Prepare datasets

 - We got MRI Brain T1 Dataset from 'Human Connectome Project' (http://www.humanconnectomeproject.org/data/)

 - We used 100 images to train, 50 images to estimate metrics.

 - Store your HR train data to 'train' folder, HR test data to 'test' folder. In test stage, data will be downscaled to estimate HR image.



## Folder Structure Example
```
.
├── train
|   ├── T1w_brain_File1.nii.gz
|   └── T1w_brain_File2.nii.gz
├── test
|   ├── T1w_brain_File3.nii.gz
|   └── T1w_brain_File4.nii.gz
├── result
|   └── 110521
|       ├── T1w_brain_File3.nii.gz
|       ├── T1w_brain_File4.nii.gz
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
cd Workspace
python train.py
```
 - Run following commmand to get upscaled images.

```
python test.py
```

 ## Code References
  - https://github.com/JalaliLabUCLA/Jalali-Lab-Implementation-of-RAISR
  - https://github.com/movehand/raisr