# SAM-Mamba：A Two-Stage Change Detection Network Combining the Adapting Segment Anything and Mamba models
The proposed SAM-Mamba architecture.
<img width="1129" height="639" alt="mamba" src="https://github.com/user-attachments/assets/6f76f7f3-697f-423d-8961-1e5f948e10c0" />

# Instructions for Use
## Environment
- First, install the mamba environment, refer to vision mamba.
## Dataset
- Download the corresponding dataset.
## Running
- For training, modify the correct path:
'''
    python train.py
'''
- For testing:
'''
    python test.py
'''
## Test weight paths for the four datasets based on Hiera-B.
链接: https://pan.baidu.com/s/1IdnAKGMn5Jkc6ARSJzbmeA?pwd=kbq5 提取码: kbq5

## Download path for the VSSM pre-trained model. Place this file in the source directory to enable code translation.
链接: https://pan.baidu.com/s/1sQJFHiVcmyA0uMikN5_zlw?pwd=26pk 提取码: 26pk

## sam2_hiera_large.pt：
https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

## sam2_hiera_base_plus.pt：
https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
