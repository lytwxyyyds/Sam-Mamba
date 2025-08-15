#!/bin/bash

# 配置路径参数（根据实际路径修改）
project_path="/home/oem/disk2/project2025/RemoteCD/SamCD-main"  # 替换为项目根目录路径，例如 "/home/user/MambaCD"
dataset_path="/home/oem/disk2/project2025/RemoteCD/SamCD-main/datasets/SYSU"  # 替换为数据集根目录路径，例如 "/data/SYSU"

# 训练超参数配置
python train.py \
    --dataset 'SYSU' \
    --batch_size 8 \
    --crop_size 256 \
    --max_iters 320000 \
    --model_type 'CD' \
    --model_param_path "${project_path}/saved_models" \
    --train_dataset_path "${dataset_path}/train" \
    --train_data_list_path "${dataset_path}/train.txt" \
    --test_dataset_path "${dataset_path}/val" \
    --test_data_list_path "${dataset_path}/val.txt" \
    --cfg "${project_path}/configs/vssm.yaml" \
    --pretrained_weight_path "${project_path}/pretraind/vssm.pth"