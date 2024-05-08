#!/bin/bash

# 指定原文件夹路径
original="/home/qilei/Experiments/meta_univ/input_imgs/test"
DATA=$(basename "$original")
# 遍历原文件夹下的子文件夹
for subfolder in "$original"/*; do
  if [ -d "$subfolder" ]; then  # 检查是否为文件夹
    # 提取子文件夹的文件名
    folder_name=$(basename "$subfolder")
    
    # 运行test_all_pretrained_models.sh脚本并传递文件名作为参数
    bash scripts/test_all_pretrained_models.sh ${folder_name} ${DATA}
  fi
done