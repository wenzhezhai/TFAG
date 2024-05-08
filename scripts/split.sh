#!/bin/bash

# 指定原文件夹路径
original="/home/qilei/Experiments/meta_univ/input_imgs/mini_test_1"

# 检查原文件夹是否存在
if [ ! -d "$original" ]; then
  echo "原文件夹不存在: $original"
  exit 1
fi

# 创建目标文件夹路径
target_dir="/home/qilei/Experiments/meta_univ/input_imgs"

# 获取原文件夹下子文件夹的数量
subfolder_count=$(find "$original" -maxdepth 1 -type d | wc -l)

# 计算每份子文件夹数量
subfolders_per_group=$((subfolder_count / 10))

# 如果子文件夹数量不足10份，可能需要做适当调整

# 分割子文件夹并移动到目标文件夹
for ((i = 1; i <= 10; i++)); do
  # 创建目标文件夹
  target_group_dir="$target_dir/all_$i"
  mkdir -p "$target_group_dir"

  # 移动子文件夹到目标文件夹
  for ((j = 1; j <= subfolders_per_group; j++)); do
    subfolder=$(find "$original" -maxdepth 1 -mindepth 1 -type d | head -n 1)
    if [ -n "$subfolder" ]; then
      mv "$subfolder" "$target_group_dir"
    fi
  done
done

echo "子文件夹已分割并移动到目标文件夹"