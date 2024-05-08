#!/bin/bash
PAT=$1
DATA=$2
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890

# 原文件夹路径
original_folder="/home/qilei/Experiments/meta_univ/finetuned"

# Check if running inside an Apptainer container
if [ -n "$APPTAINER_CONTAINER" ]; then
    PYTHON=/usr/bin/python
    echo "Running inside an Apptainer container."
else
    PYTHON=$HOME/Installs/conda/envs/datid3d/bin/python
    echo "Not running inside an Apptainer container."
fi


# 遍历原文件夹中的所有 .pkl 文件
for pkl_file in "$original_folder"/*.pkl; do
    # 提取文件名（不包括路径）
    filename=$(basename -- "$pkl_file")
    filename_without_extension="${filename%.pkl}"
    
    # 获取前三个字母
    prefix="${filename:0:3}"

    # 初始化 TYPE 和 PRETRAIN
    TYPE=""

    if [ "$prefix" == "cat" ]; then
        TYPE="cat"
    elif [ "$prefix" == "ffh" ]; then
        TYPE="ffhq"
    else
        continue
    fi
    PRETRAIN=$filename_without_extension
    # 如果TYPE不为空，运行相应的命令
    if [ -n "$TYPE" ]; then
        $PYTHON datid3d_test.py --mode manip \
            --indir input_imgs/${DATA}/${PAT} \
            --outdir test_runs/2/${PAT}_${PRETRAIN} \
            --generator_type ${TYPE} \
            --trunc='0.7' \
            --network finetuned/${filename} \
            --latent_space_type 'w_plus' \
            --regularize_vl 0.1
    fi
done
