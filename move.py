import os
import shutil

# 文件a和文件b的根路径
path_a_root = "/home/qilei/Experiments/meta_univ/test_runs/mini_test/"
path_b_root = "/home/qilei/Experiments/meta_univ/test_runs/2/"

# 遍历所有可能的“xxxx”子目录
for root, dirs, files in os.walk(path_b_root):
    for dir in dirs:
        if dir == "manip_3D_recon":
            # 构建完整的文件a和文件b路径
            path_a = os.path.join(path_a_root, root[len(path_b_root):], dir, "4_manip_result")
            path_b = os.path.join(root, dir, "4_manip_result")

            # 遍历文件b目录中的所有.jpg文件
            for filename in os.listdir(path_b):
                if filename.endswith(".jpg"):
                    # 构建完整的文件b和文件a中的文件路径
                    file_b = os.path.join(path_b, filename)
                    file_a = os.path.join(path_a, filename)

                    try:
                        # 移动文件b到文件a
                        shutil.move(file_b, file_a)
                        print(f"已移动文件: {file_b} 到 {file_a}")
                    except Exception as e:
                        print(f"移动文件时发生错误: {e}")

print("文件移动完成。")