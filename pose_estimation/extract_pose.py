import os
import shutil
import sys
from glob import glob

def run_command(command):
    print(command)
    os.system(command)

gpu_id = sys.argv[1]
custom_folder = sys.argv[2]
temp_folder = sys.argv[3]
output_folder = sys.argv[4]

# name_list = [x for ext in ['png', 'jpg'] for x in sorted(glob(f"{custom_folder}/*.{ext}")) if 'mask' not in x]
name_list = [x for x in sorted(glob(f"{custom_folder}/*.jpg")) if 'mask' not in x]

run_command('CUDA_VISIBLE_DEVICES=%s python DataProcess/Gen_HeadMask.py --img_dir ./%s/' % (gpu_id, custom_folder))
run_command('CUDA_VISIBLE_DEVICES=%s python DataProcess/Gen_Landmark.py --img_dir ./%s/' % (gpu_id, custom_folder))

for name_all in name_list:
    name = os.path.basename(name_all)[:-4]
    run_command('python align_roll.py %s %s %s' % (name, custom_folder, temp_folder))

run_command('CUDA_VISIBLE_DEVICES=%s python process_test_images.py --input_dir ./%s/ --gpu=%s' % (gpu_id, temp_folder, gpu_id))
run_command('python check_pose.py %s %s ' % (temp_folder, output_folder))
