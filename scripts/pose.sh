# cd pose_estimation
python extract_pose.py \
    1 \
    ../input_imgs \
    ../test_runs/manip_3D_recon/1_align_result \
    ../test_runs/manip_3D_recon/2_pose_result

CUDA_VISIBLE_DEVICES=0 python DataProcess/Gen_HeadMask.py \
    --img_dir ./../input_imgs/

CUDA_VISIBLE_DEVICES=0 python DataProcess/Gen_Landmark.py \
    --img_dir ./../input_imgs/

CUDA_VISIBLE_DEVICES=0 python process_test_images.py \
    --input_dir ./../test_runs/manip_3D_recon/1_align_result/ \
    --gpu=0

python batch_mtcnn.py --in_root ./../test_runs/manip_3D_recon/1_align_result/
python test.py --img_folder=./../test_runs/manip_3D_recon/1_align_result/ --gpu_ids=0 --name=pretrained --epoch=20
python crop_images.py --indir ./../test_runs/manip_3D_recon/1_align_result/ --outdir ./../test_runs/manip_3D_recon/1_align_result/cropped_images
python 3dface2idr.py --in_root ./../test_runs/manip_3D_recon/1_align_result/epoch_20_000000 --out_root ./../test_runs/manip_3D_recon/1_align_result/cropped_images

python check_pose.py \
    ../test_runs/manip_3D_recon/1_align_result \
    ../test_runs/manip_3D_recon/2_pose_result

# cd ../eg3d
python run_inversion.py \
    --outdir=../test_runs/manip_3D_recon/3_inversion_result \
    --latent_space_type=w_plus \
    --network=pretrained/ffhqrebalanced512-128.pkl \
    --image_path=../test_runs/manip_3D_recon/2_pose_result \
    --num_steps=300

python gen_samples.py \
    --network ../finetuned/ffhq-pixar.pkl \
    --w_pth=../test_runs/manip_3D_recon/3_inversion_result/input_inv.pt \
    --seeds='100-200' \
    --generator_type=ffhq \
    --outdir='../test_runs/manip_3D_recon/4_manip_result' \
    --shapes=True \
    --shape_format=.mrc \
    --shape_only_first=False \
    --trunc=0.7

python gen_videos.py \
    --network ../finetuned/ffhq-pixar.pkl \
    --w_pth=../test_runs/manip_3D_recon/3_inversion_result/input_inv.pt \
    --seeds='100-200' \
    --generator_type=ffhq \
    --outdir=../test_runs/manip_3D_recon/4_manip_result \
    --shapes=False \
    --trunc=0.7 \
    --grid=1x1 \
    --w-frames=120
