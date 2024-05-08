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