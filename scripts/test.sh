# Text-guided manipulated 3D reconstruction from images using the shifted 3D generative model
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890

PYTHON=/home/qilei/installs/conda/envs/datid3d/bin/python

$PYTHON datid3d_test.py --mode manip \
    --indir='input_imgs/mini_test_11/1112' \
    --outdir='test_runs/new/1112_lego' \
    --generator_type='ffhq' \
    --trunc='0.7' \
    --network='finetuned/ffhq-lego.pkl' \
    --latent_space_type 'w_plus' \
    --regularize_vl 0.1
#w_plus_text