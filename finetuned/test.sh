# Text-guided manipulated 3D reconstruction from images using the shifted 3D generative model

PYTHON=/home/qilei/installs/conda/envs/datid3d/bin/python

$PYTHON datid3d_test.py --mode manip \
    --indir='input_imgs/mini_test/1151' \
    --outdir='test_runs/mini_test/1151_2' \
    --generator_type='ffhq' \
    --trunc='0.7' \
    --network=finetuned/ffhq-yoda.pkl \
    --latent_space_type 'w_plus_text' \
    --regularize_vl 0.1
#w_plus_text