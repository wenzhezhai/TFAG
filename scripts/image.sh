# Sample images and shapes (as .mrc files) using the shifted 3D generative model

python datid3d_test.py --mode image \
    --indir input_imgs/3 \
    --name_tag "1" \
    --generator_type='ffhq' \
    --outdir='test_runs/1' \
    --seeds='100-200' \
    --trunc='0.7' \
    --shape=True \
    --network=finetuned/ffhq-pixar.pkl
