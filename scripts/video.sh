# Sample pose-controlled videos using the shifted 3D generative model

python datid3d_test.py --mode video \
    --indir input_imgs/1 \
    --generator_type='ffhq' \
    --outdir='test_runs/1' \
    --seeds='100-200' \
    --trunc='0.7' \
    --grid=4x4 \
    --network=finetuned/ffhq-pixar.pkl
