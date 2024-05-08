python eg3d/gen_samples.py \
    --network "finetuned/ffhq-pixar.pkl" \
    --seeds "100-200" \
    --generator_type "ffhq" \
    --outdir "test_runs/debug" \
    --shapes 1 \
    --shape_format ".mrc" \
    --shape_only_first 0 \
    --trunc "0.7"
