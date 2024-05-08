python datid3d_train.py \
    --mode='ft' \
    --pdg_prompt='a FHD photo of face of beautiful Elf with silver hair in the live action movie' \
    --pdg_generator_type='ffhq' \
    --pdg_strength=0.7 \
    --pdg_num_images=1000 \
    --pdg_sd_model_id='stabilityai/stable-diffusion-2-1-base' \
    --pdg_num_inference_steps=50 \
    --ft_generator_type='same' \
    --ft_batch=20 \
    --ft_kimg=200