
for i in {48..49}; do
    CUDA_VISIBLE_DEVICES=0 python Step5_save_probability_vectors.py --seed 4 --config  config/lung_medical_ssl_config.yml \
        --data_slide_dir /media/XXXXXX/WSI_dataset/TCGA_LUNG/LUAD_flatten \
        --slide_id TCGA-L9-A8F4-01Z-00-DX1.E2BBB8DE-94E2-4781-9B55-8A4CFBF8A69D --slide_ext .svs \
        --arch ga --n_masked_patch 0 --n_token 1 --mask_drop 0 --zoom_factor 1 \
        --ckpt_folder /fs2/comm/XXXXXXXXXXX/XXXXXX/Documents/ACMIL/wandb/offline-run-20250724_191652-lvmhm8gl/files/saved_models \
        --ckpt "checkpoint_${i}" 
done

# 
for i in {0..49}; do
    CUDA_VISIBLE_DEVICES=0 python Step5_save_MyMIL_probability.py  --seed 4 --config  config/lung_medical_ssl_config.yml \
        --data_slide_dir /media/XXXXXX/WSI_dataset/TCGA_LUNG/LUAD_flatten \
        --slide_id TCGA-L9-A8F4-01Z-00-DX1.E2BBB8DE-94E2-4781-9B55-8A4CFBF8A69D --slide_ext .svs \
        --arch ga --n_masked_patch 0 --n_token 1 --mask_drop 0 --zoom_factor 1 \
        --ckpt_folder /fs2/comm/XXXXXXXXXXX/XXXXXX/Documents/ACMIL/MYMIL_CKPTS --n_drop 10 --n_token 16 \
        --ckpt "checkpoint_${i}" >> ASMILloss.txt
done


# python fluctAttention.py --slide_id TCGA-L9-A8F4-01Z-00-DX1.E2BBB8DE-94E2-4781-9B55-8A4CFBF8A69D 


# python fluctAttentionMIL.py --slide_id TCGA-L9-A8F4-01Z-00-DX1.E2BBB8DE-94E2-4781-9B55-8A4CFBF8A69D

for i in {0..49}; do
    CUDA_VISIBLE_DEVICES=0 python Step5_save_probability_vectors.py --seed 4 --config  config/lung_medical_ssl_config.yml \
        --data_slide_dir /media/XXXXXX/WSI_dataset/TCGA_LUNG/LUAD_flatten \
        --slide_id TCGA-86-8281-01Z-00-DX1.02237263-003b-4de1-8a2f-0f3ad14768c7 --slide_ext .svs \
        --arch ga --n_masked_patch 0 --n_token 1 --mask_drop 0 --zoom_factor 1 \
        --ckpt_folder /fs2/comm/XXXXXXXXXXX/XXXXXX/Documents/ACMIL/wandb/offline-run-20250724_191652-lvmhm8gl/files/saved_models \
        --ckpt "checkpoint_${i}" 
done

for i in {0..49}; do
    CUDA_VISIBLE_DEVICES=0 python Step5_save_probability_vectors.py --seed 4 --config  config/lung_medical_ssl_config.yml \
        --data_slide_dir /media/XXXXXX/WSI_dataset/TCGA_LUNG/LUAD_flatten \
        --slide_id TCGA-78-7535-01Z-00-DX1.c4ca06f3-22d1-4e39-85c8-98d1fa2b0e60 --slide_ext .svs \
        --arch ga --n_masked_patch 0 --n_token 1 --mask_drop 0 --zoom_factor 1 \
        --ckpt_folder /fs2/comm/XXXXXXXXXXX/XXXXXX/Documents/ACMIL/wandb/offline-run-20250724_191652-lvmhm8gl/files/saved_models \
        --ckpt "checkpoint_${i}" 
done


for i in {0..49}; do
    CUDA_VISIBLE_DEVICES=0 python Step5_save_probability_vectors.py --seed 4 --config  config/lung_medical_ssl_config.yml \
        --data_slide_dir /media/XXXXXX/WSI_dataset/TCGA_LUNG/LUAD_flatten \
        --slide_id TCGA-69-A59K-01Z-00-DX1.01EAF520-9AC1-4ECC-8EF3-B9122924A1E3 --slide_ext .svs \
        --arch ga --n_masked_patch 0 --n_token 1 --mask_drop 0 --zoom_factor 1 \
        --ckpt_folder /fs2/comm/XXXXXXXXXXX/XXXXXX/Documents/ACMIL/wandb/offline-run-20250724_191652-lvmhm8gl/files/saved_models \
        --ckpt "checkpoint_${i}" 
done


