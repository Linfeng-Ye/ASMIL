for i in {0..49}; do
    CUDA_VISIBLE_DEVICES=0 python Step5_save_probability_vectors.py --seed 4 --config  config/lung_medical_ssl_config.yml \
        --data_slide_dir /media/XXXXXX/WSI_dataset/TCGA_LUNG/LUAD_flatten \
        --slide_id TCGA-L9-A8F4-01Z-00-DX1.E2BBB8DE-94E2-4781-9B55-8A4CFBF8A69D --slide_ext .svs \
        --arch ga --n_masked_patch 0 --n_token 1 --mask_drop 0 --zoom_factor 1 \
        --ckpt_folder /fs2/comm/XXXXXXXXXXX/XXXXXX/Documents/ACMIL/wandb/offline-run-20250724_191652-lvmhm8gl/files/saved_models \
        --ckpt "checkpoint_${i}" >> AMBILloss.txt
done
