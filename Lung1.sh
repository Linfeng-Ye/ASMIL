CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_MYMIL_2_featrue.py --wandb_mode offline --arch ga \
                                                                --n_token 16 --n_drop 10 --temp 0.5 \
                                                                --pretrain medical_ssl --ckpt_dir MYMIL_CKPTS \
                                                                --config config/lung_medical_ssl_config.yml 
                                                                
CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_MYMIL_2_featrue.py --wandb_mode offline --arch ga \
                                                                --n_token 16 --n_drop 0 --temp 0.5 \
                                                                --pretrain medical_ssl \
                                                                --config config/lung_medical_ssl_config.yml >> lung_medical_ssl_config_mymil.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_MYMIL_2_featrue.py --wandb_mode offline --arch ga \
                                                                --n_token 16 --n_drop 4 --temp 0.5 \
                                                                --pretrain medical_ssl \
                                                                --config config/lung_medical_ssl_config.yml >> lung_medical_ssl_config_mymil.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_MYMIL_2_featrue.py --wandb_mode offline --arch ga \
                                                                --n_token 16 --n_drop 10 --temp 0.5 \
                                                                --pretrain medical_ssl \
                                                                --config config/lung_medical_ssl_config.yml >> lung_medical_ssl_config_mymil.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_MYMIL_2_featrue.py --wandb_mode offline --arch ga \
                                                                --n_token 16 --n_drop 8 --temp 0.5 \
                                                                --pretrain medical_ssl \
                                                                --config config/lung_medical_ssl_config.yml >> lung_medical_ssl_config_mymil.txt


CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_MYMIL_2_featrue.py --wandb_mode offline --arch ga \
                                                                --n_token 8 --n_drop 2 --temp 0.07 \
                                                                --pretrain medical_ssl \
                                                                --config config/lung_medical_ssl_config.yml >> lung_medical_ssl_config_mymil.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_MYMIL_2_featrue.py --wandb_mode offline --arch ga \
                                                                --n_token 8 --n_drop 4 --temp 0.07 \
                                                                --pretrain medical_ssl \
                                                                --config config/lung_medical_ssl_config.yml >> lung_medical_ssl_config_mymil.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_MYMIL_2_featrue.py --wandb_mode offline --arch ga \
                                                                --n_token 8 --n_drop 0 --temp 0.07 \
                                                                --pretrain medical_ssl \
                                                                --config config/lung_medical_ssl_config.yml >> lung_medical_ssl_config_mymil.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_MYMIL_2_featrue.py --wandb_mode offline --arch ga \
                                                                --n_token 10 --n_drop 5 --temp 0.07 \
                                                                --pretrain medical_ssl \
                                                                --config config/lung_medical_ssl_config.yml >> lung_medical_ssl_config_mymil.txt
