CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_MYMIL_2.py --wandb_mode offline --arch ga \
                                                                --n_token 8 --n_drop 2 --temp 0.5 \
                                                                --pretrain medical_ssl \
                                                                --config config/bracs_config.yml


CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_MYMIL_2.py --wandb_mode offline --arch ga \
                                                                --n_token 8 --n_drop 2 --temp 0.5 \
                                                                --pretrain medical_ssl \
                                                                --config config/bracs_config.yml >> Sigmax_temp_test1.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_MYMIL_2.py --wandb_mode offline --arch ga \
                                                                --n_token 8 --n_drop 4 --temp 0.5 \
                                                                --pretrain medical_ssl \
                                                                --config config/bracs_config.yml >> Sigmax_temp_test1.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_MYMIL_2.py --wandb_mode offline --arch ga \
                                                                --n_token 8 --n_drop 0 --temp 0.5 \
                                                                --pretrain medical_ssl \
                                                                --config config/bracs_config.yml >> Sigmax_temp_test1.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_MYMIL_2.py --wandb_mode offline --arch ga \
                                                                --n_token 10 --n_drop 5 --temp 0.5 \
                                                                --pretrain medical_ssl \
                                                                --config config/bracs_config.yml >> Sigmax_temp_test1.txt


CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_MYMIL_2.py --wandb_mode offline --arch ga \
                                                                --n_token 10 --n_drop 0 --temp 0.5 \
                                                                --pretrain medical_ssl \
                                                                --config config/bracs_config.yml >> Sigmax_temp_wo_tokensup_3.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_MYMIL_2.py --wandb_mode offline --arch ga \
                                                                --n_token 10 --n_drop 4 --temp 0.5 \
                                                                --pretrain medical_ssl \
                                                                --config config/bracs_config.yml >> Sigmax_temp_wo_tokensup_3.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_MYMIL_2.py --wandb_mode offline --arch ga \
                                                                --n_token 10 --n_drop 2 --temp 0.5 \
                                                                --pretrain medical_ssl \
                                                                --config config/bracs_config.yml >> Sigmax_temp_wo_tokensup_3.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_MYMIL_2.py --wandb_mode offline --arch ga \
                                                                --n_token 10 --n_drop 6 --temp 0.5 \
                                                                --pretrain medical_ssl \
                                                                --config config/bracs_config.yml >> Sigmax_temp_wo_tokensup_3.txt