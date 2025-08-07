
CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_DSMIL.py --seed 1 --wandb_mode offline \
                       --arch dsmil \
                       --pretrain natural_supervsied --config config/patch_classification_camelyon17_config_smalllr.yml >> dsmil.txt


CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_DSMIL.py --seed 2 --wandb_mode offline \
                       --arch dsmil \
                       --pretrain natural_supervsied --config config/patch_classification_camelyon17_config_smalllr.yml >> dsmil.txt


CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_DSMIL.py --seed 3 --wandb_mode offline \
                       --arch dsmil \
                       --pretrain natural_supervsied --config config/patch_classification_camelyon17_config_smalllr.yml >> dsmil.txt



CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_DSMIL.py --seed 1 --wandb_mode offline \
                       --arch dsmil \
                       --pretrain natural_supervsied --config config/patch_classification_camelyon17_config.yml >> dsmil_biglr.txt


CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_DSMIL.py --seed 2 --wandb_mode offline \
                       --arch dsmil \
                       --pretrain natural_supervsied --config config/patch_classification_camelyon17_config.yml >> dsmil_biglr.txt


CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_DSMIL.py --seed 3 --wandb_mode offline \
                       --arch dsmil \
                       --pretrain natural_supervsied --config config/patch_classification_camelyon17_config.yml >> dsmil_biglr.txt
                       