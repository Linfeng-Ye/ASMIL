CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification.py --seed 1 --wandb_mode offline \
                       --arch clam_sb \
                       --pretrain natural_supervsied --config config/patch_classification_camelyon17_config.yml >> clam_sb.txt
                       
CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification.py --seed 1 --wandb_mode offline \
                       --arch transmil \
                       --pretrain natural_supervsied --config config/patch_classification_camelyon17_config.yml >> transmil.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_DTFD.py --seed 1 --wandb_mode offline \
                       --pretrain natural_supervised --config config/patch_classification_camelyon17_config.yml >> DTFD.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_IBMIL.py --seed 1 --wandb_mode offline \
                       --pretrain natural_supervised --config config/patch_classification_camelyon17_config.yml >> IBMIL.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_MHIM.py --seed 1 --wandb_mode offline \
                       --model mhim --pretrain natural_supervised --config config/patch_classification_camelyon17_config.yml  >> MHIM.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_ACMIL.py --seed 1 --wandb_mode offline \
                       --pretrain natural_supervised --config config/patch_classification_camelyon17_config.yml  >> ACMIL.txt
CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification.py --seed 2 --wandb_mode offline \
                       --arch clam_sb \
                       --pretrain natural_supervsied --config config/patch_classification_camelyon17_config.yml >> clam_sb.txt
                       
CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification.py --seed 2 --wandb_mode offline \
                       --arch transmil \
                       --pretrain natural_supervsied --config config/patch_classification_camelyon17_config.yml >> transmil.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_DTFD.py --seed 2 --wandb_mode offline \
                       --pretrain natural_supervised --config config/patch_classification_camelyon17_config.yml >> DTFD.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_IBMIL.py --seed 2 --wandb_mode offline \
                       --pretrain natural_supervised --config config/patch_classification_camelyon17_config.yml >> IBMIL.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_MHIM.py --seed 2 --wandb_mode offline \
                       --model mhim --pretrain natural_supervised --config config/patch_classification_camelyon17_config.yml  >> MHIM.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_ACMIL.py --seed 2 --wandb_mode offline \
                       --pretrain natural_supervised --config config/patch_classification_camelyon17_config.yml  >> ACMIL.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification.py --seed 2 --wandb_mode offline \
                       --arch clam_sb \
                       --pretrain natural_supervsied --config config/patch_classification_camelyon17_config.yml >> clam_sb.txt
                       
CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification.py --seed 3 --wandb_mode offline \
                       --arch transmil \
                       --pretrain natural_supervsied --config config/patch_classification_camelyon17_config.yml >> transmil.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_DTFD.py --seed 3 --wandb_mode offline \
                       --pretrain natural_supervised --config config/patch_classification_camelyon17_config.yml >> DTFD.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_IBMIL.py --seed 3 --wandb_mode offline \
                       --pretrain natural_supervised --config config/patch_classification_camelyon17_config.yml >> IBMIL.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_MHIM.py --seed 3 --wandb_mode offline \
                       --model mhim --pretrain natural_supervised --config config/patch_classification_camelyon17_config.yml  >> MHIM.txt

CUDA_VISIBLE_DEVICES=0 python Step3_WSI_classification_ACMIL.py --seed 3 --wandb_mode offline \
                       --pretrain natural_supervised --config config/patch_classification_camelyon17_config.yml  >> ACMIL.txt







