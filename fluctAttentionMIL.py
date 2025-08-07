import numpy as np
from pprint import pprint
import yaml
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import argparse
from utils.utils import save_model, Struct, set_seed, Wandb_Writer
import h5py
import time
# from architecture.transformer import ACMIL_GA
# from architecture.clam import CLAM_SB, CLAM_MB
# from architecture.transMIL import TransMIL
import torch
# from wsi_core.WholeSlideImage import WholeSlideImage
import sys
import matplotlib.pyplot as plt


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute the Jensen-Shannon divergence between two probability distributions.
    Args:
        p: Tensor of shape [..., K], where last dim sums to 1
        q: Tensor of shape [..., K], same shape as p
        eps: Small constant to prevent log(0)
    Returns:
        js: Tensor of shape [...], JS divergence between p and q
    """
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)
    m = 0.5 * (p + q)

    kl_pm = (p * (p / m).log()).sum(dim=-1)
    kl_qm = (q * (q / m).log()).sum(dim=-1)

    return 0.5 * (kl_pm + kl_qm)

def get_arguments():
    parser = argparse.ArgumentParser('Heatmap visualization', add_help=False)
    parser.add_argument('--config', dest='config', default='config/camelyon_config.yml',
                        help='settings of dataset in yaml format')
    parser.add_argument('--data_slide_dir', type=str, default='/mnt/Xsky/zyl/dataset/CAMELYON16/training')
    parser.add_argument('--slide_ext', type=str, default='.tif')
    parser.add_argument('--ckpt_folder', type=str, default='./')
    parser.add_argument(
        "--ckpt", type=str, default=''
    )
    parser.add_argument(
        "--arch", type=str, default='ga', choices=['transmil', 'clam_sb', 'clam_mb',
                                                 'ga'], help="architecture"
    )
    parser.add_argument(
        "--slide_id", type=str, default=''
    )

    # It is advisable to perform a preliminary check before executing the code.
    parser.add_argument(
        "--seed", type=int, default=2, help="set the random seed to ensure reproducibility"
    )
    parser.add_argument('--device', default=0, type=int, help="CUDA device")
    parser.add_argument(
        "--n_masked_patch", type=int, default=0, help="whether use adversarial mask"
    )
    parser.add_argument(
        "--n_token", type=int, default=1, help="number of query token"
    )
    parser.add_argument(
        "--mask_drop", type=float, default=0.2, help="number of query token"
    )
    
    parser.add_argument("--zoom_factor", type=float, default=1.0,
                        help="determine the magnitude of zoom during visualizing the heatmap, range between 0 and 1")
    
    args = parser.parse_args()
    return args

@torch.no_grad()
def main():
    args = get_arguments()
    device = 'cuda:{}'.format(args.device)
    slide_id = args.slide_id
    probability_folder = os.path.join('./MYMILattention', slide_id)
    entries = os.listdir(probability_folder)
    Nentries = len(entries)
    file = os.path.join(probability_folder, entries[0])
    probs = torch.load(file, weights_only=False)
    prob_dims = probs.shape[0]
    probs = torch.zeros(Nentries, prob_dims)
    for idx, entry in enumerate(entries):
        file = os.path.join(probability_folder, entry)
        probs[idx] = torch.tensor(torch.load(file, weights_only=False))

    results = []
    results.append(js_divergence(probs[0], torch.ones(prob_dims)/prob_dims).item())
    for i in range(probs.shape[0]-1):
        results.append(js_divergence(probs[i], probs[i+1]).item())
    plt.plot(results)
    plt.show()
    print(results)

if __name__ == '__main__':
    main()


