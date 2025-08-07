from PIL import Image
import numpy as np
import os
import matplotlib.pylab as plt

from tqdm import tqdm
# folder = 'vis/ds_lung_medical_ssl_arch_attnmil_ntoken_16_nmp_0/'
folder = 'vis/ASMIL_results/'

# folder = 'vis/ds_lung_medical_ssl_arch_attnmil_ntoken_1_nmp_0'
fignames = os.listdir(folder)

X = [3734,7200]
Y = [1470, 3600]
tarfloder = './crop_fig/MyMIL_'

for figname in tqdm(fignames):
    fig_path = os.path.join(folder, figname)
    img = Image.open(fig_path)
    arr = np.array(img)
    Crop_arr = arr[X[0]:X[1], Y[0]:Y[1],:]
    plt.clf() 
    plt.imshow(Crop_arr)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(tarfloder, figname), bbox_inches='tight', pad_inches=0)
    plt.close('all') 
    
