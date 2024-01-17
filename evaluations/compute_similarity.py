from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

import matplotlib.pyplot as plt
import lpips
import numpy as np
import torch
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--label_root", help="path to reference image folder")
parser.add_argument("--task", help="inverse task type")
args = parser.parse_args()
label_root = Path(args.label_root)
task = args.task

device = 'cuda:0'
loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)

delta_recon_root = Path(os.environ['DATA_DIR'])
# normal_recon_root = Path(f'./results/{task}/ffhq/{factor}/{sigma}/ps+/{scale}/recon')

psnr_delta_list = []
psnr_normal_list = []

ssim_list = []

lpips_delta_list = []
lpips_normal_list = []
for idx in tqdm(range(300)):
    if task == 'ct_recon' and idx == 197:
        continue
    fname = str(idx).zfill(7)
    if task == 'ct_recon':
        label = plt.imread(label_root / f'test_{idx:03}.png')[:, :, None].repeat(3, axis=-1)
    else:
        label = plt.imread(label_root / f'bedroom_{fname}.png')[:, :, :3]
    delta_recon = plt.imread(delta_recon_root / f'bedroom_{fname}.png')[:, :, :3]
    # normal_recon = plt.imread(normal_recon_root / f'{fname}.png')[:, :, :3]

    psnr_delta = peak_signal_noise_ratio(label, delta_recon)
    # psnr_normal = peak_signal_noise_ratio(label, normal_recon)

    psnr_delta_list.append(psnr_delta)
    # psnr_normal_list.append(psnr_normal)

    ssim_delta = ssim(label, delta_recon, channel_axis=2, data_range=label.max()-label.min())
    ssim_list.append(ssim_delta)

    delta_recon = torch.from_numpy(delta_recon).permute(2, 0, 1).to(device)
    # normal_recon = torch.from_numpy(normal_recon).permute(2, 0, 1).to(device)
    label = torch.from_numpy(label).permute(2, 0, 1).to(device)

    delta_recon = delta_recon.view(1, 3, 256, 256) * 2. - 1.
    # normal_recon = normal_recon.view(1, 3, 256, 256) * 2. - 1.
    label = label.view(1, 3, 256, 256) * 2. - 1.

    delta_d = loss_fn_vgg(delta_recon, label)
    # normal_d = loss_fn_vgg(normal_recon, label)

    lpips_delta_list.append(delta_d)
    # lpips_normal_list.append(normal_d)

psnr_delta_avg = sum(psnr_delta_list) / len(psnr_delta_list)
ssim_avg = sum(ssim_list) / len(ssim_list)
lpips_delta_avg = sum(lpips_delta_list) / len(lpips_delta_list)

# psnr_normal_avg = sum(psnr_normal_list) / len(psnr_normal_list)
# lpips_normal_avg = sum(lpips_normal_list) / len(lpips_normal_list)

print(f'Delta PSNR: {psnr_delta_avg}')
print(f'Delta SSIM: {ssim_avg}')
print(f'Delta LPIPS: {lpips_delta_avg}')

