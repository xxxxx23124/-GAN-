'''
这个没用
'''

import torch
import numpy as np


def mixup_data(true, fake, alpha, device):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = true.size(0)
    label = torch.full((batch_size, 1), lam, dtype=torch.float32, device=device)
    mixed = lam * true + (1 - lam) * fake
    return mixed, label, lam

def mixup_data_wgan(true, fake, alpha, device):
    batch_size = true.size(0)
    if alpha > 0:
        label = np.random.beta(alpha, alpha, size=(batch_size, 1))
    else:
        label = np.ones(shape=(batch_size, 1))
    label = torch.from_numpy(label).to(device)
    mixed = label * true + (1 - label) * fake
    return mixed, label