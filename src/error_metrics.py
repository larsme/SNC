########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MAE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, d, gt_d, **args):        
        val_pixels = (gt_d > 0).float()
        err = torch.abs(gt_d * val_pixels - d * val_pixels)
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.mean(loss / cnt)

class RMSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, d, gt_d, **args):        
        val_pixels = (gt_d > 0).float()
        err = (gt_d * val_pixels - d * val_pixels) ** 2
        loss = torch.sum(err.view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.mean(torch.sqrt(loss / cnt))

class Deltas(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, d, gt_d, **args):        
        val_pixels = (gt_d > 0).float()
        rel = torch.max((gt_d * val_pixels) / (d * val_pixels + 1e-3), (d * val_pixels) / (gt_d * val_pixels))

        cnt = torch.sum(val_pixels.view(val_pixels.size(0), 1, -1), -1, keepdim=True)

        def del_i(i):
            r = (rel < 1.01 ** i).float()
            delta = torch.sum(r.view(r.size(0), 1, -1), -1, keepdim=True) / cnt
            return torch.mean(delta)

        return del_i(1), del_i(2), del_i(3)

class wMAE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, d, cd, gt_d, **args):        
        val_pixels = (gt_d > 0).float()
        err = torch.abs(gt_d * val_pixels - d * val_pixels)
        loss = torch.sum((err * cd).view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum((val_pixels * cd).view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.mean(loss / cnt)

class wRMSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, d, cd, gt_d, **args):        
        val_pixels = (gt_d > 0).float()
        err = (gt_d * val_pixels - d * val_pixels) ** 2
        loss = torch.sum((err * cd).view(err.size(0), 1, -1), -1, keepdim=True)
        cnt = torch.sum((val_pixels * cd).view(val_pixels.size(0), 1, -1), -1, keepdim=True)
        return torch.mean(torch.sqrt(loss / cnt))

class wDeltas(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, d, cd, gt_d, **args):        
        val_pixels = (gt_d > 0).float()
        rel = torch.max((gt_d * val_pixels) / (d * val_pixels + 1e-3), (d * val_pixels) / (gt_d * val_pixels))
        cnt = torch.sum((val_pixels * cd).view(val_pixels.size(0), 1, -1), -1, keepdim=True)

        def del_i(i):
            r = (rel < 1.01 ** i).float()
            delta = torch.sum((r * cd).view(r.size(0), 1, -1), -1, keepdim=True) / cnt
            return torch.mean(delta)

        return del_i(1), del_i(2), del_i(3)

class Huber(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, d, gt_d, delta=5, **args):

        l1_loss = F.l1_loss(d, gt_d, reduce=False)
        mse_loss = F.mse_loss(d, gt_d, reduce=False)

        mask = (l1_loss < delta).float()
        loss = (0.5 * mse_loss) * mask + delta * (l1_loss - 0.5 * delta) * (1 - mask)

        return torch.mean(loss)