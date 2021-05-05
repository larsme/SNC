import torch
import torch.nn.functional as F
from einops import rearrange, reduce
import matplotlib.pyplot as plt
import numpy as np

class NC_pool(torch.nn.Module):
    def __init__(self, init_method='k', n_c=1, kernel_size=2, symmetric=True, w_pooling=False, cd_pooling=False, **kwargs):
        super().__init__()

        self.eps = 1e-20
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
        self.symmentric = symmetric
        self.w_pooling = w_pooling
        self.cd_pooling = cd_pooling
        self.n_c = n_c

        # Define Parameters
        self.w_channel = torch.nn.Parameter(data=torch.ones(n_c, n_c), requires_grad=n_c > 1)
        self.w_spatial = torch.nn.Parameter(data=torch.ones(n_c, kernel_size, kernel_size // 2 if symmetric else kernel_size))
        self.w_pool = torch.nn.Parameter(data=torch.ones(1, n_c, 1, 1), requires_grad=w_pooling)

        # Init Parameters
        if 'x' in init_method:  # Xavier
            torch.nn.init.xavier_uniform_(self.w_channel)
            torch.nn.init.xavier_uniform_(self.w_spatial)
            torch.nn.init.xavier_uniform_(self.w_pool)
        elif 'k' in init_method: # Kaiming
            torch.nn.init.kaiming_uniform_(self.w_channel)
            torch.nn.init.kaiming_uniform_(self.w_spatial)
            torch.nn.init.kaiming_uniform_(self.w_pool)

    def prepare_weights(self):
        # enforce limits
        w_channel = F.softplus(self.w_channel)
        w_spatial = F.softplus(self.w_spatial)
        w_pool = F.softplus(self.w_pool)

        # enforce symmetry by weight sharing
        if self.symmentric:
            w_spatial = torch.cat((w_spatial,w_spatial[:,:,:self.kernel_size // 2].flip(dims=(2,))), dim=2)

        # normalize by output channel
        w_channel = w_channel / reduce(w_channel,'o i -> o 1','sum')
        w_spatial = w_spatial / reduce(w_spatial,'i h w -> i 1 1','sum')

        # combine seperable convolution for speed
        w_conv = torch.einsum('o i, i h w -> o i h w', w_channel, w_spatial)

        return w_conv, w_pool

    def prep_eval(self):
        self.weights = self.prepare_weights()

    def forward(self, x):
        # x = stacked dcd, cd
        # dcd = depth * cd
        # cd = confidence over depth

        if self.training:
            w_conv, w_pool = self.prepare_weights()
        else:
            w_conv, w_pool = self.weights

        dcd, cd = x.split(x.shape[0]//2, 0)
        if self.cd_pooling:
            w_pool = (cd * cd / (dcd + self.eps)).detach()
        elif self.w_pooling:
            w_pool = (cd / (dcd + self.eps)).detach() + w_pool
        else:
            w_pool = (cd / (dcd + self.eps)).detach()
        denom = F.conv2d(w_pool, w_conv, padding=self.padding, stride=2) + self.eps
        return F.conv2d(x * w_pool.repeat(2, 1, 1, 1), w_conv, padding=self.padding, stride=2) / denom.repeat(2, 1, 1, 1)

    def visualize_weights(self, rows, cols, col):
        w_conv_d = self.prepare_weights()

        w_spatial_d = reduce(w_conv_d, 'o i h w -> i h w', 'sum').detach()
        w_channel_d = reduce(w_conv_d, 'o i h w -> o i', 'sum').detach()

        idx = col

        if self.kernel_size > 1:
            for c in range(self.n_c):
                ax = plt.subplot(rows, cols, idx)
                plt.imshow(w_spatial_d[c, :, :].cpu())
                plt.xlabel('w', fontsize=min(10,100 / rows))
                plt.ylabel('h', fontsize=min(10,100 / rows))
                plt.xticks([])
                plt.yticks([])
                idx+=cols

        if self.n_c > 1:
            ax = plt.subplot(rows, cols, idx)
            plt.xlabel('in', fontsize=min(10,100 / rows))
            plt.ylabel('out', fontsize=min(10,100 / rows))
            plt.imshow(w_channel_d.cpu())
            plt.xticks([])
            plt.yticks([])
            idx+=cols