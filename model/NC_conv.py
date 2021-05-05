import torch
import torch.nn.functional as F
from einops import rearrange, reduce
import matplotlib.pyplot as plt
import numpy as np

class NC_conv(torch.nn.Module):
    def __init__(self, init_method='k', n_in=1, n_out=1, kernel_size=1, stride=1, symmetric=True, **kwargs):
        super(NC_conv, self).__init__()

        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
        self.symmentric = symmetric
        self.n_in = n_in
        self.n_out = n_out

        # Define Parameters
        self.w_channel = torch.nn.Parameter(data=torch.ones(n_out, n_in), requires_grad=n_in > 1)
        self.w_spatial = torch.nn.Parameter(data=torch.ones(n_in, self.kernel_size, (self.kernel_size + 1) // 2 if symmetric else kernel_size), requires_grad=self.kernel_size > 1)

        # Init Parameters
        if 'x' in init_method:  # Xavier
            torch.nn.init.xavier_uniform_(self.w_channel)
            torch.nn.init.xavier_uniform_(self.w_spatial)
        elif 'k' in init_method: # Kaiming
            torch.nn.init.kaiming_uniform_(self.w_channel)
            torch.nn.init.kaiming_uniform_(self.w_spatial)

    def prepare_weights(self):
        # enforce limits
        w_channel = F.softplus(self.w_channel)
        w_spatial = F.softplus(self.w_spatial)

        # enforce symmetry by weight sharing
        if self.symmentric:
            w_spatial = torch.cat((w_spatial,w_spatial[:,:,:self.kernel_size // 2].flip(dims=(2,))), dim=2)

        # normalize by output channel
        w_channel = w_channel / reduce(w_channel,'o i -> o 1','sum')
        w_spatial = w_spatial / reduce(w_spatial,'i h w -> i 1 1','sum')

        # combine seperable convolution for speed
        w_conv = torch.einsum('o i, i h w -> o i h w', w_channel, w_spatial)

        return w_conv,

    def prep_eval(self):
        self.weights = self.prepare_weights()

    def forward(self, x):
        # x = stacked dcd, cd
        # dcd = depth * cd
        # cd = confidence over depth

        if self.training:
            w_conv, = self.prepare_weights()
        else:
            w_conv, = self.weights

        # NConv without denominator
        return F.conv2d(x, w_conv, padding=self.padding, stride=self.stride)

    def visualize_weights(self, rows, cols, col):
        w_conv_d, = self.prepare_weights()

        w_spatial_d = reduce(w_conv_d, 'o i h w -> i h w', 'sum').detach()
        w_channel_d = reduce(w_conv_d, 'o i h w -> o i', 'sum').detach()

        idx = col

        if self.kernel_size > 1:
            for c in range(self.n_in):
                ax = plt.subplot(rows, cols, idx)
                plt.imshow(w_spatial_d[c, :, :].cpu())
                plt.xlabel('w', fontsize=min(10,100 / rows))
                plt.ylabel('h', fontsize=min(10,100 / rows))
                plt.xticks([])
                plt.yticks([])
                idx+=cols
            idx+=max(0,self.n_out - self.n_in) * cols
        else:
            idx+= max(self.n_in, self.n_out) * cols

        if self.n_in > 1:
            ax = plt.subplot(rows, cols, idx)
            plt.xlabel('in', fontsize=min(10,100 / rows))
            plt.ylabel('out', fontsize=min(10,100 / rows))
            plt.imshow(w_channel_d.cpu())
            plt.xticks([])
            plt.yticks([])
            idx+=cols
        elif self.n_out > 1:
            idx+=cols