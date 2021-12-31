import torch
import torch.nn.functional as F
from einops import rearrange, reduce
import matplotlib.pyplot as plt
import numpy as np

class NC_unpool(torch.nn.Module):
    def __init__(self, init_method='k', n_c=1, kernel_size=1, stride=1, symmetric=True, **kwargs):
        super(NC_unpool, self).__init__()

        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = (self.kernel_size - 1) // 2
        self.symmentric = symmetric
        self.n_c = n_c

        # Define Parameters
        self.w_channel = torch.nn.Parameter(data=torch.ones(n_c, n_c), requires_grad=n_c > 1)
        self.w_spatial = torch.nn.Parameter(data=torch.ones(n_c, kernel_size, (kernel_size + 1) // 2 if symmetric else kernel_size))
        self.w_skip = torch.nn.Parameter(data=torch.ones(1,n_c,1,1))

        # Init Parameters
        if 'x' in init_method:  # Xavier
            torch.nn.init.xavier_uniform_(self.w_channel)
            torch.nn.init.xavier_uniform_(self.w_spatial)
            torch.nn.init.xavier_uniform_(self.w_skip)
        elif 'k' in init_method: # Kaiming
            torch.nn.init.kaiming_uniform_(self.w_channel)
            torch.nn.init.kaiming_uniform_(self.w_spatial)
            torch.nn.init.kaiming_uniform_(self.w_skip)

    def prepare_weights(self):
        # enforce limits
        w_channel = F.softplus(self.w_channel)
        w_spatial = F.softplus(self.w_spatial)
        w_skip = torch.sigmoid(self.w_skip)

        # enforce symmetry by weight sharing
        if self.symmentric:
            w_spatial = torch.cat((w_spatial,w_spatial[:,:,:self.kernel_size // 2].flip(dims=(2,))), dim=2)

        # normalize by output channel
        w_channel = w_channel / reduce(w_channel,'o i -> o 1','sum')
        w_spatial = w_spatial / reduce(w_spatial,'i h w -> i 1 1','sum')

        # combine seperable convolution for speed
        w_conv = torch.einsum('o i, i h w -> o i h w', w_channel, w_spatial)

        return w_conv, w_skip

    def prep_eval(self):
        self.weights = self.prepare_weights()

    def forward(self, x_pool, x_skip):
        # x = stacked dcd, cd
        # dcd = depth * cd
        # cd = confidence over depth
        # _skip => skip connection with target resolution

        if self.training:
            w_conv, w_skip = self.prepare_weights()
        else:
            w_conv, w_skip = self.weights

        # N(de)convs without denominator
        return w_skip * x_skip + (1 - w_skip) * F.conv_transpose2d(x_pool, w_conv, padding=self.padding, stride=self.stride)

    def visualize_weights(self, rows, cols, col):
        w_conv_d, w_skip_d = self.prepare_weights()

        w_spatial_d = reduce(w_conv_d, 'o i h w -> i h w', 'sum')
        w_channel_d = reduce(w_conv_d, 'o i h w -> o i', 'sum')

        idx = col
        if self.kernel_size > 1:
            for c in range(self.n_c):
                ax = plt.subplot(rows, cols, idx)
                plt.imshow(w_spatial_d[c, :, :])
                plt.xlabel('w', fontsize=min(10,100 / rows))
                plt.ylabel('h', fontsize=min(10,100 / rows))
                plt.xticks([])
                plt.yticks([])
                idx+=cols

        if self.n_c > 1:
            ax = plt.subplot(rows, cols, idx)
            plt.xlabel('in', fontsize=min(10,100 / rows))
            plt.ylabel('out', fontsize=min(10,100 / rows))
            plt.imshow(w_channel_d[:, :])
            plt.xticks([])
            plt.yticks([])
            idx+=cols

        ax = plt.subplot(rows, cols, idx)
        plt.imshow(w_skip_d[:,:,0,0])
        for (j,i),label in np.ndenumerate(w_skip_d[:,:, 0, 0]):
            ax.text(i,j,'{:.02f}'.format(label),ha='center',va='center', fontsize=min(10,80 / rows), color='tomato')
        plt.xticks([])
        plt.yticks([])
        if self.n_c > 1:
            plt.xlabel('c', fontsize=min(10,100 / rows))
        idx+=cols