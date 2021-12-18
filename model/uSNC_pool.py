import torch
import torch.nn.functional as F
from einops import rearrange, reduce
import matplotlib.pyplot as plt
import numpy as np

class uSNC_pool(torch.nn.Module):
    def __init__(self, init_method='k', n_c=1, kernel_size=1, symmetric=True, occl_pool_d=True, scs_pool_d=True, **kwargs):
        super(uSNC_pool, self).__init__()

        # paddings assume an even kernel size, a stride of 2 and input shape = output shape // 2
        self.eps = 1e-20
        self.n_c = n_c
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2 - 1
        self.symmentric = symmetric
        self.occl_pool_d = occl_pool_d
        self.scs_pool_d = scs_pool_d

        # Define Parameters
        self.w_channel_d = torch.nn.Parameter(data=torch.ones(n_c, n_c), requires_grad=n_c > 1)
        self.w_spatial_d = torch.nn.Parameter(data=torch.ones(n_c, kernel_size, (kernel_size + 1) // 2 if symmetric else kernel_size))

        self.w_channel_s = torch.nn.Parameter(data=torch.ones(n_c, n_c), requires_grad=n_c > 1)
        self.w_spatial_s = torch.nn.Parameter(data=torch.ones(n_c, kernel_size, (kernel_size + 1) // 2 if symmetric else kernel_size))

        # Init Parameters
        if 'x' in init_method:  # Xavier
            torch.nn.init.xavier_uniform_(self.w_channel_d)
            torch.nn.init.xavier_uniform_(self.w_channel_s)
            torch.nn.init.xavier_uniform_(self.w_spatial_d)
            torch.nn.init.xavier_uniform_(self.w_spatial_s)
        elif 'k' in init_method: # Kaiming
            torch.nn.init.kaiming_uniform_(self.w_channel_d)
            torch.nn.init.kaiming_uniform_(self.w_channel_s)
            torch.nn.init.kaiming_uniform_(self.w_spatial_d)
            torch.nn.init.kaiming_uniform_(self.w_spatial_s)

    def prepare_weights(self):
        # enforce limits
        w_channel_d = F.softplus(self.w_channel_d)
        w_spatial_d = F.softplus(self.w_spatial_d)
        w_channel_s = F.softplus(self.w_channel_s)
        w_spatial_s = F.softplus(self.w_spatial_s)

        # enforce symmetry by weight sharing
        if self.symmentric:
            w_spatial_d = torch.cat((w_spatial_d,w_spatial_d.flip(dims=(2,))), dim=2)
            w_spatial_s = torch.cat((w_spatial_s,w_spatial_s.flip(dims=(2,))), dim=2)

        # normalize by output channel
        w_channel_d = w_channel_d / reduce(w_channel_d,'o i -> o 1','sum')
        w_spatial_d = w_spatial_d / reduce(w_spatial_d,'i h w -> i 1 1','sum')
        w_channel_s = w_channel_s / reduce(w_channel_s,'o i -> o 1','sum')
        w_spatial_s = w_spatial_s / reduce(w_spatial_s,'i h w -> i 1 1','sum')

        # combine seperable convolution for speed
        w_conv_d = torch.einsum('o i, i h w -> o i h w', w_channel_d, w_spatial_d)
        w_conv_s = torch.einsum('o i, i h w -> o i h w', w_channel_s, w_spatial_s)

        return w_conv_d, w_conv_s

    def prep_eval(self):
        self.weights = self.prepare_weights()

    def forward(self, x, scs, cs):
        # x = d*cd, cd
        # d = depth
        # cd = confidence over depth
        # scs = smoothness * cs
        # cs = confidence over smoothness

        if self.training:
            w_conv_d, w_conv_s = self.prepare_weights()
        else:
            w_conv_d, w_conv_s = self.weights

        # pool depth
        # prefer high s, high cs and high cd
        # prefer low d, assuming this only makes a difference on edges where all sides have similar scs
        # this is intended to use the occluding side if conflicting lidar points overlap because of prespective shift
        # in contrast to uSNC_conv the same scs is used for the entire kernel, meaning neighbouring values can dominate an edge location
        if self.scs_pool_d:
            if self.occl_pool_d:
                dcd, cd = x.detach().split(cs.shape[0], 0)
                d_pooling = scs.detach() * cd / (dcd + self.eps)
            else:
                d_pooling = scs.detach()
            x = F.conv2d(x * d_pooling.repeat(2,1,1,1), w_conv_d, padding=self.padding, stride=2) \
                / (F.conv2d(d_pooling, w_conv_d, padding=self.padding, stride=2)+self.eps).repeat(2,1,1,1)
        elif self.occl_pool_d:
            dcd, cd = x.split(cs.shape[0], 0)
            disp = cd / (dcd + self.eps)
            x = F.conv2d(torch.cat((cd, cd*disp), 0), w_conv_d, padding=self.padding, stride=2) \
                / (F.conv2d(disp, w_conv_d, padding=self.padding, stride=2)+self.eps).repeat(2,1,1,1)
        else:
            x = F.conv2d(x, w_conv_d, padding=self.padding, stride=2)

        # pool smoothness
        # prefer low s and high cs
        # the denominator would not be needed if s was the only output
        s_pooling = (cs - scs).detach()
        s_denom = F.conv2d(s_pooling, w_conv_s, padding=self.padding, stride=2) + self.eps
        cs = F.conv2d(cs * s_pooling, w_conv_s, padding=self.padding, stride=2) / s_denom
        scs = F.conv2d(scs * s_pooling, w_conv_s, padding=self.padding, stride=2) / s_denom

        if self.training:
            scs.register_hook(lambda grad: torch.clamp(grad, -1000, 1000))
            cs.register_hook(lambda grad: torch.clamp(grad, -1000, 1000))
        return x, scs, cs

    def visualize_weights(self, rows, cols, col):
        w_conv_d, w_conv_s = self.prepare_weights()

        w_spatial_d = reduce(w_conv_d, 'o i h w -> i h w', 'sum').detach()
        w_channel_d = reduce(w_conv_d, 'o i h w -> o i', 'sum').detach()

        w_spatial_s = reduce(w_conv_s, 'o i h w -> i h w', 'sum').detach()
        w_channel_s = reduce(w_conv_s, 'o i h w -> o i', 'sum').detach()

        idx = col

        plt.axis('off')
        idx+=cols
        idx+=cols

        for c in range(self.n_c):
            if self.kernel_size > 1:
                ax = plt.subplot(rows, cols, idx)
                plt.imshow(w_spatial_s[c, :, :].cpu())
                plt.xlabel('w', fontsize=min(10,100 / rows))
                plt.ylabel('h', fontsize=min(10,100 / rows))
                plt.xticks([])
                plt.yticks([])
            idx+=cols

        if self.n_c > 1:
            ax = plt.subplot(rows, cols, idx)
            plt.xlabel('in', fontsize=min(10,100 / rows))
            plt.ylabel('out', fontsize=min(10,100 / rows))
            plt.imshow(w_channel_s.cpu().detach())
            plt.xticks([])
            plt.yticks([])
            idx+=cols

        idx+=cols

        if self.kernel_size > 1:
            for c in range(self.n_c):
                ax = plt.subplot(rows, cols, idx)
                plt.imshow(w_spatial_d[c, :, :].cpu())
                plt.xlabel('w', fontsize=min(10,100 / rows))
                plt.ylabel('h', fontsize=min(10,100 / rows))
                plt.xticks([])
                plt.yticks([])
                idx+=cols
        else:
            idx+= self.n_c * cols

        if self.n_c > 1:
            ax = plt.subplot(rows, cols, idx)
            plt.xlabel('in', fontsize=min(10,100 / rows))
            plt.ylabel('out', fontsize=min(10,100 / rows))
            plt.imshow(w_channel_d[:, :].cpu())
            plt.xticks([])
            plt.yticks([])
            idx+=cols