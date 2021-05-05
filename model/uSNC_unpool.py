import torch
import torch.nn.functional as F
from einops import rearrange, reduce
import matplotlib.pyplot as plt
import numpy as np

class uSNC_unpool(torch.nn.Module):
    def __init__(self, init_method='k', n_c=1, kernel_size=1, symmetric=True, scs_unpool_d=True, focused_unpool_s=True, unfocused_unpool_s=True, **kwargs):
        super(uSNC_unpool, self).__init__()

        # paddings assume an even kernel size, a stride of 2 and input shape = output shape // 2
        self.eps = 1e-20
        self.n_c = n_c
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2 - 1
        self.symmentric = symmetric
        self.scs_unpool_d = scs_unpool_d
        self.focused_unpool_s = focused_unpool_s
        self.unfocused_unpool_s = unfocused_unpool_s

        # Define Parameters
        self.w_channel_d = torch.nn.Parameter(data=torch.ones(n_c, n_c), requires_grad=n_c > 1)
        self.w_spatial_d = torch.nn.Parameter(data=torch.ones(n_c, kernel_size, (kernel_size + 1) // 2 if symmetric else kernel_size))
        self.w_skip_d = torch.nn.Parameter(data=torch.ones(1,n_c,1,1))

        self.w_channel_s = torch.nn.Parameter(data=torch.ones(n_c, n_c), requires_grad=n_c > 1)
        self.w_spatial_s = torch.nn.Parameter(data=torch.ones(n_c, kernel_size, (kernel_size + 1) // 2 if symmetric else kernel_size))
        self.w_skip_s = torch.nn.Parameter(data=torch.ones(1,n_c,1,1))

        # Init Parameters
        if 'x' in init_method:  # Xavier
            torch.nn.init.xavier_uniform_(self.w_channel_d)
            torch.nn.init.xavier_uniform_(self.w_channel_s)
            torch.nn.init.xavier_uniform_(self.w_spatial_d)
            torch.nn.init.xavier_uniform_(self.w_skip_d)
            torch.nn.init.xavier_uniform_(self.w_spatial_s)
            torch.nn.init.uniform_(self.w_skip_s)
        elif 'k' in init_method: # Kaiming
            torch.nn.init.kaiming_uniform_(self.w_channel_d)
            torch.nn.init.kaiming_uniform_(self.w_channel_s)
            torch.nn.init.kaiming_uniform_(self.w_spatial_d)
            torch.nn.init.kaiming_uniform_(self.w_skip_d)
            torch.nn.init.kaiming_uniform_(self.w_spatial_s)
            torch.nn.init.uniform_(self.w_skip_s)

    def prepare_weights(self):
        # enforce limits
        w_channel_d = F.softplus(self.w_channel_d)
        w_spatial_d = F.softplus(self.w_spatial_d)
        w_skip_d = torch.sigmoid(self.w_skip_d)
        w_channel_s = F.softplus(self.w_channel_s)
        w_spatial_s = F.softplus(self.w_spatial_s)
        w_skip_s = torch.sigmoid(self.w_skip_s)

        # enforce symmetry by weight sharing
        if self.symmentric:
            w_spatial_d = torch.cat((w_spatial_d,w_spatial_d[:,:,:self.kernel_size // 2].flip(dims=(2,))), dim=2)
            w_spatial_s = torch.cat((w_spatial_s,w_spatial_s[:,:,:self.kernel_size // 2].flip(dims=(2,))), dim=2)

        # normalize by output channel
        w_channel_d = w_channel_d / reduce(w_channel_d,'o i -> o 1','sum')
        w_spatial_d = w_spatial_d / reduce(w_spatial_d,'i h w -> i 1 1','sum')
        w_channel_s = w_channel_s / reduce(w_channel_s,'o i -> o 1','sum')
        w_spatial_s = w_spatial_s / reduce(w_spatial_s,'i h w -> i 1 1','sum')

        # combine seperable convolution for speed
        w_conv_d = torch.einsum('o i, i h w -> o i h w', w_channel_d, w_spatial_d)
        w_conv_s = torch.einsum('o i, i h w -> o i h w', w_channel_s, w_spatial_s)

        return w_conv_d, w_conv_s, w_skip_d, w_skip_s

    def prep_eval(self):
        self.weights = self.prepare_weights()

    def forward(self, x_pool, scs_pool, cs_pool, x_skip, scs_skip, cs_skip):
        # x = d*cd, cd
        # d = depth
        # cd = confidence over depth
        # scs = smoothness * cs
        # cs = confidence over smoothness
        # _skip => skip connection with target resolution

        if self.training:
            w_conv_d, w_conv_s, w_skip_d, w_skip_s = self.prepare_weights()
        else:
            w_conv_d, w_conv_s, w_skip_d, w_skip_s = self.weights

        # unpooling d
        # even if there is an edge, it would be difficult to assign the d_pool to one side, so it is unpooled without s_skip
        if self.kernel_size == 2 or not self.scs_unpool_d:
            # no need for smoothness since it would factor out if nothig overlaps
            x_pool = F.conv_transpose2d(x_pool, w_conv_d, padding=self.padding, stride=2)
        else:
            # only use a single s_pool factor as opposed to uSNC
            # if a location is on an edge, each side of the unpooled version will depend on values from their side of the edge
            x_pool = F.conv_transpose2d(x_pool * scs_pool.repeat(2,1,1,1), w_conv_d, padding=self.padding, stride=2) \
                / (F.conv_transpose2d(scs_pool, w_conv_d, padding=self.padding, stride=2) + self.eps).repeat(2,1,1,1)

        # unpooling s
        # if the pooled data predicts an edge, the skip connection knows where (low s) or where not (high scs at one point, less confidence at another)
        #    => during unpooling of s, favour locations with low scs_skip
        # if the pooled data does not predict an edge, it should not be focused onto edges
        #    => deconv without skip
        # combine both versions with a nconv weighted by the unfocused version

        if self.unfocused_unpool_s and self.focused_unpool_s:
            w_s = 1 / (scs_skip + self.eps).detach()
            w_s_sum = F.conv2d(w_s, w_conv_s, padding=self.padding, stride=2) + self.eps
            # divide by w_s_sum first in this deconvolution, then multiply by w_s
            cs_pool_focus = F.conv_transpose2d(cs_pool / w_s_sum , w_conv_s, padding=self.padding, stride=2) * w_s
            scs_pool_focus = F.conv_transpose2d(scs_pool / w_s_sum, w_conv_s, padding=self.padding, stride=2) * w_s

            cs_pool_unfocused = F.conv_transpose2d(cs_pool, w_conv_s, padding=self.padding, stride=2)
            scs_pool_unfocused = F.conv_transpose2d(scs_pool, w_conv_s, padding=self.padding, stride=2)

            s_pool_unfocused = (scs_pool_unfocused / (cs_pool_unfocused + self.eps)).detach()
            cs_pool = scs_pool_unfocused + (1 - s_pool_unfocused) * cs_pool_focus
            scs_pool = s_pool_unfocused * scs_pool_unfocused + (1 - s_pool_unfocused) * scs_pool_focus
        elif self.focused_unpool_s:
            w_s = 1 / (scs_skip + self.eps).detach()
            w_s_sum = F.conv2d(w_s, w_conv_s, padding=self.padding, stride=2) + self.eps
            # divide by w_s_sum first in this deconvolution, then multiply by w_s
            cs_pool = F.conv_transpose2d(cs_pool / w_s_sum , w_conv_s, padding=self.padding, stride=2) * w_s
            scs_pool = F.conv_transpose2d(scs_pool / w_s_sum, w_conv_s, padding=self.padding, stride=2) * w_s
        else:
            cs_pool = F.conv_transpose2d(cs_pool, w_conv_s, padding=self.padding, stride=2)
            scs_pool = F.conv_transpose2d(scs_pool, w_conv_s, padding=self.padding, stride=2)
        s_pool = scs_pool / (cs_pool + self.eps)
        if self.focused_unpool_s:
            s_pool = s_pool.detach()

        # combining pool and skip
        # in general, each should have proportionally higher c in areas they are more suited to in terms of distance
        # additionally, skip is prefered around edges because of its higher resolution
        # to determine whether there is an edge, s_pool used
        # it has values anywhere where there is data to interpolate, likely includes less input errors and is less likely to have gaps in edges
        #    => use w_skip, w_pool*s_pool
        
        w_pool_d = (1 - w_skip_d) * s_pool
        w_sum_d = w_skip_d + w_pool_d + self.eps
        x = (w_skip_d * x_skip + w_pool_d.repeat(2,1,1,1) * x_pool) / w_sum_d.repeat(2,1,1,1)

        w_pool_s = (1 - w_skip_s) * s_pool
        w_sum_s = w_skip_s + w_pool_s + self.eps
        cs = (w_skip_s * cs_skip + w_pool_s * cs_pool) / w_sum_s
        scs = (w_skip_s * scs_skip + w_pool_s * scs_pool) / w_sum_s

        if self.training:
            scs.register_hook(lambda grad: torch.clamp(grad, -1000, 1000))
            cs.register_hook(lambda grad: torch.clamp(grad, -1000, 1000))
        return x, scs, cs

    def visualize_weights(self, rows, cols, col):
        w_conv_d, w_conv_s, w_skip_d, w_skip_s = self.prepare_weights()

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
            plt.imshow(w_channel_s.cpu())
            plt.xticks([])
            plt.yticks([])
            idx+=cols

        ax = plt.subplot(rows, cols, idx)
        plt.imshow(w_skip_s[:,:, 0, 0].cpu().detach())
        for (j,i),label in np.ndenumerate(w_skip_s[:,:, 0, 0].cpu().detach()):
            ax.text(i,j,'{:.02f}'.format(label),ha='center',va='center', fontsize=min(10,80 / rows), color='tomato')
        plt.xticks([])
        plt.yticks([])
        if self.n_c > 1:
            plt.ylabel('c', fontsize=min(10,100 / rows))
        ax.tick_params(length=0)
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

        if self.n_c > 1:
            ax = plt.subplot(rows, cols, idx)
            plt.xlabel('in', fontsize=min(10,100 / rows))
            plt.ylabel('out', fontsize=min(10,100 / rows))
            plt.imshow(w_channel_d[:, :].cpu())
            plt.xticks([])
            plt.yticks([])
            idx+=cols

        ax = plt.subplot(rows, cols, idx)
        plt.imshow(w_skip_d[:,:,0,0].cpu().detach())
        for (j,i),label in np.ndenumerate(w_skip_d[:,:, 0, 0].cpu().detach()):
            ax.text(i,j,'{:.02f}'.format(label),ha='center',va='center', fontsize=min(10,80 / rows), color='tomato')
        plt.xticks([])
        plt.yticks([])
        if self.n_c > 1:
            plt.xlabel('c', fontsize=min(10,100 / rows))
        idx+=cols