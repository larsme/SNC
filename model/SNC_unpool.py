import torch
import torch.nn.functional as F
from einops import rearrange, reduce
import matplotlib.pyplot as plt
import numpy as np

class SNC_unpool(torch.nn.Module):
    def __init__(self, init_method='k', n_c=1, kernel_size=1, symmetric=True, scs_unpool_d=True, focused_unpool_e=True, unfocused_unpool_e=True, no_prop=False, **kwargs):
        super(SNC_unpool, self).__init__()

        # paddings assume an even kernel size, a stride of 2 and input shape = output shape // 2
        self.eps = 1e-20
        self.n_c = n_c
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2 - 1
        self.symmentric = symmetric
        self.scs_unpool_d = scs_unpool_d
        self.focused_unpool_e = focused_unpool_e
        self.unfocused_unpool_e = unfocused_unpool_e
        self.no_prop = no_prop

        # Define Parameters
        self.w_channel_d = torch.nn.Parameter(data=torch.ones(n_c, n_c), requires_grad=n_c > 1)
        self.w_spatial_d = torch.nn.Parameter(data=torch.ones(n_c, kernel_size, kernel_size // 2 if symmetric else kernel_size))
        self.w_skip_d = torch.nn.Parameter(data=torch.ones(1,n_c,1,1))
        self.w_channel_e = torch.nn.Parameter(data=torch.ones(n_c, n_c), requires_grad=n_c > 1)

        if symmetric:
            self.w_spatial_e_0 = torch.nn.Parameter(data=torch.ones(n_c, kernel_size, kernel_size))
            self.w_spatial_e_1 = torch.nn.Parameter(data=torch.ones(n_c, kernel_size, kernel_size // 2))
            self.w_spatial_e_3 = torch.nn.Parameter(data=torch.ones(n_c, kernel_size, kernel_size // 2))
            self.w_dir_e = torch.nn.Parameter(data=torch.ones(n_c, 10))
            self.w_skip_e = torch.nn.Parameter(data=torch.ones(1,n_c, 3,1,1), requires_grad=not no_prop)
        else:
            self.w_spatial_e = torch.nn.Parameter(data=torch.ones(n_c, 4, kernel_size, kernel_size))
            self.w_dir_e = torch.nn.Parameter(data=torch.ones(4, n_c, 4))
            self.w_skip_e = torch.nn.Parameter(data=torch.ones(1, n_c, 4, 1, 1), requires_grad=not no_prop)

        # Init Parameters
        if 'x' in init_method:  # Xavier
            torch.nn.init.xavier_uniform_(self.w_channel_d)
            torch.nn.init.xavier_uniform_(self.w_channel_e)
            torch.nn.init.xavier_uniform_(self.w_dir_e)
            torch.nn.init.xavier_uniform_(self.w_spatial_d)
            torch.nn.init.xavier_uniform_(self.w_skip_d)
            torch.nn.init.xavier_uniform_(self.w_skip_e)
            if symmetric:
                torch.nn.init.xavier_uniform_(self.w_spatial_e_0)
                torch.nn.init.xavier_uniform_(self.w_spatial_e_1)
                torch.nn.init.xavier_uniform_(self.w_spatial_e_3)
            else:
                torch.nn.init.xavier_uniform_(self.w_spatial_e)
        elif 'k' in init_method: # Kaiming
            torch.nn.init.kaiming_uniform_(self.w_channel_d)
            torch.nn.init.kaiming_uniform_(self.w_channel_e)
            torch.nn.init.kaiming_uniform_(self.w_dir_e)
            torch.nn.init.kaiming_uniform_(self.w_spatial_d)
            torch.nn.init.kaiming_uniform_(self.w_skip_d)
            torch.nn.init.kaiming_uniform_(self.w_skip_e)
            if symmetric:
                torch.nn.init.kaiming_uniform_(self.w_spatial_e_0)
                torch.nn.init.kaiming_uniform_(self.w_spatial_e_1)
                torch.nn.init.kaiming_uniform_(self.w_spatial_e_3)
            else:
                torch.nn.init.kaiming_uniform_(self.w_spatial_e)
        if 'e' in init_method:
            self.w_dir_e.data = torch.eye(4)[:,None,:]

    def prepare_weights(self):
        # enforce limits
        w_channel_d = F.softplus(self.w_channel_d)
        w_spatial_d = F.softplus(self.w_spatial_d)
        w_skip_d = torch.sigmoid(self.w_skip_d)
        w_dir_e = F.softplus(self.w_dir_e)
        w_channel_e = F.softplus(self.w_channel_e)
        w_skip_e = torch.sigmoid(self.w_skip_e)
        if self.symmentric:
            w_spatial_e_0 = F.softplus(self.w_spatial_e_0)
            w_spatial_e_1 = F.softplus(self.w_spatial_e_1)
            w_spatial_e_3 = F.softplus(self.w_spatial_e_3)
        else:
            w_spatial_e = F.softplus(self.w_spatial_e)

        if self.no_prop:
            w_skip_e *= 0

        # enforce symmetry by weight sharing
        if self.symmentric:
            w_spatial_d = torch.cat((w_spatial_d,w_spatial_d.flip(dims=(2,))), dim=2)
            #0 => /
            #1 => -
            #2 => \
            #3 => |
            # 1, 3 are symmetric; 2 is a mirror of 0
            w_spatial_e = torch.stack((w_spatial_e_0,
                                    torch.cat((w_spatial_e_1, w_spatial_e_1.flip(dims=(2,))), dim=2),
                                    w_spatial_e_0.flip(dims=(2,)),
                                    torch.cat((w_spatial_e_3, w_spatial_e_3.flip(dims=(2,))), dim=2)), dim=1)
            # connect directions to each other; with connections from or to 0 and 2 sharing the same weights
            w_dir_e = w_dir_e.unbind(1)
            w_dir_e = torch.stack((torch.stack((w_dir_e[0], w_dir_e[1], w_dir_e[2], w_dir_e[3]), dim = 1),
                                   torch.stack((w_dir_e[4], w_dir_e[5], w_dir_e[4], w_dir_e[6]), dim = 1),
                                   torch.stack((w_dir_e[2], w_dir_e[1], w_dir_e[0], w_dir_e[3]), dim = 1),
                                   torch.stack((w_dir_e[7], w_dir_e[8], w_dir_e[7], w_dir_e[9]), dim = 1)),
                                   dim=0)
            w_skip_e = torch.cat((w_skip_e, w_skip_e[:,:,0,None,:,:]), dim=2)

        # normalize by output channel
        # technically not needed for d, but here for consistency
        w_channel_d = w_channel_d / reduce(w_channel_d,'o i -> o 1','sum')
        w_spatial_d = w_spatial_d / reduce(w_spatial_d,'i h w -> i 1 1','sum')
        w_channel_e = w_channel_e / reduce(w_channel_e,'o i -> o 1','sum')
        w_dir_e = w_dir_e / reduce(w_dir_e,    'd2 i d1 -> d2 i 1','sum')
        w_spatial_e = w_spatial_e / reduce(w_spatial_e,'i d h w -> i d 1 1','sum')

        # combine seperable convolution for speed
        w_conv_d = torch.einsum('o i, i h w -> o i h w', w_channel_d, w_spatial_d)
        w_conv_e = rearrange(torch.einsum('o i, p i d, i d h w -> o p i d h w', w_channel_e, w_dir_e, w_spatial_e), 'o d2 i d1 h w -> (o d2) (i d1) h w')

        return w_conv_d, w_conv_e, w_skip_d, w_skip_e

    def prep_eval(self):
        self.weights = self.prepare_weights()

    def forward(self, x_skip, ece_skip, ce_skip, x_pool, ece_pool, ce_pool):
        # x = d*cd, cd
        # d = depth
        # cd = confidence over depth
        # ece = directed smoothness * ce;  dim 2 corresponds to edge directions: /, -, \, |
        # ce = confidence over directed smoothness

        if self.training:
            w_conv_d, w_conv_e, w_skip_d, w_skip_e = self.prepare_weights()
        else:
            w_conv_d, w_conv_e, w_skip_d, w_skip_e = self.weights

        # unpooling d
        # even if there is an edge, it would be difficult to assign the d_pool to one side, so it is unpooled without s_skip
        if self.kernel_size == 2 or not self.scs_unpool_d:
            # no need for smoothness since it would factor out if nothig overlaps
            x_pool = F.conv_transpose2d(x_pool, w_conv_d, padding=self.padding, stride=2)
        else:
            # only use a single s_pool factor as opposed to uSNC
            # if a location is on an edge, each side of the unpooled version will depend on values from their side of the edge
            scs_pool = reduce(ece_pool, 'b c d h w -> b c h w', 'prod')
            x_pool = F.conv_transpose2d(x_pool * scs_pool.repeat(2,1,1,1), w_conv_d, padding=self.padding, stride=2) \
                / (F.conv_transpose2d(scs_pool, w_conv_d, padding=self.padding, stride=2) + self.eps).repeat(2,1,1,1)
 
        # unpooling e
        # if the pooled data predicts an edge, the skip connection knows where (low e) or where not (high ece at one point, less confidence at another)
        #    => during unpooling of e, favour locations with low ece_skip
        # if the pooled data does not predict an edge, it should not be focused onto edges
        #    => deconv without skip
        # combine both versions with a nconv weighted by the unfocused version

        ece_pool = rearrange(ece_pool, 'b c d h w -> b (c d) h w')
        ce_pool = rearrange(ce_pool, 'b c d h w -> b (c d) h w')
        if self.unfocused_unpool_e and self.focused_unpool_e:
            w_s = 1 / (rearrange(ece_skip, 'b c d h w -> b (c d) h w') + self.eps)
            w_s_sum = F.conv2d(w_s, w_conv_e, padding=self.padding, stride=2) + self.eps
            # divide by w_s_sum first in this deconvolution, then multiply by w_s
            ce_pool_focus = F.conv_transpose2d(ce_pool / w_s_sum , w_conv_e, padding=self.padding, stride=2) * w_s
            ece_pool_focus = F.conv_transpose2d(ece_pool / w_s_sum, w_conv_e, padding=self.padding, stride=2) * w_s

            ce_pool_unfocused = F.conv_transpose2d(ce_pool, w_conv_e, padding=self.padding, stride=2)
            ece_pool_unfocused = F.conv_transpose2d(ece_pool, w_conv_e, padding=self.padding, stride=2)

            e_pool_unfocused = (ece_pool_unfocused / (ce_pool_unfocused + self.eps)).detach()
            ce_pool = rearrange(ece_pool_unfocused + (1 - e_pool_unfocused) * ce_pool_focus, 'b (c d) h w -> b c d h w', d=4)
            ece_pool = rearrange(e_pool_unfocused * ece_pool_unfocused + (1 - e_pool_unfocused) * ece_pool_focus, 'b (c d) h w -> b c d h w',d=4)
        elif self.focused_unpool_e:
            w_s = 1 / (rearrange(ece_skip, 'b c d h w -> b (c d) h w') + self.eps)
            w_s_sum = F.conv2d(w_s, w_conv_e, padding=self.padding, stride=2) + self.eps
            # divide by w_s_sum first in this deconvolution, then multiply by w_s
            ce_pool = rearrange(F.conv_transpose2d(ce_pool / w_s_sum , w_conv_e, padding=self.padding, stride=2) * w_s, 'b (c d) h w -> b c d h w',d=4)
            ece_pool = rearrange(F.conv_transpose2d(ece_pool / w_s_sum, w_conv_e, padding=self.padding, stride=2) * w_s, 'b (c d) h w -> b c d h w',d=4)
        else:
            ce_pool = rearrange(F.conv_transpose2d(ce_pool, w_conv_e, padding=self.padding, stride=2), 'b (c d) h w -> b c d h w',d=4)
            ece_pool = rearrange(F.conv_transpose2d(ece_pool, w_conv_e, padding=self.padding, stride=2), 'b (c d) h w -> b c d h w',d=4)
        s_pool = reduce(ece_pool / (ce_pool + self.eps), 'b c d h w -> b c h w', 'prod')

        # combining pool and skip
        # in general, each should have proportionally higher c in areas they are more suited to in terms of distance
        # additionally, skip is prefered around edges because of its higher resolution
        # to determine whether there is an edge, s_pool used
        # it has values anywhere where there is data to interpolate, likely includes less input errors and is less likely to have gaps in edges
        #    => use w_skip, w_pool*s_pool

        w_pool_d = (1 - w_skip_d) * s_pool
        w_sum_d = w_skip_d + w_pool_d + self.eps
        x = (w_skip_d * x_skip + w_pool_d.repeat(2,1,1,1) * x_pool) / w_sum_d.repeat(2,1,1,1)

        w_pool_e = (1 - w_skip_e) * s_pool[:,:,None,:,:]
        w_sum_e = w_skip_e + w_pool_e + self.eps
        ce = (w_skip_e * ce_skip + w_pool_e * ce_pool) / w_sum_e
        ece = (w_skip_e * ece_skip + w_pool_e * ece_pool) / w_sum_e

        if ece.requires_grad:
            ece.register_hook(lambda grad: torch.clamp(grad, -1000, 1000))
            ce.register_hook(lambda grad: torch.clamp(grad, -1000, 1000))

        return x, ece, ce

    def visualize_weights(self, rows, cols, col):
        w_conv_d, w_conv_e, w_skip_d, w_skip_e = self.prepare_weights()

        w_spatial_d = reduce(w_conv_d, 'o i h w -> i h w', 'sum')
        w_channel_d = reduce(w_conv_d, 'o i h w -> o i', 'sum')

        w_conv_e = rearrange(w_conv_e, '(o d2) (i d1) h w -> o d2 i d1 h w', d1=4, d2=4)
        w_spatial_e = reduce(w_conv_e, 'o d2 i d1 h w -> i d1 h w', 'sum')
        w_channel_e = reduce(w_conv_e, 'o d2 i d1 h w -> o i', 'sum')
        w_dir_e = reduce(w_conv_e, 'o d2 i d1 h w -> d2 i d1', 'sum')

        idx = col
        plt.axis('off')

        idx+=cols * self.n_c
        idx+=cols

        for c in range(self.n_c):
            for d in range(4):
                if self.kernel_size > 1:
                    ax = plt.subplot(rows, cols, idx)
                    plt.imshow(w_spatial_e[c, d, :, :])
                    plt.xlabel('w', fontsize=min(10,100 / rows))
                    plt.ylabel('h', fontsize=min(10,100 / rows))
                    plt.xticks([])
                    plt.yticks([])
                idx+=cols

        if self.n_c > 1:
            ax = plt.subplot(rows, cols, idx)
            plt.xlabel('in', fontsize=min(10,100 / rows))
            plt.ylabel('out', fontsize=min(10,100 / rows))
            plt.imshow(w_channel_e)
            plt.xticks([])
            plt.yticks([])
            idx+=cols

        for c in range(self.n_c):
            ax = plt.subplot(rows, cols, idx)
            plt.imshow(w_dir_e[:,c, :])
            plt.xticks([0,1,2,3],['/','-','\\','|'], fontsize=min(10,100 / rows))
            plt.yticks([0,1,2,3],['/','-','\\','|'], fontsize=min(10,100 / rows))
            ax.tick_params(length=0)
            plt.ylabel('out', fontsize=min(10,100 / rows))
            idx+=cols

        ax = plt.subplot(rows, cols, idx)
        plt.imshow(w_skip_e[0,:,:, 0, 0])
        for (j,i),label in np.ndenumerate(w_skip_e[0,:,:, 0, 0]):
            ax.text(i,j,'{:.02f}'.format(label),ha='center',va='center', fontsize=min(10,80 / rows), color='tomato')
        plt.xticks([0,1,2,3],['/','-','\\','|'], fontsize=min(10,100 / rows))
        plt.yticks([])
        if self.n_c > 1:
            plt.ylabel('c', fontsize=min(10,100 / rows))
        ax.tick_params(length=0)
        idx+=cols

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