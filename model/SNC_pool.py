import torch
import torch.nn.functional as F
from einops import rearrange, reduce
import matplotlib.pyplot as plt

class SNC_pool(torch.nn.Module):
    def __init__(self, init_method='k', n_c=1, kernel_size=1, symmetric=True, occl_pool_d=True, scs_pool_d=True, no_prop=False, **kwargs):
        super(SNC_pool, self).__init__()

        # paddings assume an even kernel size, a stride of 2 and input shape = output shape // 2
        self.eps = 1e-20
        self.n_c = n_c
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2 - 1
        self.symmentric = symmetric
        self.occl_pool_d = occl_pool_d
        self.scs_pool_d = scs_pool_d
        self.no_prop = no_prop

        # Define Parameters
        self.w_channel_d = torch.nn.Parameter(data=torch.ones(n_c, n_c), requires_grad=n_c > 1)
        self.w_spatial_d = torch.nn.Parameter(data=torch.ones(n_c, kernel_size, kernel_size // 2 if symmetric else kernel_size))
        self.w_channel_e = torch.nn.Parameter(data=torch.ones(n_c, n_c), requires_grad=n_c > 1)

        if symmetric:
            self.w_spatial_e_0 = torch.nn.Parameter(data=torch.ones(n_c, kernel_size, kernel_size), requires_grad=not no_prop)
            self.w_spatial_e_1 = torch.nn.Parameter(data=torch.ones(n_c, kernel_size, kernel_size // 2), requires_grad=not no_prop)
            self.w_spatial_e_3 = torch.nn.Parameter(data=torch.ones(n_c, kernel_size, kernel_size // 2), requires_grad=not no_prop)
            self.w_dir_e = torch.nn.Parameter(data=torch.ones(n_c, 10), requires_grad=not no_prop)
        else:
            self.w_spatial_e = torch.nn.Parameter(data=torch.ones(n_c, 4, kernel_size, kernel_size), requires_grad=not no_prop)
            self.w_dir_e = torch.nn.Parameter(data=torch.ones(4, n_c, 4), requires_grad=not no_prop)

        # Init Parameters
        if 'x' in init_method:  # Xavier
            torch.nn.init.xavier_uniform_(self.w_channel_d)
            torch.nn.init.xavier_uniform_(self.w_channel_e)
            torch.nn.init.xavier_uniform_(self.w_dir_e)
            torch.nn.init.xavier_uniform_(self.w_spatial_d)
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
        w_dir_e = F.softplus(self.w_dir_e)
        w_channel_e = F.softplus(self.w_channel_e)
        if self.symmentric:
            w_spatial_e_0 = F.softplus(self.w_spatial_e_0)
            w_spatial_e_1 = F.softplus(self.w_spatial_e_1)
            w_spatial_e_3 = F.softplus(self.w_spatial_e_3)
        else:
            w_spatial_e = F.softplus(self.w_spatial_e)

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

        return w_conv_d, w_conv_e

    def prep_eval(self):
        self.weights = self.prepare_weights()

    def forward(self, x, ece, ce):
        # x = d*cd, cd
        # d = depth
        # cd = confidence over depth
        # ece = directed smoothness * ce;  dim 2 corresponds to edge directions: /, -, \, |
        # ce = confidence over directed smoothness

        if self.training:
            w_conv_d, w_conv_e = self.prepare_weights()
        else:
            w_conv_d, w_conv_e = self.weights

        # pool depth
        # prefer high s, high cs and high cd
        # prefer low d, assuming this only makes a difference on edges where all sides have similar ece
        # this is intended to use the occluding side if conflicting lidar points overlap because of prespective shift
        # in contrast to uSNC_conv the same ece is used for the entire kernel, meaning neighbouring values can dominate an edge location
        scs = reduce(ece, 'b c d h w -> b c h w', 'prod')
        if self.scs_pool_d:
            if self.occl_pool_d:
                dcd, cd = x.detach().split(ce.shape[0], 0)
                d_pooling = scs.detach() * cd / (dcd + self.eps)
            else:
                d_pooling = scs.detach()
            x = F.conv2d(x * d_pooling.repeat(2,1,1,1), w_conv_d, padding=self.padding, stride=2) \
                / (F.conv2d(d_pooling, w_conv_d, padding=self.padding, stride=2)+self.eps).repeat(2,1,1,1)
        elif self.occl_pool_d:
            dcd, cd = x.split(ce.shape[0], 0)
            disp = cd / (dcd + self.eps)
            x = F.conv2d(torch.cat((cd, cd*disp), 0), w_conv_d, padding=self.padding, stride=2) \
                / (F.conv2d(disp, w_conv_d, padding=self.padding, stride=2)+self.eps).repeat(2,1,1,1)
        else:
            x = F.conv2d(x, w_conv_d, padding=self.padding, stride=2)

        # pool smoothness
        # prefer low s and high ce
        # the denominator would not be needed if s was the only output
        e_pooling = (ce - ece).detach()
        e_denom = F.conv2d(rearrange(e_pooling, 'b c d h w -> b (c d) h w'), w_conv_e, padding=self.padding, stride=2) + self.eps
        ce = rearrange(F.conv2d(rearrange(ce * e_pooling, 'b c d h w -> b (c d) h w'), w_conv_e, padding=self.padding, stride=2) / e_denom, 'b (c d) h w -> b c d h w', d=4)
        ece = rearrange(F.conv2d(rearrange(ece * e_pooling, 'b c d h w -> b (c d) h w'), w_conv_e, padding=self.padding, stride=2) / e_denom, 'b (c d) h w -> b c d h w', d=4)
        
        if ece.requires_grad:
            ece.register_hook(lambda grad: torch.clamp(grad, -1000, 1000))
            ce.register_hook(lambda grad: torch.clamp(grad, -1000, 1000))

        return x, ece, ce

    def visualize_weights(self, rows, cols, col):
        w_conv_d, w_conv_e = self.prepare_weights()

        w_spatial_d = reduce(w_conv_d, 'o i h w -> i h w', 'sum').detach()
        w_channel_d = reduce(w_conv_d, 'o i h w -> o i', 'sum').detach()

        w_conv_e = rearrange(w_conv_e, '(o d2) (i d1) h w -> o d2 i d1 h w', d1=4, d2=4).detach()
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
                    plt.imshow(w_spatial_e[c, d, :, :].cpu())
                    plt.xlabel('w', fontsize=min(10,100 / rows))
                    plt.ylabel('h', fontsize=min(10,100 / rows))
                    plt.xticks([])
                    plt.yticks([])
                idx+=cols

        if self.n_c > 1:
            ax = plt.subplot(rows, cols, idx)
            plt.xlabel('in', fontsize=min(10,100 / rows))
            plt.ylabel('out', fontsize=min(10,100 / rows))
            plt.imshow(w_channel_e.cpu())
            plt.xticks([])
            plt.yticks([])
            idx+=cols

        for c in range(self.n_c):
            ax = plt.subplot(rows, cols, idx)
            plt.imshow(w_dir_e[:,c, :].cpu())
            plt.xticks([0,1,2,3],['/','-','\\','|'], fontsize=min(10,100 / rows))
            plt.yticks([0,1,2,3],['/','-','\\','|'], fontsize=min(10,100 / rows))
            ax.tick_params(length=0)
            plt.ylabel('out', fontsize=min(10,100 / rows))
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

        if self.n_c > 1:
            ax = plt.subplot(rows, cols, idx)
            plt.xlabel('in', fontsize=min(10,100 / rows))
            plt.ylabel('out', fontsize=min(10,100 / rows))
            plt.imshow(w_channel_d[:, :].cpu())
            plt.xticks([])
            plt.yticks([])
            idx+=cols