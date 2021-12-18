import torch
import torch.nn.functional as F
from einops import rearrange, reduce
import matplotlib.pyplot as plt
import numpy as np

from model.retrieve_indices import retrieve_indices

class uSNC_conv(torch.nn.Module):
    def __init__(self, init_method='k', n_in=1, n_out=1, kernel_size=1, max_pool_size=1, symmetric=True, **kwargs):
        super(uSNC_conv, self).__init__()

        # paddings assume an odd kernel size and input shape = output shape
        self.eps = 1e-20
        self.n_in = n_in
        self.n_out = n_out
        assert n_in == 1 or n_out == 1 or n_out == n_in
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.max_pool_size = max_pool_size
        self.max_pool_padding = max_pool_size // 2
        self.symmentric = symmetric

        # Define Parameters
        self.w_channel_d = torch.nn.Parameter(data=torch.ones(n_out, n_in), requires_grad=n_in > 1)
        self.w_spatial_d = torch.nn.Parameter(data=torch.ones(n_in, kernel_size, (kernel_size + 1) // 2 if symmetric else kernel_size), requires_grad=kernel_size > 1)

        self.w_pow_s = torch.nn.Parameter(data=torch.ones(n_in))
        self.w_prop_s = torch.nn.Parameter(data=torch.ones(n_in))
        self.w_channel_s = torch.nn.Parameter(data=torch.ones(n_out, n_in), requires_grad=n_in > 1)
        self.w_spatial_s = torch.nn.Parameter(data=torch.ones(n_in, kernel_size, (kernel_size + 1) // 2 if symmetric else kernel_size), requires_grad=kernel_size > 1)

        self.unfold = torch.nn.Unfold(kernel_size=self.kernel_size,dilation=1,padding=self.padding,stride=1)

        # Init Parameters
        if 'x' in init_method:  # Xavier
            torch.nn.init.xavier_uniform_(self.w_channel_d)
            torch.nn.init.xavier_uniform_(self.w_channel_s)
            torch.nn.init.xavier_uniform_(self.w_spatial_d)
            torch.nn.init.uniform_(self.w_prop_s)
            torch.nn.init.uniform_(self.w_pow_s)
            torch.nn.init.xavier_uniform_(self.w_spatial_s)
        elif 'k' in init_method: # Kaiming
            torch.nn.init.kaiming_uniform_(self.w_channel_d)
            torch.nn.init.kaiming_uniform_(self.w_channel_s)
            torch.nn.init.kaiming_uniform_(self.w_spatial_d)
            torch.nn.init.uniform_(self.w_prop_s)
            torch.nn.init.uniform_(self.w_pow_s)
            torch.nn.init.kaiming_uniform_(self.w_spatial_s)

    def prepare_weights(self):
        # enforce limits
        w_channel_d = F.softplus(self.w_channel_d)
        w_spatial_d = F.softplus(self.w_spatial_d)
        w_prop_s = torch.sigmoid(self.w_prop_s)
        w_pow_s = F.softplus(self.w_pow_s)
        w_channel_s = F.softplus(self.w_channel_s)
        w_spatial_s = F.softplus(self.w_spatial_s)

        # enforce symmetry by weight sharing
        if self.symmentric:
            w_spatial_d = torch.cat((w_spatial_d,w_spatial_d[:,:,:self.kernel_size // 2].flip(dims=(2,))), dim=2)
            w_spatial_s = torch.cat((w_spatial_s,w_spatial_s[:,:,:self.kernel_size // 2].flip(dims=(2,))), dim=2)

        # normalize by output channel
        # technically not needed for d, but here for consistency
        w_channel_d = w_channel_d / reduce(w_channel_d,'o i -> o 1','sum')
        w_spatial_d = w_spatial_d / reduce(w_spatial_d,'i h w -> i 1 1','sum')
        w_channel_s = w_channel_s / reduce(w_channel_s,'o i -> o 1','sum')
        w_spatial_s = w_spatial_s / reduce(w_spatial_s,'i h w -> i 1 1','sum')

        # combine seperable convolution for speed
        w_conv_d = rearrange(torch.einsum('o i, i h w -> o i h w', w_channel_d, w_spatial_d), 'o i h w -> o i (h w)')

        if self.n_in >= self.n_out:
            w_conv_s = torch.einsum('o i, i h w -> o i h w', w_channel_s, w_spatial_s)
            return w_conv_d, w_prop_s, w_pow_s, w_conv_s, None
        else:
            w_conv_s = reorder(w_spatial_s, 'i h w -> i 1 h w')
            return w_conv_d, w_prop_s, w_pow_s, w_conv_s, w_channel_s

    def prep_eval(self):
        self.weights = self.prepare_weights()

    def forward(self, x, scs, cs):
        # x = d*cd, cd
        # d = depth
        # cd = confidence over depth
        # scs = smoothness * cs
        # cs = confidence over smoothness

        if self.training:
            w_conv_d, w_prop_s, w_pow_s, w_conv_s, w_channel_s = self.prepare_weights()
        else:
            w_conv_d, w_prop_s, w_pow_s, w_conv_s, w_channel_s = self.weights
            
        dcd, cd = x.split(cs.shape[0], 0)
        d_pad, cd_pad = F.pad(dcd.detach() / (cd.detach() + self.eps), (self.max_pool_padding,) * 4), F.pad(cd.detach(), (self.max_pool_padding,) * 4)
        # pick the confidence weighted min and max depths
        # divide min and max to the power of w_pow_s to calculate a new smoothness estimate
        _, j_max = F.max_pool2d(d_pad * cd_pad, kernel_size=self.max_pool_size, return_indices=True, padding=0, stride=1)
        _, j_min = F.max_pool2d(cd_pad / (d_pad + self.eps), kernel_size=self.max_pool_size, return_indices=True, padding=0, stride=1)
        s_new = ((retrieve_indices(d_pad, j_min) + self.eps) / (retrieve_indices(d_pad, j_max) + self.eps)).pow(w_pow_s)
        cs_new = retrieve_indices(cd_pad, j_max) * retrieve_indices(cd_pad, j_min)

        # propagate smoothness
        if self.n_in >= self.n_out:
            cs = F.conv2d(cs * w_prop_s + cs_new * (1 - w_prop_s), w_conv_s, padding=self.padding)
            scs = F.conv2d(scs * w_prop_s + s_new * cs_new * (1 - w_prop_s), w_conv_s, padding=self.padding)
            s = scs / (cs + self.eps)
        else:
            cs = F.conv2d(cs * w_prop_s + cs_new * (1 - w_prop_s), w_conv_s, padding=self.padding, groups=self.n_in)
            scs = F.conv2d(scs * w_prop_s + s_new * cs_new * (1 - w_prop_s), w_conv_s, padding=self.padding, groups=self.n_in)
            s = scs / (cs + self.eps)
            cs = torch.einsum('o i, b i h w -> b o h w', w_channel_s, cs)
            scs = torch.einsum('o i, b i h w -> b o h w', w_channel_s, scs)

        # propagate depth
        if self.kernel_size == 3:
            # no penalty to stay in place
            # propagating from a location to a neighbour is penalized by either being an edge
            w_s = rearrange(self.unfold(s), 'b (i k) (h w) -> b i k h w', i = self.n_in, h=s.shape[2]) * rearrange(s, 'b i h w -> b i 1 h w')
            w_s[:,:,4,:,:] = 1
        elif self.kernel_size == 5:
            w_s = self.w_s5(s)
        if self.n_out < self.n_in:
            w_s = w_s.expand(-1,d.shape[1],-1,-1,-1)
            
        x = (torch.stack((torch.einsum('b i k h w, o i k-> b o h w', self.unfold(dcd).view(w_s.shape) * w_s, w_conv_d),
                          torch.einsum('b i k h w, o i k-> b o h w', self.unfold(cd).view(w_s.shape) * w_s, w_conv_d)), 
                         1) / (torch.einsum('b i k h w, o i k -> b o h w', w_s, w_conv_d)[:,None,:,:,:] + self.eps)).view(2 * s.shape[0], *s.shape[1:])

        if self.training:
            scs.register_hook(lambda grad: torch.clamp(grad, -1000, 1000))
            cs.register_hook(lambda grad: torch.clamp(grad, -1000, 1000))
        return x, scs, cs

    def w_s5(self, s):
        # no penalty to stay in place
        # propagating from a location to a neighbour is penalized by either being an edge
        # propagating to a neighbour's neighbour is penalized by all 3 being edges
        # in cases where multiple middle neighbours are equally suited, they are averaged
        # it should be noted that smoothness of the middle neighbour is only used once, as opposed to once for entering and once for exiting

        s_pad = F.pad(s, (2,2,2,2))

        # direct neighbours
        s11 = s * s_pad[:, :, 1:-3,1:-3]
        s12 = s * s_pad[:, :, 1:-3,2:-2]
        s13 = s * s_pad[:, :, 1:-3,3:-1]
        s21 = s * s_pad[:, :, 2:-2,1:-3]
        s23 = s * s_pad[:, :, 2:-2,3:-1]
        s31 = s * s_pad[:, :, 3:-1,1:-3]
        s32 = s * s_pad[:, :, 3:-1,2:-2]
        s33 = s * s_pad[:, :, 3:-1,3:-1]

        # diagonals
        s00 = s11 * s_pad[:, :,  :-4, :-4]
        s04 = s13 * s_pad[:, :,  :-4,4:  ]
        s40 = s31 * s_pad[:, :, 4:  , :-4]
        s44 = s33 * s_pad[:, :, 4:  ,4:  ]

        # center lines
        s02 = s12 * s_pad[:, :,  :-4,2:-2]
        s20 = s21 * s_pad[:, :, 2:-2, :-4]
        s24 = s23 * s_pad[:, :, 2:-2,4:  ]
        s42 = s32 * s_pad[:, :, 4:  ,2:-2]

        # in between
        s01 = (s11 + s12) * 0.5 * s_pad[:, :,  :-4,1:-3]
        s03 = (s12 + s13) * 0.5 * s_pad[:, :,  :-4,3:-1]
        s10 = (s11 + s21) * 0.5 * s_pad[:, :, 1:-3, :-4]
        s30 = (s21 + s31) * 0.5 * s_pad[:, :, 3:-1, :-4]
        s14 = (s13 + s23) * 0.5 * s_pad[:, :, 1:-3,4:  ]
        s34 = (s23 + s33) * 0.5 * s_pad[:, :, 3:-1,4:  ]
        s41 = (s31 + s32) * 0.5 * s_pad[:, :, 4:  ,1:-3]
        s43 = (s32 + s33) * 0.5 * s_pad[:, :, 4:  ,3:-1]

        return torch.stack((s00, s01, s02, s03, s04, s10, s11, s12, s13, s14, s20, s21, torch.ones_like(s), s23, s24, s30, s31, s32, s33, s34, s40, s41, s42, s43, s44), dim = 2)

    def visualize_weights(self, rows, cols, col):
        w_conv_d, w_prop_s, w_pow_s, w_conv_s, w_channel_s = self.prepare_weights()

        w_conv_d = rearrange(w_conv_d, 'o i (h w) -> o i h w', h=self.kernel_size).detach()
        w_spatial_d = reduce(w_conv_d, 'o i h w -> i h w', 'sum')
        w_channel_d = reduce(w_conv_d, 'o i h w -> o i', 'sum')

        w_spatial_s = reduce(w_conv_s, 'o i h w -> i h w', 'sum').detach()
        if self.n_in == self.n_out:
            w_channel_s = reduce(w_conv_s, 'o i h w -> o i', 'sum').detach()

        idx = col

        ax = plt.subplot(rows, cols, idx)
        plt.imshow(w_pow_s[:,None].cpu().detach())
        for (j,i),label in np.ndenumerate(w_pow_s[:,None].cpu().detach()):
            ax.text(i,j,'{:.02f}'.format(label),ha='center',va='center', fontsize=min(10,80 / rows), color='tomato')
        if self.n_in > 1:
        	plt.xlabel('c', fontsize=min(10,100 / rows))
        plt.xticks([])
        plt.yticks([])
        idx+=cols

        ax = plt.subplot(rows, cols, idx)
        plt.imshow(w_prop_s[:,None].cpu().detach())
        for (j,i),label in np.ndenumerate(w_prop_s[:,None].cpu().detach()):
            ax.text(i,j,'{:.02f}'.format(label),ha='center',va='center', fontsize=min(10,80 / rows), color='tomato')
        plt.xticks([])
        plt.yticks([])
        if self.n_in > 1:
            plt.ylabel('c', fontsize=min(10,100 / rows))
        ax.tick_params(length=0)
        idx+=cols

        for c in range(self.n_in):
            if self.kernel_size > 1:
                ax = plt.subplot(rows, cols, idx)
                plt.imshow(w_spatial_s[c, :, :].cpu())
                plt.xlabel('w', fontsize=min(10,100 / rows))
                plt.ylabel('h', fontsize=min(10,100 / rows))
                plt.xticks([])
                plt.yticks([])
            idx+=cols
        idx+=max(0,self.n_out - self.n_in) * cols

        if self.n_in > 1:
            ax = plt.subplot(rows, cols, idx)
            plt.xlabel('in', fontsize=min(10,100 / rows))
            plt.ylabel('out', fontsize=min(10,100 / rows))
            plt.imshow(w_channel_s.cpu().detach())
            plt.xticks([])
            plt.yticks([])
            idx+=cols
        elif self.n_out > 1:
            idx+=cols

        idx+=cols

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
            plt.imshow(w_channel_d[:, :].cpu())
            plt.xticks([])
            plt.yticks([])
            idx+=cols
        elif self.n_out > 1:
            idx+=cols