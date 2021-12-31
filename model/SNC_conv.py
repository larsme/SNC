import torch
import torch.nn.functional as F
from einops import rearrange, reduce
import matplotlib.pyplot as plt
import numpy as np

from model.retrieve_indices import retrieve_indices

class SNC_conv(torch.nn.Module):
    def __init__(self, init_method='k', n_in=1, n_out=1, kernel_size=1, max_pool_size=1, symmetric=True, no_prop=False, **kwargs):
        super(SNC_conv, self).__init__()

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
        self.no_prop = no_prop

        # Define Parameters
        self.w_channel_d = torch.nn.Parameter(data=torch.ones(n_out, n_in), requires_grad=n_in > 1)
        self.w_spatial_d = torch.nn.Parameter(data=torch.ones(n_in, kernel_size, (kernel_size + 1) // 2 if symmetric else kernel_size), requires_grad=self.kernel_size > 1)
        self.w_channel_e = torch.nn.Parameter(data=torch.ones(n_out, n_in), requires_grad=n_in > 1)

        if symmetric:
            self.w_spatial_e_0 = torch.nn.Parameter(data=torch.ones(n_in, kernel_size, kernel_size), requires_grad=self.kernel_size > 1)
            self.w_spatial_e_1 = torch.nn.Parameter(data=torch.ones(n_in, kernel_size, (kernel_size + 1) // 2), requires_grad=self.kernel_size > 1)
            self.w_spatial_e_3 = torch.nn.Parameter(data=torch.ones(n_in, kernel_size, (kernel_size + 1) // 2), requires_grad=self.kernel_size > 1)
            self.w_dir_e = torch.nn.Parameter(data=torch.ones(n_in, 10))
            self.w_pow_e = torch.nn.Parameter(data=torch.ones(n_in, 5))
            self.w_prop_e = torch.nn.Parameter(data=torch.ones(1, n_in, 3, 1, 1), requires_grad=not no_prop)
        else:
            self.w_spatial_e = torch.nn.Parameter(data=torch.ones(n_in, 4, kernel_size, kernel_size), requires_grad=self.kernel_size > 1)
            self.w_dir_e = torch.nn.Parameter(data=torch.ones(4, n_in, 4))
            self.w_pow_e = torch.nn.Parameter(data=torch.ones(n_in, 2, 4))
            self.w_prop_e = torch.nn.Parameter(data=torch.ones(1, n_in, 4, 1, 1), requires_grad=not no_prop)

        self.unfold = torch.nn.Unfold(kernel_size=kernel_size,dilation=1,padding=self.padding,stride=1)

        # Init Parameters
        if 'x' in init_method:  # Xavier
            torch.nn.init.xavier_uniform_(self.w_channel_d)
            torch.nn.init.xavier_uniform_(self.w_channel_e)
            torch.nn.init.xavier_uniform_(self.w_dir_e)
            torch.nn.init.xavier_uniform_(self.w_spatial_d)
            torch.nn.init.xavier_uniform_(self.w_prop_e)
            torch.nn.init.xavier_uniform_(self.w_pow_e)
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
            torch.nn.init.kaiming_uniform_(self.w_prop_e)
            torch.nn.init.kaiming_uniform_(self.w_pow_e)
            if symmetric:
                torch.nn.init.kaiming_uniform_(self.w_spatial_e_0)
                torch.nn.init.kaiming_uniform_(self.w_spatial_e_1)
                torch.nn.init.kaiming_uniform_(self.w_spatial_e_3)
            else:
                torch.nn.init.kaiming_uniform_(self.w_spatial_e)

    def prepare_weights(self):
        # enforce limits
        w_channel_d = F.softplus(self.w_channel_d)
        w_spatial_d = F.softplus(self.w_spatial_d)
        w_prop_e = torch.sigmoid(self.w_prop_e)
        w_pow_e = F.softplus(self.w_pow_e)
        w_dir_e = F.softplus(self.w_dir_e)
        w_channel_e = F.softplus(self.w_channel_e)
        if self.symmentric:
            w_spatial_e_0 = F.softplus(self.w_spatial_e_0)
            w_spatial_e_1 = F.softplus(self.w_spatial_e_1)
            w_spatial_e_3 = F.softplus(self.w_spatial_e_3)
        else:
            w_spatial_e = F.softplus(self.w_spatial_e)

        if self.no_prop:
            w_prop_e = torch.zeros_like(w_prop_e)

        # enforce symmetry by weight sharing
        if self.symmentric:
            w_spatial_d = torch.cat((w_spatial_d,w_spatial_d[:,:,:self.kernel_size // 2].flip(dims=(2,))), dim=2)
            #0 => /
            #1 => -
            #2 => \
            #3 => |
            # 1, 3 are symmetric; 2 is a mirror of 0
            w_spatial_e = torch.stack((w_spatial_e_0,
                                    torch.cat((w_spatial_e_1, w_spatial_e_1[:,:,:-1].flip(dims=(2,))), dim=2),
                                    w_spatial_e_0.flip(dims=(2,)),
                                    torch.cat((w_spatial_e_3, w_spatial_e_3[:,:,:-1].flip(dims=(2,))), dim=2)), dim=1)
            # connect directions to each other; with connections from or to 0 and 2 sharing the same weights
            w_dir_e = w_dir_e.unbind(1)
            w_dir_e = torch.stack((torch.stack((w_dir_e[0], w_dir_e[1], w_dir_e[2], w_dir_e[3]), dim = 1),
                                   torch.stack((w_dir_e[4], w_dir_e[5], w_dir_e[4], w_dir_e[6]), dim = 1),
                                   torch.stack((w_dir_e[2], w_dir_e[1], w_dir_e[0], w_dir_e[3]), dim = 1),
                                   torch.stack((w_dir_e[7], w_dir_e[8], w_dir_e[7], w_dir_e[9]), dim = 1)),
                                   dim=0)
            # 1 w_pow_e per side; 0 and 2 are mirrors; 3 has symmetric sides
            w_pow_e = w_pow_e.unbind(1)
            w_pow_e = torch.stack((torch.stack((w_pow_e[0], w_pow_e[1]), dim=1),
                                   torch.stack((w_pow_e[2], w_pow_e[3]), dim=1),
                                   torch.stack((w_pow_e[1], w_pow_e[0]), dim=1),
                                   torch.stack((w_pow_e[4], w_pow_e[4]), dim=1)), dim=2)
            w_prop_e = torch.cat((w_prop_e[:,:,1,None,:,:], w_prop_e), dim=2)

        # normalize by output channel
        # technically not needed for d, but here for consistency
        w_channel_d = w_channel_d / reduce(w_channel_d,'o i -> o 1','sum')
        w_spatial_d = w_spatial_d / reduce(w_spatial_d,'i h w -> i 1 1','sum')
        w_channel_e = w_channel_e / reduce(w_channel_e,'o i -> o 1','sum')
        w_dir_e = w_dir_e / reduce(w_dir_e,    'd2 i d1 -> d2 i 1','sum')
        w_spatial_e = w_spatial_e / reduce(w_spatial_e,'i d h w -> i d 1 1','sum')

        # combine seperable convolution for speed
        w_conv_d = rearrange(torch.einsum('o i, i h w -> o i h w', w_channel_d, w_spatial_d), 'o i h w -> o i (h w)')
        if self.n_in >= self.n_out:
            w_conv_e = rearrange(torch.einsum('o i, p i d, i d h w -> o p i d h w', w_channel_e, w_dir_e, w_spatial_e), 'o d2 i d1 h w -> (o d2) (i d1) h w')
            return w_conv_d, w_prop_e, w_pow_e, w_conv_e, None
        else:
            w_conv_e = rearrange(torch.einsum('p i d, i d h w -> p i d h w', w_dir_e, w_spatial_e), 'd2 i d1 h w -> d2 (i d1) h w')
            return w_conv_d, w_prop_e, w_pow_e, w_conv_e, w_channel_e

    def prep_eval(self):
        self.weights = self.prepare_weights()

    def forward(self, x, ece, ce):
        # x = d*cd, cd
        # d = depth
        # cd = confidence over depth
        # ece = directed smoothness * ce;  dim 2 corresponds to edge directions: /, -, \, |
        # ce = confidence over directed smoothness

        if self.training:
            w_conv_d, w_prop_e, w_pow_e, w_conv_e, w_channel_e = self.prepare_weights()
        else:
            w_conv_d, w_prop_e, w_pow_e, w_conv_e, w_channel_e = self.weights

        # calculate e from d
        # e_new = d_side_a / d_side_b
        # ce_new = cd_side_a * cd_side_b
        # using an argmin to decide which side is which
        # for kernel_size == 5, side_a corresponds to a 3x3 argmin(cd/d) and side_b to a 3x3 argmax(dcd)
        dcd, cd = x.split(ce.shape[0], 0)
        d_pad, cd_pad = F.pad(dcd.detach() / (cd.detach() + self.eps), (1,1,1,1)), F.pad(cd.detach(), (1,1,1,1))
        if self.max_pool_size == 3:
            e_new = torch.min(torch.stack((d_pad[:,:, :-2, :-2] / (d_pad[:,:,2:  ,2:  ] + self.eps),
                                           d_pad[:,:, :-2,1:-1] / (d_pad[:,:,2:  ,1:-1] + self.eps),
                                           d_pad[:,:, :-2,2:  ] / (d_pad[:,:,2:  , :-2] + self.eps),
                                           d_pad[:,:,1:-1,2:  ] / (d_pad[:,:,1:-1, :-2] + self.eps)), dim=2).clamp(self.eps,1).pow(w_pow_e[None,:,0,:,None, None]),
                              torch.stack((d_pad[:,:,2:  ,2:  ] / (d_pad[:,:, :-2, :-2] + self.eps),
                                           d_pad[:,:,2:  ,1:-1] / (d_pad[:,:, :-2,1:-1] + self.eps),
                                           d_pad[:,:,2:  , :-2] / (d_pad[:,:, :-2,2:  ] + self.eps),
                                           d_pad[:,:,1:-1, :-2] / (d_pad[:,:,1:-1,2:  ] + self.eps)), dim=2).clamp(self.eps,1).pow(w_pow_e[None,:,1,:,None, None])).clamp(0,1)

            ce_new = torch.stack((cd_pad[:,:, :-2, :-2] * cd_pad[:,:,2:  ,2:  ],
                                  cd_pad[:,:, :-2,1:-1] * cd_pad[:,:,2:  ,1:-1],
                                  cd_pad[:,:, :-2,2:  ] * cd_pad[:,:,2:  , :-2],
                                  cd_pad[:,:,1:-1,2:  ] * cd_pad[:,:,1:-1, :-2]), dim = 2)
        else:
            # padding = 2 the next step needs a padding of one as well
            _, j_max = F.max_pool2d(d_pad * cd_pad, kernel_size=3, stride=1, return_indices=True, padding=1)
            _, j_min = F.max_pool2d(cd_pad / (d_pad + self.eps), kernel_size=3, stride=1, return_indices=True, padding=1)
            d_min, d_max = retrieve_indices(d_pad, j_min), retrieve_indices(d_pad, j_max)
            cd_min, cd_max = retrieve_indices(cd_pad, j_min), retrieve_indices(cd_pad, j_max)

            # d_min on one side divided by d_max on the other = low value => edge in the middle
            # clamp to 1 is needed to prevent min > max, which can happen with confidence weighting because they originate in different regions
            e_new = torch.stack((torch.stack((d_min[:,:, :-2, :-2] / (d_max[:,:,2:  ,2:  ] + self.eps), d_min[:,:,2:  ,2:  ] / (d_max[:,:, :-2, :-2] + self.eps)), dim=2),
                                 torch.stack((d_min[:,:, :-2,1:-1] / (d_max[:,:,2:  ,1:-1] + self.eps), d_min[:,:,2:  ,1:-1] / (d_max[:,:, :-2,1:-1] + self.eps)), dim=2),
                                 torch.stack((d_min[:,:, :-2,2:  ] / (d_max[:,:,2:  , :-2] + self.eps), d_min[:,:,2:  , :-2] / (d_max[:,:, :-2,2:  ] + self.eps)), dim=2),
                                 torch.stack((d_min[:,:,1:-1,2:  ] / (d_max[:,:,1:-1, :-2] + self.eps), d_min[:,:,1:-1, :-2] / (d_max[:,:,1:-1,2:  ] + self.eps)), dim=2)),
                                dim=3).clamp(self.eps,1).pow(w_pow_e[None,:,:,:,None,None])

            ce_new = torch.stack((torch.stack((cd_min[:,:, :-2, :-2] * cd_max[:,:,2:  ,2:  ], cd_min[:,:,2:  ,2:  ] * cd_max[:,:, :-2, :-2]), dim=2),
                                  torch.stack((cd_min[:,:, :-2,1:-1] * cd_max[:,:,2:  ,1:-1], cd_min[:,:,2:  ,1:-1] * cd_max[:,:, :-2,1:-1]), dim=2),
                                  torch.stack((cd_min[:,:, :-2,2:  ] * cd_max[:,:,2:  , :-2], cd_min[:,:,2:  , :-2] * cd_max[:,:, :-2,2:  ]), dim=2),
                                  torch.stack((cd_min[:,:,1:-1,2:  ] * cd_max[:,:,1:-1, :-2], cd_min[:,:,1:-1, :-2] * cd_max[:,:,1:-1,2:  ]), dim=2)), dim=3)

            # dim 2 contains d_min(side1) / d_max(side2) and d_min(side2) / d_max(side1)
            # take the confidence weighted min of both and gather the respective items
            j_min = torch.argmax(ce_new / e_new, dim=2, keepdim=True)
            e_new = e_new.gather(index=j_min, dim=2).squeeze(2)
            ce_new = ce_new.gather(index=j_min, dim=2).squeeze(2)

        # propagate smoothness
        if self.n_in >= self.n_out:
            ce = rearrange(F.conv2d(rearrange(ce * w_prop_e + ce_new * (1 - w_prop_e), 'b c d h w -> b (c d) h w'), w_conv_e, padding=self.padding), 'b (c d) h w -> b c d h w', d=4)
            ece = rearrange(F.conv2d(rearrange(ece * w_prop_e + e_new * ce_new * (1 - w_prop_e), 'b c d h w -> b (c d) h w'), w_conv_e, padding=self.padding), 'b (c d) h w -> b c d h w', d=4)
            e = ece / (ce + self.eps)
        else:
            ce = rearrange(F.conv2d(rearrange(ce * w_prop_e + ce_new * (1 - w_prop_e), 'b c d h w -> b (c d) h w'), w_conv_e, padding=self.padding, groups=self.n_in), 'b (c d) h w -> b c d h w', d=4)
            ece = rearrange(F.conv2d(rearrange(ece * w_prop_e + e_new * ce_new * (1 - w_prop_e), 'b c d h w -> b (c d) h w'), w_conv_e, padding=self.padding, groups=self.n_in), 'b (c d) h w -> b c d h w', d=4)
            e = ece / (ce + self.eps)
            ce = torch.einsum('o i, b i d h w -> b o d h w', w_channel_e, ce)
            ece = torch.einsum('o i, b i d h w -> b o d h w', w_channel_e, ece)

        # propagate depth
        if self.kernel_size == 3:
            # no penalty to stay in place
            # propagating from a location to a neighbour is penalized by either being an edge in the respective direction
            e_ext = F.pad(e, (1,1,1,1))
            e0, e1, e2, e3 = e.unbind(2)
            w_e = torch.stack((e0 * e_ext[:,:,0, :-2,:-2],  e1 * e_ext[:,:,1, :-2,1:-1],   e2 * e_ext[:,:,2, :-2,2:],
                               e3 * e_ext[:,:,3,1:-1,:-2],  torch.ones_like(e[:,:,1,:,:]), e3 * e_ext[:,:,3,1:-1,2:],
                               e2 * e_ext[:,:,2,2:  ,:-2],  e1 * e_ext[:,:,1,2:  ,1:-1],   e0 * e_ext[:,:,0,2:  ,2:]), dim=2)
        elif self.kernel_size == 5:
            w_e = self.w_e5(e)
        if self.n_out < self.n_in:
            w_e = w_e.expand(-1,cd.shape[1],-1,-1,-1)

        x = torch.cat((torch.einsum('b i k h w, o i k-> b o h w', self.unfold(dcd).view(w_e.shape) * w_e, w_conv_d),
                       torch.einsum('b i k h w, o i k-> b o h w', self.unfold(cd).view(w_e.shape) * w_e, w_conv_d)), 0) \
                          / (torch.einsum('b i k h w, o i k -> b o h w', w_e, w_conv_d) + self.eps).repeat(2,1,1,1)
        
        if ece.requires_grad:
            ece.register_hook(lambda grad: torch.clamp(grad, -1000, 1000))
            ce.register_hook(lambda grad: torch.clamp(grad, -1000, 1000))

        return x, ece, ce

    def w_e5(self, e):
        # no penalty to stay in place
        # propagating from a location to a neighbour is penalized by either being an edge in the respective direction
        # propagating to a neighbour's neighbour is penalized by all 3 being edges
        # in cases where multiple middle neighbours are equally suited, their paths are averaged
        # it should be noted that smoothness of the middle neighbour is only used once, as opposed to once for entering and once for exiting
        # when entering and exiting the middle neighbour in different directions, the one closer to the middle is used

        e_pad = F.pad(e, (2,2,2,2))

        # direct neighbours
        e = e.unbind(2)
        e11 = e[0] * e_pad[:, :, 0, 1:-3,1:-3]
        e12 = e[1] * e_pad[:, :, 1, 1:-3,2:-2]
        e13 = e[2] * e_pad[:, :, 2, 1:-3,3:-1]
        e21 = e[3] * e_pad[:, :, 3, 2:-2,1:-3]
        e23 = e[3] * e_pad[:, :, 3, 2:-2,3:-1]
        e31 = e[2] * e_pad[:, :, 2, 3:-1,1:-3]
        e32 = e[1] * e_pad[:, :, 1, 3:-1,2:-2]
        e33 = e[0] * e_pad[:, :, 0, 3:-1,3:-1]

        # diagonals
        e00 = e11 * e_pad[:, :, 0,  :-4, :-4]
        e04 = e13 * e_pad[:, :, 2,  :-4,4:  ]
        e40 = e31 * e_pad[:, :, 2, 4:  , :-4]
        e44 = e33 * e_pad[:, :, 0, 4:  ,4:  ]

        # center lines
        e02 = e12 * e_pad[:, :, 1,  :-4,2:-2]
        e20 = e21 * e_pad[:, :, 3, 2:-2, :-4]
        e24 = e23 * e_pad[:, :, 3, 2:-2,4:  ]
        e42 = e32 * e_pad[:, :, 1, 4:  ,2:-2]

        # in between
        # up, down
        e01 = (e11 * e_pad[:, :, 1, 0:-4,1:-3] + e12 * e_pad[:, :, 0, 0:-4,1:-3]) * 0.5
        e03 = (e13 * e_pad[:, :, 1, 0:-4,3:-1] + e12 * e_pad[:, :, 2, 0:-4,3:-1]) * 0.5
        e41 = (e31 * e_pad[:, :, 1, 4:  ,1:-3] + e32 * e_pad[:, :, 2, 4:  ,1:-3]) * 0.5
        e43 = (e33 * e_pad[:, :, 1, 4:  ,3:-1] + e32 * e_pad[:, :, 0, 4:  ,3:-1]) * 0.5
        # left, right
        e10 = (e11 * e_pad[:, :, 3, 1:-3,0:-4] + e21 * e_pad[:, :, 0, 1:-3,0:-4]) * 0.5
        e14 = (e13 * e_pad[:, :, 3, 1:-3,4:  ] + e23 * e_pad[:, :, 2, 1:-3,4:  ]) * 0.5
        e30 = (e31 * e_pad[:, :, 3, 3:-1,0:-4] + e21 * e_pad[:, :, 2, 3:-1,0:-4]) * 0.5
        e34 = (e33 * e_pad[:, :, 3, 3:-1,4:  ] + e23 * e_pad[:, :, 0, 3:-1,4:  ]) * 0.5

        return torch.stack((e00, e01, e02, e03, e04, e10, e11, e12, e13, e14, e20, e21, torch.ones_like(e00), e23, e24, e30, e31, e32, e33, e34, e40, e41, e42, e43, e44), dim = 2)

    def visualize_weights(self, rows, cols, col):
        w_conv_d, w_prop_e, w_pow_e, w_conv_e, w_channel_e = self.prepare_weights()

        w_conv_d = rearrange(w_conv_d, 'o i (h w) -> o i h w', h=self.kernel_size)
        w_spatial_d = reduce(w_conv_d, 'o i h w -> i h w', 'sum')
        w_channel_d = reduce(w_conv_d, 'o i h w -> o i', 'sum')

        w_conv_e = rearrange(w_conv_e, '(o d2) (i d1) h w -> o d2 i d1 h w', d1=4, d2=4)
        w_spatial_e = reduce(w_conv_e, 'o d2 i d1 h w -> i d1 h w', 'sum')
        if self.n_in >= self.n_out:
            w_channel_e = reduce(w_conv_e, 'o d2 i d1 h w -> o i', 'sum')
        w_dir_e = reduce(w_conv_e, 'o d2 i d1 h w -> d2 i d1', 'sum')

        idx = col

        for c in range(self.n_in):
            ax = plt.subplot(rows, cols, idx)
            plt.imshow(w_pow_e[c, :,:])
            for (j,i),label in np.ndenumerate(w_pow_e[c, :,:]):
                ax.text(i,j,'{:.02f}'.format(label),ha='center',va='center', fontsize=min(10,60 / rows), color='tomato')
            plt.xticks([0,1,2,3],[r'$^a / _b$',r'$\frac{a}{b}$',r'$_b \backslash ^a$','b|a'], fontsize=min(10,100 / rows))
            plt.yticks([0,1], [r'$\frac{a}{b}$',r'$\frac{b}{a}$'], fontsize=min(10,100 / rows))
            ax.tick_params(length=0)
            idx+=cols
        for c in range(self.n_in, self.n_out):
            idx+=cols

        ax = plt.subplot(rows, cols, idx)
        plt.imshow(w_prop_e[0,:,:, 0, 0])
        for (j,i),label in np.ndenumerate(w_prop_e[0,:,:, 0, 0]):
            ax.text(i,j,'{:.02f}'.format(label),ha='center',va='center', fontsize=min(10,80 / rows), color='tomato')
        plt.xticks([0,1,2,3],['/','-','\\','|'], fontsize=min(10,100 / rows))
        plt.yticks([])
        if self.n_in > 1:
            plt.ylabel('c', fontsize=min(10,100 / rows))
        ax.tick_params(length=0)
        idx+=cols

        for c in range(self.n_in):
            for d in range(4):
                if self.kernel_size > 1:
                    ax = plt.subplot(rows, cols, idx)
                    plt.imshow(w_spatial_e[c, d, :, :])
                    plt.xlabel('w', fontsize=min(10,100 / rows))
                    plt.ylabel('h', fontsize=min(10,100 / rows))
                    plt.xticks([])
                    plt.yticks([])
                idx+=cols
        idx+=max(0,self.n_out - self.n_in) * 4 * cols

        if self.n_in > 1:
            ax = plt.subplot(rows, cols, idx)
            plt.xlabel('in', fontsize=min(10,100 / rows))
            plt.ylabel('out', fontsize=min(10,100 / rows))
            plt.imshow(w_channel_e)
            plt.xticks([])
            plt.yticks([])
            idx+=cols
        elif self.n_out > 1:
            idx+=cols

        for c in range(self.n_in):
            ax = plt.subplot(rows, cols, idx)
            plt.imshow(w_dir_e[:,c, :])
            plt.xticks([0,1,2,3],['/','-','\\','|'], fontsize=min(10,100 / rows))
            plt.yticks([0,1,2,3],['/','-','\\','|'], fontsize=min(10,100 / rows))
            ax.tick_params(length=0)
            plt.ylabel('out', fontsize=min(10,100 / rows))
            idx+=cols
        idx+=max(0,self.n_out - self.n_in) * cols

        idx+=cols

        if self.kernel_size > 1:
            for c in range(self.n_in):
                ax = plt.subplot(rows, cols, idx)
                plt.imshow(w_spatial_d[c, :, :])
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
            plt.imshow(w_channel_d[:, :])
            plt.xticks([])
            plt.yticks([])
            idx+=cols
        elif self.n_out > 1:
            idx+=cols