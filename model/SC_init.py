import torch
import torch.nn.functional as F
from einops import rearrange, reduce
import matplotlib.pyplot as plt
import numpy as np

class SC_init(torch.nn.Module):
    def __init__(self, init_method='k', n_colors=1, kernel_size=1, max_pool_size=1, symmetric=True, no_prop=False, **kwargs):
        super().__init__()

        # paddings assume an odd kernel size and input shape = output shape
        self.eps = 1e-20
        self.n_c = n_colors
        self.kernel_size = 5
        self.padding = kernel_size // 2
        self.symmentric = symmetric

        # Define Parameters
        
        if self.symmentric:
            self.w_conv_0 = torch.nn.Parameter(data=torch.ones(n_colors, 1, kernel_size, kernel_size))
            self.w_conv_1 = torch.nn.Parameter(data=torch.ones(n_colors, 1, kernel_size, (kernel_size + 1) // 2))
            self.w_conv_3 = torch.nn.Parameter(data=torch.ones(n_colors, 1, kernel_size, kernel_size))
        
            # Init Parameters
            if 'x' in init_method:  # Xavier
                torch.nn.init.xavier_uniform_(self.w_conv_0)
                torch.nn.init.xavier_uniform_(self.w_conv_1)
                torch.nn.init.xavier_uniform_(self.w_conv_3)
            elif 'k' in init_method: # Kaiming
                torch.nn.init.kaiming_uniform_(self.w_conv_0)
                torch.nn.init.kaiming_uniform_(self.w_conv_1)
                torch.nn.init.kaiming_uniform_(self.w_conv_3)
        else:
            self.w_conv = torch.nn.Parameter(data=torch.ones(4*n_colors,1, kernel_size, kernel_size))

            # Init Parameters
            if 'x' in init_method:  # Xavier
                torch.nn.init.xavier_uniform_(self.w_conv)
            elif 'k' in init_method: # Kaiming
                torch.nn.init.kaiming_uniform_(self.w_conv)

    def prepare_weights(self):        
        if self.symmentric:
            #0 => /
            #1 => -
            #2 => \
            #3 => |
            # 1, 3 are symmetric; 2 is a mirror of 0
            w_conv = torch.cat((self.w_conv_0,
                                torch.cat((self.w_conv_1,self. w_conv_1[...,:-1].flip(dims=(3,))), dim=3),
                                self.w_conv_0.flip(dims=(3,)),
                                self.w_conv_3), dim=0)
        else:
            w_conv = self.w_conv     
        
        w_conv = w_conv - w_conv.mean(dim=(2,3), keepdim=True)

        return w_conv,

    def prep_eval(self):
        self.weights = self.prepare_weights()

    def forward(self, color):
        # cd = confidence over depth
        # e = directed smoothness;  dim 2 corresponds to edge directions: /, -, \, |
        # ce = confidence over e

        if self.training:
            w_conv, = self.prepare_weights()
        else:
            w_conv, = self.weights

        e = torch.exp(-reduce(F.conv2d(color, w_conv, padding=self.padding, groups=self.n_c).abs(),'b (d c) h w -> b 1 d h w', d=4, reduction='sum'))
        ce = torch.ones_like(e)

        return e, ce

    def visualize_weights(self, rows, cols, col):

        if self.training:
            w_conv, = self.prepare_weights()
        else:
            w_conv, = self.weights

        w_conv = rearrange(w_conv, '(d c) 1 h w -> d c h w', d=4)
        idx = col

        for d in range(4):    
            for c in range(self.n_c):
                ax = plt.subplot(rows, cols,idx)
                plt.imshow(w_conv[d, c, :,:])
                plt.xticks([])
                plt.yticks([])
                plt.xlabel(['/','-','\\','|'][d] + (' {}'.format(c) if self.n_c > 1 else ''))
                idx+=cols