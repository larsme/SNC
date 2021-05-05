import torch
import torch.nn.functional as F
from einops import rearrange, reduce
import matplotlib.pyplot as plt
import numpy as np

class uSC_init(torch.nn.Module):
    def __init__(self, init_method='k', n_colors=1, kernel_size=1, max_pool_size=1, symmetric=True, no_prop=False, **kwargs):
        super().__init__()

        # paddings assume an odd kernel size and input shape = output shape
        self.eps = 1e-20
        self.n_c = n_colors
        self.kernel_size = 5
        self.padding = kernel_size // 2

        # Define Parameters
        self.w = torch.nn.Parameter(data=torch.ones(n_colors))
        
        # Init Parameters
        if 'x' in init_method:  # Xavier
            torch.nn.init.xavier_uniform_(self.w)
        elif 'k' in init_method: # Kaiming
            torch.nn.init.kaiming_uniform_(self.w)

    def prepare_weights(self):        
        w = -F.softplus(self.w)

        return w,

    def prep_eval(self):
        self.weights = self.prepare_weights()

    def forward(self, color):
        # cd = confidence over depth
        # e = directed smoothness;  dim 2 corresponds to edge directions: /, -, \, |
        # ce = confidence over e

        if self.training:
            w, = self.prepare_weights()
        else:
            w, = self.weights

        delta = torch.max_pool2d(color,self.kernel_size, 1, self.padding) + torch.max_pool2d(-color,self.kernel_size, 1, self.padding)
        s = torch.exp(torch.einsum('b c h w, c -> b 1 h w', delta, w))
        cs = torch.ones_like(e)

        return s, cs

    def visualize_weights(self, rows, cols, col):

        if self.training:
            w, = self.prepare_weights()
        else:
            w, = self.weights

        w = w.cpu().detach()[None,:]
        ax = plt.subplot(rows, cols,idx)
        plt.imshow(w)
        plt.xticks([])
        plt.yticks([])
        for (j,i),label in np.ndenumerate(w):
            ax.text(i,j,'{:.02f}'.format(label),ha='center',va='center', fontsize=min(10,60 / rows), color='tomato')
        plt.xlabel('color')
        idx+=cols