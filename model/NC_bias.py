import torch
import torch.nn.functional as F
from einops import rearrange, reduce
import matplotlib.pyplot as plt

class NC_bias(torch.nn.Module):
    def __init__(self, use_bias):
        super().__init__()

        self.eps = 1e-20
        self.use_bias = use_bias
        if use_bias:
            self.offset = torch.nn.Parameter(data=torch.zeros(1))

        # example alternate bias implementation in case someone wants to try these; i didn't
        #self.ceratin_w = torch.nn.Parameter(data=torch.ones(1), requires_grad=use_bias)
        #self.uncertain_mean = torch.nn.Parameter(data=torch.ones(1)*40, requires_grad=use_bias)

        # manually initialized with constants since there is only ever one of these
        # default offset = no offset
        # uncertain_mean=40(m), because that's a depth value roughly in the middle of the possible range
        # ceratin_w=1, because it is a compromise between the default (no interference = inf=sigmoid^-1(1)) and a non-vanishing gradient

    def prepare_weights(self):
        # enforce limits
        #return torch.sigmoid(self.ceratin_w), self.uncertain_mean, self.offset
        return self.offset if self.use_bias else None

    def prep_eval(self):
        self.weights = self.prepare_weights()

    def forward(self, x):
        # x = dcd, cd
        # dcd = depth * cd
        # cd = confidence over depth

        B, C,H,W = x.shape
        dcd, cd = x.view(2, B//2, C,H,W).unbind(0)
        d = dcd / (cd + self.eps)

        if self.use_bias:
            if self.training:
                #ceratin_w, uncertain_mean,
                b2 = self.prepare_weights()
            else:
                #ceratin_w, uncertain_mean,
                b2 = self.weights

            d = d + b2

        return d, cd

    def visualize_weights(self, rows, cols, col):
        #ceratin_w, uncertain_mean,

        if not self.use_bias:
            return
        b2 = self.prepare_weights()

        idx = col

        #ax = plt.subplot(rows, cols, idx)
        #plt.imshow([[uncertain_mean.item()]])
        #plt.colorbar()
        #plt.xticks([])
        #plt.yticks([])
        #plt.xlabel('default depth')
        #idx+=cols
        #ax = plt.subplot(rows, cols, idx)
        #plt.imshow([[ceratin_w.item()]])
        #plt.colorbar()
        #plt.xticks([])
        #plt.yticks([])
        #plt.xlabel('c weight')
        #idx+=cols
        ax = plt.subplot(rows, cols, idx)
        plt.imshow([[b2.item()]])
        plt.colorbar()
        plt.xticks([])
        plt.yticks([])
        plt.xlabel('offset')
        idx+=cols