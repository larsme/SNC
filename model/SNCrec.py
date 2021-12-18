import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import importlib

from model.NC import NC
from src.KittiDepthDataset import project_velo_grid_to_image

class SNCrec(torch.nn.Module):

    def __init__(self, params):
        super().__init__()
        
        self.inner_model = getattr(importlib.import_module('model.' + params['inner_model']['model']), params['inner_model']['model'])(params['inner_model'])

        self.lidar_padding = self.inner_model.lidar_padding
        self.sum_pad_d = self.inner_model.sum_pad_d
        self.n_colors = self.inner_model.n_colors
        
        self.w_pow_d = torch.nn.Parameter(torch.rand((1, 1, 1, 1)))
        self.w_pow_e = torch.nn.Parameter(torch.rand((1, 1, 3, 1, 1)))

        self.eps = 1e-20

    def prepare_weights(self):
        w_pow_d = F.softplus(self.w_pow_d)
        w_pow_e = F.softplus(self.w_pow_e)

        w_pow_e = torch.cat((w_pow_e[:,:,1,None,...], w_pow_e), 2)

        return w_pow_d, w_pow_e


    def prep_eval(self):
        self.inner_model.prep_eval()
        self.weights = self.prepare_weights()

    def forward(self, d, cd=None, ece=None, ce=None, x=None, rgb=None, **args):
        
        if self.training:
            w_pow_d, w_pow_e = self.prepare_weights()
        else:
            w_pow_d, w_pow_e = self.weights

        with torch.no_grad():
            if cd is None:
                cd = (d > 0).float()
            outs1 = self.inner_model(d, cd, ece, ce, x, rgb, **args)
        cd = cd * torch.pow((outs1['d'] / (d + self.eps)).clamp_max(1), w_pow_d)
        e1 = outs1['e']
        ce1 = torch.pow(outs1['ce'], w_pow_e)
        outs2 = self.inner_model(d, cd,  e1 * ce1, ce1, x, rgb, **args)
        return outs2

    def streaming_perception(self, d, cd=None, ece=None, ce=None, x=None, rgb=None, **args):
        
        if self.training:
            w_pow_d, w_pow_e = self.prepare_weights()
        else:
            w_pow_d, w_pow_e = self.weights

        if cd is None:
            cd = (d > 0).float()
        outs1 = self.inner_model.streaming_perception(d, cd, ece, ce, x, rgb, **args)
        
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.tight_layout()

        cd = cd * torch.pow(torch.minimum(outs1['d'], d) / (d + self.eps), w_pow_d)
        e1 = outs1['e']
        ce1 = torch.pow(outs1['ce'], w_pow_e)
        return self.inner_model.streaming_perception(d, cd,  e1 * ce1, ce1, x, rgb, **args)


    def visualize_weights(self, img_file_path=''):
        w_pow_d, w_pow_e = self.prepare_weights()

        plt.figure()
        plt.table(rowLabels = ['w_pow_d', 'w_pow_e /', 'w_pow_e -', 'w_pow_e \\', 'w_pow_e |'],
                  cellText = [['{:.2f}'.format(x.squeeze().item())] for x in (w_pow_d, *w_pow_e.unbind(2))],
                  loc = 'center')
        plt.axis('off')
        plt.tight_layout()
        self.inner_model.visualize_weights()