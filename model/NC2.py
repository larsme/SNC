import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import importlib

from model.NC import NC
from src.KittiDepthDataset import project_velo_grid_to_image

class NC2(torch.nn.Module):

    def __init__(self, params):
        super().__init__()
        
        self.stage1 = getattr(importlib.import_module('model.' + params['inner_model']['model']), params['inner_model']['model'])(params['inner_model'])
        self.stage2 = getattr(importlib.import_module('model.' + params['inner_model']['model']), params['inner_model']['model'])(params['inner_model'])
        
        self.stage1.load_state_dict(torch.load(params['inner_model_path']))
        for param in self.stage1.parameters():
            param.requires_grad = False

        self.lidar_padding = self.stage1.lidar_padding
        self.sum_pad_d = self.stage1.sum_pad_d
        
        self.w_pow_d = torch.nn.Parameter(torch.rand((1, 1, 1, 1)))

        self.eps = 1e-20

    def prepare_weights(self):
        w_pow_d = F.softplus(self.w_pow_d)

        return w_pow_d, 


    def prep_eval(self):
        self.stage1.prep_eval()
        self.stage2.prep_eval()

        self.weights = self.prepare_weights()

    def forward(self, d, cd=None, **args):
        
        if self.training:
            w_pow_d,  = self.prepare_weights()
        else:
            w_pow_d, = self.weights
            
        with torch.no_grad():
            if cd is None:
                cd = (d > 0).float()
            outs1 = self.stage1(d, cd,**args)
        cd = cd * torch.pow((outs1['d'] / (d + self.eps)).clamp_max(1), w_pow_d)
        outs2 = self.stage2(d, cd, **args)
        return outs2

    def streaming_perception(self, d, cd=None, **args):
        
        if self.training:
            w_pow_d, = self.prepare_weights()
        else:
            w_pow_d, = self.weights

        if cd is None:
            cd = (d > 0).float()
        outs1 = self.stage1.streaming_perception(d, cd, **args)
        
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        plt.tight_layout()

        cd = cd * torch.pow(torch.minimum(outs1['d'], d) / (d + self.eps), w_pow_d)

        return self.stage2.streaming_perception(d, cd, **args)


    def visualize_weights(self, img_file_path=''):
        w_pow_d, w_pow_e = self.prepare_weights()

        self.stage1.visualize_weights()
        plt.figure()
        plt.table(rowLabels = ['w_pow_d'],
                  cellText = [['{:.2f}'.format(w_pow_d.squeeze().item())]],
                  loc = 'center')
        plt.axis('off')
        plt.tight_layout()
        self.stage2.visualize_weights()