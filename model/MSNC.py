import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import importlib

from model.NC import NC
from src.KittiDepthDataset import project_velo_grid_to_image

class MSNC(torch.nn.Module):

    def __init__(self, params):
        super().__init__()
        
        self.filter_occlusion = params['filter_occlusion']
        self.image_model = getattr(importlib.import_module('model.' + params['image_model']['model']),params['image_model']['model'])(params['image_model'])

        if params['lidar_model'] == 'None':
            self.lidar_model = None
            self.lidar_padding = False
        else:
            self.lidar_model = getattr(importlib.import_module('model.' + params['lidar_model']['model']),params['lidar_model']['model'])(params['lidar_model'])

            # padding = lidar model padding, since external functions accessing this value care about the input
            self.sum_pad_d = self.lidar_model.sum_pad_d
            self.lidar_padding = self.lidar_model.lidar_padding
            # tell the lidar model it is not using cropping either way so it doesn't crop the padded elements at the end
            # this is in case some are visible to the potentially padded image model; the projection has its own cropping if they are not
            self.lidar_model.lidar_padding = False 

    def prep_eval(self):
        if self.lidar_model is not None:
            self.lidar_model.prep_eval()
        self.image_model.prep_eval()

    def forward(self, d, dirs, offsets, im_shape, cd=None, r=None, **args):
        if self.lidar_model is None:
            if cd is None:
                cd = (d > 0).float()
        else:
            outs = self.lidar_model.forward(d=d, cd=cd,r=r, dirs=dirs, offsets=offsets)
            d, cd = outs['d'], outs['cd']

        d, cd, x = project_velo_grid_to_image(self.image_model.sum_pad_d if self.image_model.lidar_padding else 0, 
                                              d, cd, None, dirs, offsets, im_shape, self.filter_occlusion)        

        return self.image_model.forward(d=d, cd=cd)

    def streaming_perception(self, d, dirs, offsets, im_shape, cd=None, r=None, **args):
        if self.lidar_model is None:
            if cd is None:
                cd = (d > 0).float()
        else:
            outs = self.lidar_model.streaming_perception(d=d, cd=cd,r=r, dirs=dirs, offsets=offsets)
            d, cd = outs['d'], outs['cd']
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')
            plt.tight_layout()

        d, cd, x = project_velo_grid_to_image(self.image_model.sum_pad_d if self.image_model.lidar_padding else 0,
                                              d, cd, None, dirs, offsets, im_shape, self.filter_occlusion)  

        return self.image_model.streaming_perception(d=d, cd=cd)


    def visualize_weights(self, img_file_path=''):
        if self.lidar_model is not None:
            self.lidar_model.visualize_weights()
        self.image_model.visualize_weights()