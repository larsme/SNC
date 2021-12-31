import torch
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

from model.NC_conv import NC_conv as Conv
from model.NC_conv import NC_disp_conv as Pool
from model.NC_unpool import NC_unpool as Unpool
from model.NC_bias import NC_bias as Bias

class NC(torch.nn.Module):

    def __init__(self, params):
        super().__init__()
        
        n_c = params['n_channels']
        self.n_c = n_c
        self.n_stages = params['n_stages']
        self.lidar_padding = params['lidar_padding']
        conv_k = params['conv_kernel_size']
        self.sum_pad_d = (conv_k-1)//2
        for l in range(self.n_stages - 1):
            self.sum_pad_d = (conv_k-1)//2 + 1 + 2 * self.sum_pad_d + 1 + (conv_k-1)//2
        self.sum_pad_d = 2 ** (self.n_stages - 2) * int(np.ceil((1 + self.sum_pad_d) / 2 ** (self.n_stages - 2)))
        self.weight_disp = params['weight_disp']
        self.eps=1e-20
        
        convs = [Conv(n_in=1, n_out=n_c, kernel_size=conv_k, **params)]
        unpools = []
        for l in range(self.n_stages - 1):
            convs.append(Conv(n_in=n_c, n_out=n_c, kernel_size=conv_k, **params))
            unpools.append(Unpool(n_c=n_c, kernel_size=4, stride=2, **params))
        if self.n_stages > 1:
            self.pool = (Pool(n_in=n_c, n_out=n_c, kernel_size=4, stride=2, **params) if params['pool_disp'] 
                        else Conv(n_in=n_c, n_out=n_c, kernel_size=4, stride=2, **params))
            self.unpools = torch.nn.ModuleList(unpools)
            convs.append(Conv(n_in=n_c, n_out=1, kernel_size=conv_k, **params))
        self.convs = torch.nn.ModuleList(convs)
        self.bias = Bias(params['use_bias'])

    def prep_eval(self):
        for conv in self.convs:
            conv.prep_eval()
        if self.n_stages > 1:
            self.pool.prep_eval()
            for unpool in self.unpools:
                unpool.prep_eval()
        self.bias.prep_eval()

    def forward(self, d, cd=None, **args):
        if self.weight_disp:
            if cd is None:
                dcd = (d > 0).float()
            else:
                dcd = cd
            cd = dcd / (d+self.eps)
        else:
            if cd is None:
                cd = (d > 0).float()
                dcd = d
            else:
                dcd = d * cd
        x = torch.cat((dcd, cd), dim=0)
        
        # Encoder
        x = self.convs[0](x)
        skip = []
        for stage in range(self.n_stages - 1):
            skip.append(x)
            x = self.pool(x)
            x = self.convs[1](x)

        # Decoder
        for stage in range(self.n_stages - 1):
            x_skip = skip[-1]
            del skip[-1]
            x = self.unpools[stage](x_skip=x_skip, x_pool=x)
            x = self.convs[2 + stage](x)

        if self.lidar_padding:
            x = x[..., self.sum_pad_d:-self.sum_pad_d]

        d, cd = self.bias(x)

        return {'d': d, 'cd': cd}

    def streaming_perception(self, d, cd=None, **args):

        lower_quantile = np.quantile(d[d > 0].numpy(), 0.05)
        upper_quantile = np.quantile(d[d > 0].numpy(), 0.95)
        cmap = plt.cm.get_cmap('nipy_spectral', 256)
        cmap = np.ndarray.astype(np.array([cmap(i) for i in range(256)])[:, :3] * 255, np.uint8)
        plt.figure()

        rows = self.n_stages
        plt.subplot(rows, 4, 1).title.set_text('encoder depth')
        plt.subplot(rows, 4, 2).title.set_text('encoder conf')
        plt.subplot(rows, 4, 3).title.set_text('decoder depth')
        plt.subplot(rows, 4, 4).title.set_text('decoder conf')

        def plot(x, row):
            idx = 2 * (row // rows) + 4 * (row % rows)
            
            dcd = x[:x.shape[0] // 2,:,:,:].numpy().squeeze()
            cd = x[x.shape[0] // 2:,:,:,:].numpy().squeeze()
            d = dcd / (cd + 1e-20)
            
            plt.subplot(rows, 4, idx + 1)
            sparse_depth_img = cmap[np.ndarray.astype(np.interp(d, (lower_quantile, upper_quantile), (0, 255)), np.int_),:]
            sparse_depth_img[cd == 0,:] = 128
            sparse_depth_img = Image.fromarray(sparse_depth_img)
            plt.imshow(sparse_depth_img)
            plt.axis('off')

            plt.subplot(rows, 4, idx + 2)
            c_img = cmap[np.ndarray.astype(np.interp(cd / np.max(cd), (0, 1), (0, 255)), np.int_), :]
            c_img[cd == 0,:] = 128
            c_img = Image.fromarray(c_img)
            plt.imshow(c_img)
            plt.axis('off')

        if cd is None:
            dcd = d
            cd = (dcd > 0).float()
        else:
            dcd = d * cd
        x = torch.cat((dcd, cd), dim=0)

                
        # Encoder
        row = 0
        plot(x, row)
        row+=1
        x = self.convs[0](x)
        skip = []
        for stage in range(self.n_stages - 1):
            skip.append(x)
            x = self.pool(x)
            plot(x, row)
            row+=1
            x = self.convs[1](x)

        # Decoder
        for stage in range(self.n_stages - 1):
            x_skip = skip[-1]
            del skip[-1]
            x = self.unpools[stage](x_skip=x_skip, x_pool=x)
            plot(x, row)
            row+=1
            x = self.convs[2 + stage](x)

        if self.lidar_padding:
            x = x[..., self.sum_pad_d:-self.sum_pad_d]
        d, cd = self.bias(dcd=x[:cd.shape[0],:,:,:], cd=x[cd.shape[0]:,:,:,:])
        
        plot(torch.cat((d*cd, cd), dim=0), row)  

        return {'d': d, 'cd': cd}

    def visualize_weights(self, img_file_path=''):
        plt.figure()

        cols = 11 if self.bias.use_bias else 10
        rows = self.n_c + 1 + int(self.n_c != 1) + int(self.bias.use_bias) + 1
        col = 1
        idx = 1            
        for c in range(self.n_c):
            ax = plt.subplot(rows, cols, idx)
            plt.axis('off')
            ax.text(0,0.5,'w spatial d', clip_on=True, fontsize=12)
            idx+=cols

        if self.n_c > 1:
            ax = plt.subplot(rows, cols, idx)
            plt.axis('off')
            ax.text(0,0.5,'w channel d', clip_on=True, fontsize=12)
            idx+=cols

        ax = plt.subplot(rows, cols, idx)
        plt.axis('off')
        ax.text(0,0.5,'w skip d', clip_on=True, fontsize=12)
        idx+=cols
        col+=1

        plt.subplot(rows, cols, col).title.set_text('initial\nconv')
        self.convs[0].visualize_weights(rows, cols, col)
        col+=1

        plt.subplot(rows, cols, col).title.set_text('encoder\npooling')
        if self.n_stages > 1:
            self.pool.visualize_weights(rows, cols, col)
        col+=1

        plt.subplot(rows, cols, col).title.set_text('encoder\nconvs')
        self.convs[1].visualize_weights(rows, cols, col)
        col+=1
                
        if self.n_stages > 1:
            for i in range(len(self.unpools)):
                plt.subplot(rows, cols, col).title.set_text('decoder\ndeconv {}'.format(i))
                self.unpools[i].visualize_weights(rows, cols, col)
                col+=1

                plt.subplot(rows, cols, col).title.set_text('decoder\nconv {}'.format(i))
                self.convs[i + 2].visualize_weights(rows, cols, col)
                col+=1

        if self.bias.use_bias:
            plt.subplot(rows, cols, col).title.set_text('bias')
            self.bias.visualize_weights(rows, cols, col)
            col+=1
        plt.subplots_adjust(hspace=0.5)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        if img_file_path != '':
            plt.savefig(img_file_path, bbox_inches='tight')