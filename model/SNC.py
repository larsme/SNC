import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange, reduce

from model.SNC_conv import SNC_conv as Conv
from model.SNC_pool import SNC_pool as Pool
from model.SNC_unpool import SNC_unpool as Unpool
from model.NC_bias import NC_bias as Bias
from model.SC_init import SC_init

class SNC(torch.nn.Module):

    def __init__(self, params):
        super().__init__()

        self.eps = 1e-20
        n_c = params['n_channels']
        self.n_c = n_c
        self.n_colors = 0 + 3*int(params['rgb']) + int(params['reflectance'])
        self.n_stages = params['n_stages']
        self.lidar_padding = params['lidar_padding']
        self.sum_pad_d = 1
        for l in range(self.n_stages - 1):
            self.sum_pad_d = 1 + 1 + 2 * self.sum_pad_d + 1 + 1
        self.sum_pad_d = 2 ** (self.n_stages - 2) * int(np.ceil((1 + self.sum_pad_d) / 2 ** (self.n_stages - 2)))

        self.sc_init = SC_init(kernel_size=5, n_colors=self.n_colors, **params) if self.n_colors > 0 else None

        convs = [Conv(n_in=1, n_out=n_c, kernel_size=5, max_pool_size=5, **params)]
        unpools = []
        for l in range(self.n_stages - 1):
            convs.append(Conv(n_in=n_c, n_out=n_c, kernel_size=3, max_pool_size=3, **params))
            unpools.append(Unpool(n_c=n_c, kernel_size=4, stride=2, **params))
        convs.append(Conv(n_in=n_c, n_out=1, kernel_size=3, max_pool_size=3, **params))

        self.convs = torch.nn.ModuleList(convs)
        self.pool = Pool(n_c=n_c, kernel_size=4, stride=2, **params)
        self.unpools = torch.nn.ModuleList(unpools)
        self.bias = Bias(True) if params['use_bias'] else None

    def prep_eval(self):
        if self.sc_init is not None:
            self.sc_init.prep_eval()
        for conv in self.convs:
            conv.prep_eval()
        self.pool.prep_eval()
        for unpool in self.unpools:
            unpool.prep_eval()
        if self.bias is not None:
            self.bias.prep_eval()

    def forward(self, d, cd=None, ece=None, ce=None, x=None, rgb=None, **args):

        if self.sc_init is not None:
            if rgb is None:
                color = x
            else:
                color = rgb
            ece, ce = self.sc_init(color)
            if self.lidar_padding and rgb is not None:
                pad = (self.sum_pad_d, self.sum_pad_d, self.sum_pad_d, self.sum_pad_d)
                ece, ce = F.pad(ece, pad),  F.pad(ce, pad)
        else:
            if ece is None:
                ece = torch.ones_like(d)[:,:,None,:,:].expand(-1, -1,4,-1,-1)
            if ce is None:
                ce = torch.ones_like(ece)
        if cd is None:
            cd = (d > 0).float()
        x = torch.cat((d*cd, cd), dim=0)

        # Encoder
        x, ece, ce = self.convs[0](x, ece, ce)
        skip = []
        for stage in range(self.n_stages - 1):
            skip.append((x, ece, ce))
            x, ece, ce = self.pool(x, ece, ce)
            x, ece, ce = self.convs[1](x, ece, ce)

        # Decoder
        for stage in range(self.n_stages - 1):
            (x_skip, ece_skip, ce_skip) = skip[-1]
            del skip[-1]
            x, ece, ce = self.unpools[stage](x_skip=x_skip, ece_skip=ece_skip, ce_skip=ce_skip, x_pool=x, ece_pool=ece, ce_pool=ce)
            x, ece, ce = self.convs[2 + stage](x, ece, ce)

        if self.lidar_padding:
            x, ece, ce = x[..., self.sum_pad_d:-self.sum_pad_d], ece[..., self.sum_pad_d:-self.sum_pad_d],  ce[..., self.sum_pad_d:-self.sum_pad_d]

        if self.bias is not None:
            d, cd = self.bias(x.split(d.shape(0), 0))
        else:
            dcd, cd = x.split(d.shape[0], 0)
            d = dcd / (cd + self.eps)

        return {'d': d, 'cd': cd, 'e': ece / (ce + self.eps), 'ce': ce}

    def streaming_perception(self, d, cd=None, ece=None, ce=None, x=None, rgb=None, **args):

        lower_quantile = np.quantile(d[d > 0].cpu().numpy(), 0.05)
        upper_quantile = np.quantile(d[d > 0].cpu().numpy(), 0.95)
        cmap = plt.cm.get_cmap('nipy_spectral', 256)
        cmap = np.ndarray.astype(np.array([cmap(i) for i in range(256)])[:, :3] * 255, np.uint8)
        plt.figure()

        def plot(x, ece, ce, row):
            dcd, cd = x.split(ce.shape[0], 0)
            d = dcd / (cd + self.eps)
            rows = 2 * self.n_stages

            plt.subplot(rows, 4, 4 * row + 1)
            d, cd, s, cs = d.cpu().numpy().squeeze(), cd.cpu().numpy().squeeze(), reduce(ece / (ce + 1e-20), 'b c d h w -> h w', 'prod').cpu().numpy(),reduce(ce, 'b c d h w -> h w', 'mean').cpu().numpy()
            d_img = cmap[np.ndarray.astype(np.interp(d, (lower_quantile, upper_quantile), (0, 255)), np.int_),:]
            d_img[cd == 0,:] = 128
            plt.imshow(d_img)
            plt.axis('off')
            if row == 0:
                plt.gca().title.set_text('depth')

            plt.subplot(rows, 4, 4 * row + 2)
            c_img = cmap[np.ndarray.astype(np.interp(cd / np.max(cd), (0, 1), (0, 255)), np.int_), :]
            c_img[cd == 0,:] = 128
            plt.imshow(c_img)
            plt.axis('off')
            if row == 0:
                plt.gca().title.set_text('depth confidence')

            plt.subplot(rows, 4, 4 * row + 3)
            s_img = cmap[np.ndarray.astype(np.interp(s, (0, 1), (0, 255)), np.int_), :]
            s_img[cs == 0,:] = 128
            plt.imshow(s_img)
            plt.axis('off')
            if row == 0:
                plt.gca().title.set_text('smoothness')

            plt.subplot(rows, 4, 4 * row + 4)
            cs_img = cmap[np.ndarray.astype(np.interp(cs / np.max(cs), (0, 1), (0, 255)), np.int_),:]
            cs_img[cs == 0,:] = 128
            plt.imshow(cs_img)
            plt.axis('off')
            if row == 0:
                plt.gca().title.set_text('smoothness confidence')
            
        if self.sc_init is not None:
            if rgb is None:
                color = x
            else:
                color = rgb
            ece, ce = self.sc_init(color)
            if self.lidar_padding and rgb is not None:
                pad = (self.sum_pad_d, self.sum_pad_d, self.sum_pad_d, self.sum_pad_d)
                ece, ce = F.pad(ece, pad),  F.pad(ce, pad)
        else:
            if ece is None:
                ece = torch.ones_like(d)[:,:,None,:,:].expand(-1, -1,4,-1,-1)
            if ce is None:
                ce = torch.ones_like(ece)
        if cd is None:
            cd = (d > 0).float()
        x = torch.cat((d*cd, cd), dim=0)

                
        # Encoder
        row = 0
        plot(x, ece, ce, row)
        row+=1
        x, ece, ce = self.convs[0](x, ece, ce)

        skip = []
        for stage in range(self.n_stages - 1):
            skip.append((x, ece, ce))
            x, ece, ce = self.pool(x, ece, ce)
            plot(x, ece, ce, row)
            row+=1
            x, ece, ce = self.convs[1](x, ece, ce)

        # Decoder
        for stage in range(self.n_stages - 1):
            (x_skip, ece_skip, ce_skip) = skip[-1]
            del skip[-1]
            x, ece, ce = self.unpools[stage](x_skip=x_skip, ece_skip=ece_skip, ce_skip=ce_skip, x_pool=x, ece_pool=ece, ce_pool=ce)
            plot(x, ece, ce, row)
            row+=1
            x, ece, ce = self.convs[2 + stage](x, ece, ce)

        if self.lidar_padding:
            x, ece, ce = x[..., self.sum_pad_d:-self.sum_pad_d], ece[..., self.sum_pad_d:-self.sum_pad_d],  ce[..., self.sum_pad_d:-self.sum_pad_d]
            
        if self.bias is not None:
            d, cd = self.bias(x.split(d.shape(0), 0))
            x = torch.cat((d*cd, cd), dim=0)
            
        plot(x, ece, ce, row)
        row+=1

        dcd, cd = x.split(d.shape[0], 0)
        d = dcd / (cd + self.eps)

        return {'d': d, 'cd': cd, 'e': ece / (ce + self.eps), 'ce': ce}

    def visualize_weights(self, img_file_path=''):
        plt.figure()

        cols = 10 + int(self.bias is not None)
        rows = 7 * self.n_c + (3 if self.n_c == 1 else 5) + int(self.bias is not None)
        if self.sc_init is not None:
            cols += self.sc_init.n_c
            rows = max(rows, self.sc_init.n_c*4)
        col = 1
        dirs = ['/','-','\\','|']

        if(self.sc_init is not None):
            self.sc_init.visualize_weights(rows, cols, col)
            col+= self.sc_init.n_c


        idx = col
        for c in range(self.n_c):
            ax = plt.subplot(rows, cols, idx)
            plt.axis('off')
            ax.text(0,0.5,'w pow e', clip_on=True, fontsize=12)
            idx+=cols

        ax = plt.subplot(rows, cols, idx)
        plt.axis('off')
        ax.text(0,0.5,'w prop e', clip_on=True, fontsize=12)
        idx+=cols

        for c in range(self.n_c):
            for d in range(4):
                ax = plt.subplot(rows, cols, idx)
                plt.axis('off')
                ax.text(0,0.5,'w spatial e {}'.format(dirs[d]), clip_on=True, fontsize=12)
                idx+=cols
        if self.n_c > 1:
            ax = plt.subplot(rows, cols, idx)
            plt.axis('off')
            ax.text(0,0.5,'w channel e', clip_on=True, fontsize=12)
            idx+=cols
        for c in range(self.n_c):
            ax = plt.subplot(rows, cols, idx)
            plt.axis('off')
            ax.text(0,0.5,'w dir e', clip_on=True, fontsize=12)
            idx+=cols

        ax = plt.subplot(rows, cols, idx)
        plt.axis('off')
        ax.text(0,0.5,'w skip e', clip_on=True, fontsize=12)
        idx+=cols

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

        if self.bias is not None:
            ax = plt.subplot(rows, cols, idx)
            plt.axis('off')
            ax.text(0,0.5,'bias offset d', clip_on=True, fontsize=12)
            idx+=cols
            col+=1

        plt.subplot(rows, cols, col).title.set_text('initial\nconv')
        self.convs[0].visualize_weights(rows, cols, col)
        col+=1

        plt.subplot(rows, cols, col).title.set_text('encoder\npooling')
        self.pool.visualize_weights(rows, cols, col)
        col+=1

        plt.subplot(rows, cols, col).title.set_text('encoder\nconvs')
        self.convs[1].visualize_weights(rows, cols, col)
        col+=1

        for i in range(len(self.unpools)):
            plt.subplot(rows, cols, col).title.set_text('decoder\ndeconv {}'.format(i))
            self.unpools[i].visualize_weights(rows, cols, col)
            col+=1

            plt.subplot(rows, cols, col).title.set_text('decoder\nconv {}'.format(i))
            self.convs[i + 2].visualize_weights(rows, cols, col)
            col+=1

        if self.bias is not None:
            plt.subplot(rows, cols, col).title.set_text('bias')
            self.bias.visualize_weights(rows, cols, col)
            col+=1
        plt.subplots_adjust(hspace=0.5)
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        if img_file_path != '':
            plt.savefig(img_file_path, bbox_inches='tight')