import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from model.uSNC_conv import uSNC_conv as Conv
from model.uSNC_pool import uSNC_pool as Pool
from model.uSNC_unpool import uSNC_unpool as Unpool
from model.NC_bias import NC_bias as Bias
from model.uSC_init import uSC_init

class uSNC(torch.nn.Module):

    def __init__(self, params):
        super().__init__()
        self.eps = 1e-20
        n_c = params['n_channels']
        self.n_c = n_c
        self.n_colors = 0 + 3*int(params['rgb']) + int(params['reflectance'])
        self.n_stages = params['n_stages']
        self.lidar_padding = params['lidar_padding']
        self.sum_pad_d = 1
        for l in range(self.n_stages-1):
            self.sum_pad_d = 1 + 1 + 2*self.sum_pad_d + 1 + 1
        self.sum_pad_d = 2**(self.n_stages-2) * int(np.ceil((1 + self.sum_pad_d) / 2**(self.n_stages-2)))
        
        self.sc_init = uSC_init(kernel_size=5, n_colors=self.n_colors, **params) if self.n_colors > 0 else None

        convs = [Conv(n_in=1, n_out=n_c, kernel_size=5, max_pool_size=5, **params)]
        unpools = []
        for l in range(self.n_stages-1):
            convs.append(Conv(n_in=n_c, n_out=n_c, kernel_size=3, max_pool_size=3, **params))
            unpools.append(Unpool(n_c=n_c, kernel_size=4, stride=2, **params))
        convs.append(Conv(n_in=n_c, n_out=1, kernel_size=3, max_pool_size=3, **params))

        self.convs = torch.nn.ModuleList(convs)
        self.pool = Pool(n_c=n_c, kernel_size=4, stride=2, **params)
        self.unpools = torch.nn.ModuleList(unpools)
        self.bias = Bias(True) if params['use_bias'] else None

    def prep_eval(self):
        for conv in self.convs:
            conv.prep_eval()
        self.pool.prep_eval()
        for unpool in self.unpools:
            unpool.prep_eval()
        if self.bias is not None:
            self.bias.prep_eval()

    def forward(self, d, cd=None, scs=None, cs=None, **args):
        
        if self.sc_init is not None:
            if rgb is None:
                color = x
            else:
                color = rgb
            scs, cs = self.sc_init(color)
            if self.lidar_padding and rgb is not None:
                pad = (self.sum_pad_d, self.sum_pad_d, self.sum_pad_d, self.sum_pad_d)
                scs, cs = F.pad(scs, pad),  F.pad(cs, pad)
        else:
            if scs is None:
                scs = torch.ones_like(d)
            if cs is None:
                cs = torch.ones_like(scs)
        if cd is None:
            cd = (d > 0).float()
        x = torch.cat((d*cd, cd), dim=0)
            
        # Encoder
        x, scs, cs = self.convs[0](x, scs, cs)
        skip = []
        for stage in range(self.n_stages-1):
            skip.append((x, scs, cs))
            x, scs, cs = self.pool(x, scs, cs)
            x, scs, cs = self.convs[1](x, scs, cs)

        # Decoder
        for stage in range(self.n_stages-1):
            (d_skip, cd_skip, scs_skip, cs_skip) = skip[-1]
            del skip[-1]
            x, scs, cs = self.unpools[stage](x_skip=x_skip, scs_skip=scs_skip, cs_skip=cs_skip, x_pool=x, scs_pool=scs, cs_pool=cs)
            x, scs, cs = self.convs[2+stage](x, scs, cs)

        if self.lidar_padding:
            x, scs, cs = x[..., self.sum_pad_d:-self.sum_pad_d],  scs[..., self.sum_pad_d:-self.sum_pad_d],  cs[..., self.sum_pad_d:-self.sum_pad_d]

        if self.bias is not None:
            d, cd = self.bias(x.split(d.shape(0), 0))
        else:
            dcd, cd = x.split(d.shape[0], 0)
            d = dcd / (cd + self.eps)

        return {'d': d, 'cd': cd, 's': scs / (cs + self.eps), 'cs': cs}

    def streaming_percsption(self, d, cd=None, scs=None, cs=None, **args):

        lower_quantile = np.quantile(d[d > 0].cpu().numpy(), 0.05)
        upper_quantile = np.quantile(d[d > 0].cpu().numpy(), 0.95)
        cmap = plt.cm.get_cmap('nipy_spectral', 256)
        cmap = np.ndarray.astype(np.array([cmap(i) for i in range(256)])[:, :3] * 255, np.uint8)
        plt.figure()

        def plot(x, scs, cs, row):
            dcd, cd = x.split(ce.shape[0], 0)
            d = dcd / (cd + self.eps)
            rows = 2*self.n_stages

            plt.subplot(rows, 4, 4 * row + 1)
            d, cd, s, cs = d.cpu().numpy().squeeze(), cd.cpu().numpy().squeeze(), (scs / (cs + 1e-20)).cpu().numpy().squeeze(), cs.cpu().numpy().squeeze()
            d_img = cmap[np.ndarray.astype(np.interp(d, (lower_quantile, upper_quantile), (0, 255)), np.int_),:]
            d_img[cd == 0,:] = 128
            plt.imshow(d_img)

            plt.subplot(rows, 4, 4 * row + 2)
            c_img = cmap[np.ndarray.astype(np.interp(cd / np.max(cd), (0, 1), (0, 255)), np.int_), :]
            c_img[cd == 0,:] = 128
            plt.imshow(c_img)

            plt.subplot(rows, 4, 4 * row + 3)
            s_img = cmap[np.ndarray.astype(np.interp(s, (0, 1), (0, 255)), np.int_), :]
            s_img[cs == 0,:] = 128
            plt.imshow(s_img)

            plt.subplot(rows, 4, 4 * row + 4)
            cs_img = cmap[np.ndarray.astype(np.interp(cs / np.max(cs), (0, 1), (0, 255)), np.int_),:]
            cs_img[cs == 0,:] = 128
            plt.imshow(cs_img)
            
        if self.sc_init is not None:
            if rgb is None:
                color = x
            else:
                color = rgb
            scs, cs = self.sc_init(color)
            if self.lidar_padding and rgb is not None:
                pad = (self.sum_pad_d, self.sum_pad_d, self.sum_pad_d, self.sum_pad_d)
                scs, cs = F.pad(scs, pad),  F.pad(cs, pad)
        else:
            if scs is None:
                scs = torch.ones_like(d)
            if cs is None:
                cs = torch.ones_like(scs)
        if cd is None:
            cd = (d > 0).float()
        x = torch.cat((d*cd, cd), dim=0)
                   
            
        # Encoder
        row=0
        plot(x, scs, cs, row)
        row+=1
        x, scs, cs = self.convs[0](x, scs, cs)
        skip = []
        for stage in range(self.n_stages-1):
            skip.append((x, scs, cs))
            x, scs, cs = self.pool(x, scs, cs)
            plot(x, scs, cs, row)
            row+=1
            x, scs, cs = self.convs[1](x, scs, cs)

        # Decoder
        for stage in range(self.n_stages-1):
            (x_skip, scs_skip, cs_skip) = skip[-1]
            del skip[-1]
            x, scs, cs = self.unpools[stage](x=x_skip, scs_skip=scs_skip, cs_skip=cs_skip, x_pool=x, scs_pool=scs, cs_pool=cs)
            plot(x, scs, cs, row)
            row+=1
            x, scs, cs = self.convs[2+stage](x, scs, cs)
            

        if self.lidar_padding:
            x, scs, cs = x[..., self.sum_pad_d:-self.sum_pad_d],  scs[..., self.sum_pad_d:-self.sum_pad_d],  cs[..., self.sum_pad_d:-self.sum_pad_d]
            
        if self.bias is not None:
            d, cd = self.bias(x.split(d.shape(0), 0))
            x = torch.cat((d*cd, cd), dim=0)

        plot(d, cd, scs, cs, row)
        
        dcd, cd = x.split(d.shape[0], 0)
        d = dcd / (cd + self.eps)

        return {'d': d, 'cd': cd, 's': scs / (cs + self.eps), 'cs': cs}

    def visualize_weights(self, img_file_path=''):
        plt.figure()

        cols = 12 if self.bias is not None else 11
        rows = 2 * self.n_c + (4 if self.n_c == 1 else 6)
        col = 1
        idx = 1
        ax = plt.subplot(rows, cols, idx)
        plt.axis('off')
        ax.text(0,0.5,'w pow s', clip_on=True, fontsize=12)
        idx+=cols

        ax = plt.subplot(rows, cols, idx)
        plt.axis('off')
        ax.text(0,0.5,'w prop s', clip_on=True, fontsize=12)
        idx+=cols

        for c in range(self.n_c):
            ax = plt.subplot(rows, cols, idx)
            plt.axis('off')
            ax.text(0,0.5,'w spatial s', clip_on=True, fontsize=12)
            idx+=cols        
        
        if self.n_c > 1:
            ax = plt.subplot(rows, cols, idx)
            plt.axis('off')
            ax.text(0,0.5,'w channel s', clip_on=True, fontsize=12)
            idx+=cols

        ax = plt.subplot(rows, cols, idx)
        plt.axis('off')
        ax.text(0,0.5,'w skip s', clip_on=True, fontsize=12)
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