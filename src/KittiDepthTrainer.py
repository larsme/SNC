import os
import sys
import yaml
import torch
import numpy as np
import time
import importlib
import PIL.Image as Image
import matplotlib.pyplot as plt
import os.path
import glob
import shutil
import pandas as pd

from src.losses import *
from src.error_metrics import *
from src.KittiDepthDataloader import KittiDepthDataloader

err_metrics = ['MAE', 'RMSE', 'Delta1', 'Delta2', 'Delta3', 'Parameters', 'BatchDuration', 'wMAE', 'wRMSE', 'wDelta1', 'wDelta2', 'wDelta3']

class KittiDepthTrainer:
    def __init__(self, params_dir=None, run=0, sets=['train', 'val'], mode='', epoch=None, save_chkpt_each=1, **args):

        # Call the constructor of the parent class (trainer)
        super().__init__()

        with open('{}/params.yaml'.format(params_dir), 'r') as fp:
            self.params = yaml.safe_load(fp)

        self.params_dir = params_dir
        self.run = run

        self.device = torch.device(self.params['training']['device'])
        self.model = getattr(importlib.import_module('model.' + self.params['model']['model']),self.params['model']['model'])(self.params['model'])
        self.model.to(self.device)
        self.save_chkpt_each = save_chkpt_each
        self.epoch = epoch
        if not self.load_checkpoint() and self.save_chkpt_each > 0:
            print('checkpoint not found, starting from scratch')
            self.save_checkpoint()

        self.input_rgb = self.params['model']['rgb']
        self.load_rgb = self.input_rgb or mode == 'display'

        lidar_padding = False
        if self.params['model']['reflectance']:
            lidar_padding=0
        if self.model.lidar_padding:
            lidar_padding = self.model.sum_pad_d
        self.dataloaders, self.dataset_sizes = KittiDepthDataloader(self.params, sets, mode, self.device, lidar_padding=lidar_padding)

        self.ref_loss = None

    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""

        file_path = '{}/checkpoint_run{:04d}_ep{:04d}.pt'.format(self.params_dir,self.run, self.epoch)
        if not os.path.exists(file_path):
            torch.save(self.model.state_dict(), file_path)
            print('saved {}'.format(file_path))

    def load_checkpoint(self, checkpoint=None):

        file_path = '{}/checkpoint_run{:04d}_ep{:04d}.pt'.format(self.params_dir,self.run, self.epoch)
        if self.epoch < 0:
            # Load most recent checkpoint
            for possible_path in glob.glob('{}/checkpoint_run{:04d}_ep*.pt'.format(self.params_dir,self.run)):
                epoch = int(possible_path.split('_ep')[1].split('.pt')[0])
                if (epoch > self.epoch) and (epoch <= self.params['training']["num_epochs"]):
                    self.epoch = epoch
                    file_path = possible_path

        if os.path.exists(file_path):
            if self.params['model']['model'] == 'MSNC' and self.model.lidar_model is None:
                try:
                    self.model.load_state_dict(torch.load(file_path))
                except:
                    self.model.image_model.load_state_dict(torch.load(file_path))
            else:
                self.model.load_state_dict(torch.load(file_path))
            print('loaded {}'.format(file_path))
            return True
        else:
            self.epoch = 0
            return False

    def count_parameters(self):
        parameter_count = 0
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                parameter_count += parameter.numel()
        print('the model has %s leanable parameters\n' % (parameter_count))
        return parameter_count

    def train(self, trainsets=['train'], evalsets=['val'], evaluate_all_epochs=False):

        # Fix CUDNN error for non-contiguous inputs
        import torch.backends.cudnn as cudnn
        cudnn.enabled = False
        # While cudnn improves performance, it can also lead to negative outputs of convolutions with positive weights and positive tensors, breaking SNC in the process
        # https://github.com/pytorch/pytorch/issues/30934
        # While I have only noticed this with directed smoothness gating so far, let's stay on the safe side.
        # Feel free to enable this again for the less fragile versions, i don't think this will ever be a problem for non-gated NConvs
        #cudnn.benchmark = True

        objective = globals()[self.params['training']['loss']]()
        optimizer = getattr(torch.optim, self.params['training']['optimizer'])(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.params['training']['lr'], weight_decay=0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.params['training']['lr_decay_step'], gamma=self.params['training']['lr_decay'])

        print('#############################\n### Experiment Parameters ###\n#############################')
        print(yaml.dump(self.params))

        for epoch in range(self.epoch, self.params['training']['num_epochs']): # range function returns max_epochs-1
            self.epoch = epoch

            if evaluate_all_epochs:
                self.evaluate(evalsets, objective)

            self.epoch = epoch + 1
            if self.load_checkpoint():
                print('Checkpoint for epoch {} already exists, loading instead'.format(self.epoch))
                continue
            else:
                self.epoch = epoch + 1

            print('\nTraining Epoch {}: (lr={}) '.format(self.epoch, optimizer.param_groups[0]['lr']))

            # Train the epoch
            self.model.train(True)
            for set in trainsets:
                i = 0
                for (item, sparse_depth, sparse_intensity, gt_depth, rgb, dirs, offsets, im_shape) in self.dataloaders[set]:
                    loss, npoints = self.train_step(optimizer, objective, sparse_depth, sparse_intensity, gt_depth, rgb, dirs, offsets, im_shape)
                    i += 1
                    str = 'trained batch {} of {} on {} set - loss = {:.4f}\t'.format(i, len(self.dataloaders[set]), set, loss)
                    if self.ref_loss is None:
                        self.ref_loss = loss * 3
                    if np.isnan(loss):
                        return
                    str += ' ' * int((shutil.get_terminal_size().columns - len(str)) * loss / self.ref_loss) + '|'
                    print(str)

            # Decay Learning Rate
            scheduler.step() # LR decay

            # Save checkpoint
            if self.epoch % self.params['training']['save_chkpt_each'] == 0:
                self.save_checkpoint()

        self.save_checkpoint()
        print("Training Finished.\n")
        self.evaluate(evalsets, objective)

    def train_step(self, optimizer, objective, sparse_depth, sparse_intensity, gt_depth, rgb, dirs, offsets, im_shape):
        output = self.model(d=sparse_depth, rgb=rgb, r = sparse_intensity, dirs=dirs, offsets=offsets, im_shape=im_shape)

        # Calculate loss for valid pixel in the ground truth
        loss = objective(output['d'], gt_depth, output['cd'], self.epoch)
        loss.backward()
        skip = False
        for param in self.model.parameters():
            if param.requires_grad and param.grad.isnan().any():
                print(param)
                skip = True
        if skip:
            self.model.visualize_weights()
        else:
            optimizer.step()
            for param in self.model.parameters():
                param.grad = None

        return loss.detach().item(), sparse_depth.detach().size(0)

    def evaluate(self, evalsets, objective):
        self.model.train(False)
        self.model.prep_eval()

        parameter_count = self.count_parameters()

        with torch.no_grad():
            for set in evalsets:
                metrics_file = '{}/metrics_{}.csv'.format(self.params_dir, set)
                if os.path.exists(metrics_file):
                    metrics_df = pd.read_csv(metrics_file).set_index(['Epoch', 'Run'])
                    if metrics_df.index.isin([(self.epoch,self.run)]).any():
                        print('Evaluation already done')
                        print(metrics_df)
                        continue
                else:
                    metrics_df = pd.DataFrame()

                metrics = {'loss':0}
                for metric in err_metrics:
                    metrics[metric] = 0
                count = 0

                print('Evaluating on [{}] set, Epoch [{}] ! \n'.format(set, self.epoch))
                i = 0
                # Iterate over data.
                for (item, sparse_depth, sparse_intensity, gt_depth, rgb, dirs, offsets, im_shape) in self.dataloaders[set]:
                    i+=1
                    print('eval batch %d of %d' % (i, len(self.dataloaders[set])), end='\r')

                    t = time.time()
                    output = self.model(d=sparse_depth, rgb=rgb, dirs=dirs, offsets=offsets, r=sparse_intensity, im_shape=im_shape)
                    elapsed = time.time() - t

                    loss = objective(output['d'], gt_depth, output['cd'], self.epoch)

                    # metrics
                    count += len(item)
                    metrics['loss']+=loss.item() * len(item)
                    for metric in err_metrics:
                        if "wDelta" in metric:
                            if metric == 'wDelta1':
                                fn = globals()['wDeltas']()
                                error = fn(**output, gt_d=gt_depth)
                                metrics['wDelta1']+=error[0].item() * len(item)
                                metrics['wDelta2']+=error[1].item() * len(item)
                                metrics['wDelta3']+=error[2].item() * len(item)
                        elif "Delta" in metric:
                            if metric == 'Delta1':
                                fn = globals()['Deltas']()
                                error = fn(**output, gt_d=gt_depth)
                                metrics['Delta1']+=error[0].item() * len(item)
                                metrics['Delta2']+=error[1].item() * len(item)
                                metrics['Delta3']+=error[2].item() * len(item)
                        elif 'BatchDuration' == metric:
                            metrics['BatchDuration']+=elapsed * len(item)
                        elif 'Parameters' == metric:
                            continue
                        else:
                            metrics[metric] += globals()[metric]()(**output, gt_d=gt_depth).item() * len(item)
                for metric in metrics:
                    metrics[metric]/=count
                if 'Parameters' in err_metrics:
                    metrics['Parameters'] = parameter_count

                self.ref_loss = metrics['loss'] * 3

                metrics['Run'] = self.run
                metrics['Epoch'] = self.epoch
                metrics_df = pd.concat((metrics_df, pd.DataFrame(metrics, columns=list(metrics), index=[0]).set_index(['Epoch', 'Run'])))
                metrics_df = pd.DataFrame.sort_index(metrics_df, level=['Epoch', 'Run'])
                metrics_df.to_csv(metrics_file)

                print('Evaluation results on {}:\n============================='.format(set))
                print(metrics_df)

    def display_predictions(self, set='val', color_nine_pixels_for_every_sparse_one=True, target_item=None):

        self.model.train(False)
        self.model.prep_eval()

        prev_items = []

        fig = None
        with torch.no_grad():
            while True:
                # Iterate over data.
                for (item, sparse_depth, sparse_intensity, gt_depth, rgb, dirs, offsets, im_shape) in self.dataloaders[set]:
                    item = item.item()

                    if target_item is not None:
                        if target_item > len(self.dataloaders[set]):
                            break
                        (item, sparse_depth, sparse_intensity, gt_depth, rgb, dirs, offsets, im_shape) = self.dataloaders[set].dataset[target_item]
                        sparse_depth, sparse_intensity, gt_depth, rgb, dirs, offsets, im_shape = (
                             sparse_depth[None,...], torch.tensor(sparse_intensity)[None,...], gt_depth[None,...], rgb[None,...], torch.tensor(dirs)[None,...], torch.tensor(offsets)[None,...], torch.tensor(im_shape)[None,...])

                    if fig is None or plt.fignum_exists(fig.number):
                        print(item)
                        prev_items.append(item)

                        outs = self.model(d=sparse_depth, x=sparse_intensity, rgb=rgb, dirs=dirs, offsets=offsets, im_shape=im_shape)

                        d = outs['d'].squeeze().cpu().numpy()
                        cd = outs['cd'].squeeze().cpu().numpy()
                        if 's' in outs:
                            s = outs['s'].squeeze().cpu().numpy()
                            cs = outs['cs'].squeeze().cpu().numpy()
                        elif 'e' in outs:
                            s = outs['e'].squeeze().cpu().numpy().prod(0)
                            cs = outs['ce'].squeeze().cpu().numpy().mean(0)

                        if fig is None:
                            fig,ax = plt.subplots(1 + len(outs) // 2, 2,)
                            plt.axis("off")
                            plt.tight_layout()
                            for a in fig.axes:
                                a.get_xaxis().set_visible(False)
                                a.get_yaxis().set_visible(False)
                        fig.canvas.set_window_title(item)

                        img_rgb = Image.fromarray((rgb.squeeze().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8))
                        if self.input_rgb:
                            ax[0][1].set_title('input camera image')
                        else:
                            ax[0][1].set_title('unseen camera image')
                        ax[0][1].imshow(img_rgb)
                        #img_rgb.save('rgb.png')

                        sparse_depth = sparse_depth.squeeze().cpu().numpy()
                        if color_nine_pixels_for_every_sparse_one:
                                sparse_depth[1:,:][sparse_depth[1:,:] == 0] = sparse_depth[:-1,:][sparse_depth[1:,:] == 0]
                                sparse_depth[:-1,:][sparse_depth[:-1,:] == 0] = sparse_depth[1:,:][sparse_depth[:-1,:] == 0]
                                sparse_depth[:,1:][sparse_depth[:,1:] == 0] = sparse_depth[:,:-1][sparse_depth[:,1:] == 0]
                                sparse_depth[:,:-1][sparse_depth[:,:-1] == 0] = sparse_depth[:,1:][sparse_depth[:,:-1] == 0]

                        # defing color maps
                        lower_quantile = np.quantile(sparse_depth[sparse_depth > 0], 0.05)
                        upper_quantile = np.quantile(sparse_depth[sparse_depth > 0], 0.95)
                        cmap = plt.cm.get_cmap('nipy_spectral', 256)
                        cmap = np.ndarray.astype(np.array([cmap(i) for i in range(256)])[:, :3] * 255, np.uint8)

                        sparse_depth_img = cmap[np.ndarray.astype(np.interp(sparse_depth, (lower_quantile, upper_quantile), (0, 255)), np.int_),:]
                        sparse_depth_img[sparse_depth == 0,:] = 128
                        sparse_depth_img = Image.fromarray(sparse_depth_img)
                        ax[0][0].set_title('sparse depth input')
                        ax[0][0].imshow(sparse_depth_img)

                        pred_depth_img = cmap[np.ndarray.astype(np.interp(d, (lower_quantile, upper_quantile), (0, 255)), np.int_), :]
                        pred_depth_img[cd == 0,:] = 128
                        pred_depth_img = Image.fromarray(pred_depth_img)
                        ax[1][0].set_title('dense depth output')
                        ax[1][0].imshow(pred_depth_img)

                        c_img = cmap[np.ndarray.astype(np.interp(cd / np.max(cd), (0, 1), (0, 255)), np.int_), :]
                        c_img[cd == 0,:] = 128
                        c_img = Image.fromarray(c_img)
                        ax[1][1].set_title('depth output confidence')
                        ax[1][1].imshow(c_img)

                        if len(outs) > 2:
                            s_img = cmap[np.ndarray.astype(np.interp(s, (0, 1), (0, 255)), np.int_), :]
                            s_img[cs == 0,:] = 128
                            s_img = Image.fromarray(s_img)
                            ax[2][0].set_title('dense smoothness output')
                            ax[2][0].imshow(s_img)

                            cs_img = cmap[np.ndarray.astype(np.interp(cs / np.max(cs), (0, 1), (0, 255)), np.int_),:]
                            cs_img[cs == 0,:] = 128
                            cs_img = Image.fromarray(cs_img)
                            ax[2][1].set_title('smoothness output confidence')
                            ax[2][1].imshow(cs_img)

                        target_item = None

                        while(True):
                            global k
                            def press(event):
                                global k
                                k = event.key
                            fig.canvas.mpl_connect('key_press_event', press)
                            while not plt.waitforbuttonpress(): pass
                            if k == 'w':
                                os.makedirs('images', exist_ok=True)
                                checkpoint_path = '{}/checkpoint_run{:04d}_ep{:04d}.pt'.format(self.params_dir,self.run, self.epoch)
                                file_start = 'images/{}_{}'.format(checkpoint_path.replace('.pt','').replace('/','_').replace('\\','_'), item)
                                img_rgb.save('{}_camera_img.png'.format(file_start))
                                sparse_depth_img.save('{}_lidar_img.png'.format(file_start))
                                pred_depth_img.save('{}_pred_depth.png'.format(file_start))
                                c_img.save('{}_pred_certainty.png'.format(file_start))
                                if len(outs) > 2:
                                    s_img.save('{}_pred_smoothness.png'.format(file_start))
                                    cs_img.save('{}_pred_smoothness_certainty.png'.format(file_start))
                                plt.savefig('{}_prediction.png'.format(file_start))
                            elif k == 'q':
                                fig.close()
                                return
                            elif len(k) == 1 and ord(k) >= ord('0') and ord(k) <= ord('9'):
                                if target_item is None:
                                    target_item = 0
                                target_item = target_item * 10 + int(k)
                                fig.canvas.set_window_title('{} -> {}'.format(item, target_item))
                            elif k == 'left':
                                target_item = item - 1
                                break
                            elif k == 'right':
                                target_item = item + 1
                                break
                            elif k == 'backspace' and len(prev_items) > 1:
                                target_item = prev_items[-2]
                                del prev_items[-2]
                                del prev_items[-1]
                                break
                            elif k == 'enter':
                                break
                            elif k == 'p':
                                target_item=item
                                fig=None
                                plt.close()
                                (_item, _sparse_depth, _sparse_intensity, _gt_depth, _rgb, _dirs, _offsets, _im_shape) = self.dataloaders[set].dataset[item]
                                _sparse_depth, _sparse_intensity, _gt_depth, _rgb, _dirs, _offsets, _im_shape = (
                                    _sparse_depth[None,...], torch.tensor(_sparse_intensity)[None,...], _gt_depth[None,...], _rgb[None,...], torch.tensor(_dirs)[None,...], torch.tensor(_offsets)[None,...], torch.tensor(_im_shape)[None,...])
                                self.model.streaming_perception(d=_sparse_depth, rgb=_rgb, x=_sparse_intensity, dirs=_dirs, offsets=_offsets, im_shape=_im_shape)
                                mng = plt.get_current_fig_manager()
                                mng.window.state('zoomed')
                                plt.tight_layout()
                                plt.show()
                                break
                            elif k=='l':
                                target_item=item
                                fig=None
                                plt.close()
                                self.dataloaders[set].dataset.test_projections(item)
                                break
                            else:
                                print(k)
                                break

    def visualize_weights(self):
        checkpoint_path = '{}/weights_run{:04d}_ep{:04d}.pt'.format(self.params_dir,self.run, self.epoch)
        img_file_path = 'images/{}_weights.png'.format(checkpoint_path.replace('.pt','').replace('/weights','').replace('/','_').replace('\\','_'))
        self.model.visualize_weights(img_file_path)