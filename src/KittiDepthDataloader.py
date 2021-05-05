########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

import os
import torch
from torch.utils.data import DataLoader, Dataset

from src.KittiDepthDataset import KittiDepthDataset

def KittiDepthDataloader(params, sets, mode, device, lidar_padding):

    kitti_depth_dir = params['paths']['kitti_depth_dataset_dir']
    kitti_rgb_dir = params['paths']['kitti_rgb_dataset_dir']
    load_rgb = params['model']['rgb'] or mode == 'predictions'
    lidar_projection = params['model']['lidar_projection'] if 'lidar_projection' in params['model'] else False

    dataloaders = {}

    if 'train' in sets:
        train_set = KittiDepthDataset(kitti_depth_path=os.path.join(kitti_depth_dir, 'train'), setname='train',
                                      load_rgb=load_rgb, rgb_dir=kitti_rgb_dir, lidar_padding=lidar_padding, lidar_projection=lidar_projection, device=device)
        train_set.sparse_depth_paths = train_set.sparse_depth_paths[0:params['training']['train_on']]
        train_set.gt_depth_paths = train_set.gt_depth_paths[0:params['training']['train_on']]

        dataloaders['train'] = DataLoader(train_set, shuffle=True, batch_size=1 if mode == 'predictions' else params['training']['train_batch_size'], num_workers=0)

    if 'val' in sets:
        val_set = KittiDepthDataset(kitti_depth_path=os.path.join(kitti_depth_dir, 'val'), setname='val',
                                    load_rgb=load_rgb, rgb_dir=kitti_rgb_dir, lidar_padding=lidar_padding, lidar_projection=lidar_projection, device=device)
        dataloaders['val'] = DataLoader(val_set, shuffle = mode == 'predictions', batch_size=1 if mode == 'predictions' else params['training']['val_batch_size'], num_workers=0)

    if 'selval' in sets:
        test_set = KittiDepthDataset(kitti_depth_path=os.path.join(kitti_depth_dir, 'val_selection_cropped'), setname='selval',
                                     load_rgb=load_rgb, rgb_dir=kitti_rgb_dir, lidar_padding=lidar_paddingg, lidar_projection=lidar_projection, device=device)
        dataloaders['selval'] = DataLoader(test_set, shuffle=mode == 'predictions', batch_size=1 if mode == 'predictions' else params['training']['val_batch_size'], num_workers=0)

    if 'test' in sets:
        test_set = KittiDepthDataset(kitti_depth_path=os.path.join(kitti_depth_dir, 'test_depth_completion_anonymous'), setname='test',
                                     load_rgb=load_rgb, rgb_dir=kitti_rgb_dir, device=device)
        dataloaders['test'] = DataLoader(test_set, shuffle=False, batch_size=1 if mode == 'predictions' else params['training']['val_batch_size'], num_workers=0)

    dataset_sizes = {}
    for set in dataloaders:
        dataset_sizes[set] = len(dataloaders[set])
    print(dataset_sizes)

    return dataloaders, dataset_sizes