########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import glob
import os
from einops import rearrange, reduce
from scipy.interpolate import interp1d

class KittiDepthDataset(Dataset):

    def __init__(self, kitti_depth_path, setname='train', load_rgb=False, rgb_dir=None, lidar_padding=None, lidar_projection=False, device=None):

        self.kitti_depth_path = kitti_depth_path
        self.setname = setname
        self.load_rgb = load_rgb
        self.rgb_dir = rgb_dir
        self.lidar_padding = lidar_padding
        self.lidar_projection = lidar_projection
        self.device = device

        if setname == 'train' or setname == 'val':
            self.sparse_depth_paths = list(sorted(glob.iglob(self.kitti_depth_path + "/*/*/velodyne_raw/*/*.png", recursive=True)))
            self.gt_depth_paths = list(sorted(glob.iglob(self.kitti_depth_path + "/*/*/groundtruth/*/*.png", recursive=True)))
            assert (len(self.sparse_depth_paths) == len(self.gt_depth_paths))
        elif setname == 'selval':
            self.sparse_depth_paths = list(sorted(glob.iglob(self.kitti_depth_path + "/**/velodyne_raw/**.png", recursive=True)))
            self.gt_depth_paths = list(sorted(glob.iglob(self.kitti_depth_path + "/**/groundtruth_depth/**.png", recursive=True)))
            assert (len(self.sparse_depth_paths) == len(self.gt_depth_paths))
        else:
            self.sparse_depth_paths = list(sorted(glob.iglob(self.kitti_depth_path + "/**/velodyne_raw/**.png", recursive=True)))
            self.gt_depth_paths = []
        self.sparse_depth_paths = [x.replace('\\','/') for x in  self.sparse_depth_paths]
        self.gt_depth_paths = [x.replace('\\','/') for x in  self.gt_depth_paths]
        self.poses = {}

        if self.setname != 'train' and self.setname != 'val':
            assert self.lidar_padding is False

        # Check if Data filename is equal to GT filename
        if self.setname == 'train' or self.setname == 'val':
            for item in range(len(self.sparse_depth_paths)):
                sparse_depth_path = self.sparse_depth_paths[item].split(self.setname)[1]
                gt_depth_path = self.gt_depth_paths[item].split(self.setname)[1]
                # print((sparse_depth_path, gt_depth_path))

                assert (sparse_depth_path[0:25] == gt_depth_path[0:25])  # Check folder name

                sparse_depth_path = sparse_depth_path.split('image')[1]
                gt_depth_path = gt_depth_path.split('image')[1]

                assert (sparse_depth_path == gt_depth_path)  # Check filename

        elif self.setname == 'selval':
            for item in range(len(self.sparse_depth_paths)):
                sparse_depth_path = self.sparse_depth_paths[item].split('00000')[1]
                gt_depth_path = self.gt_depth_paths[item].split('00000')[1]
                assert (sparse_depth_path == gt_depth_path)

    def __len__(self):
        return len(self.sparse_depth_paths)

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

        # crop height to 368 & width to 1224 since kitti is not consistent and these are multiples of 8
        gt_depth = np.array(Image.open(str(self.gt_depth_paths[item])))[-368:,-1224:] / 256
        if self.lidar_projection:
            sparse_depth, sparse_intensity, dirs, offsets, im_shape = self.project_velo_points_to_velo_grid(item)
            dirs, offsets, im_shape = torch.tensor(dirs, device=self.device), torch.tensor(offsets, device=self.device), torch.tensor(im_shape, device=self.device)
            sparse_intensity = torch.tensor(sparse_intensity[None,...], dtype=torch.float, device=self.device)
        else:
            dirs, offsets, im_shape = 0, 0, 0
            if self.lidar_padding is not False:   # can have a value of 0
                sparse_depth, sparse_intensity = self.project_velo_points_to_image(item)
                sparse_intensity = torch.tensor(sparse_intensity[None,...], dtype=torch.float, device=self.device)
            else:
                sparse_intensity = 0
                sparse_depth = np.array(Image.open(str(self.sparse_depth_paths[item])))[-368:,-1224:] / 256
                

        # add channel dim & convert to Pytorch Tensors
        gt_depth = torch.tensor(gt_depth[None,...], dtype=torch.float, device=self.device)
        sparse_depth = torch.tensor(sparse_depth[None,...], dtype=torch.float, device=self.device)

        # Read RGB images
        if self.load_rgb:
            if self.setname == 'train' or self.setname == 'val':
                split = (self.gt_depth_paths[item].split(self.setname)[1]).split('/')
                drive_dir = split[1]
                day_dir = drive_dir.split('_drive')[0]
                img_source_dir = split[4]
                img_idx_dir = split[5]
                rgb_path = self.rgb_dir + '/' + day_dir + '/' + drive_dir + '/' + img_source_dir + '/data/' + img_idx_dir
            elif self.setname == 'selval':
                sparse_depth_path = str(self.sparse_depth_paths[item])
                idx = sparse_depth_path.find('velodyne_raw')
                fname = sparse_depth_path[idx + 12:]
                idx2 = fname.find('velodyne_raw')
                rgb_path = sparse_depth_path[:idx] + 'image' + fname[:idx2] + 'image' + fname[idx2 + 12:]
            elif self.setname == 'test':
                sparse_depth_path = str(self.sparse_depth_paths[item])
                idx = sparse_depth_path.find('velodyne_raw')
                fname = sparse_depth_path[idx + 12:]
                idx2 = fname.find('test')
                rgb_path = sparse_depth_path[:idx] + 'image' + fname[idx2 + 4:]
            rgb = np.array(Image.open(rgb_path), dtype=np.float) / 255
            rgb = np.transpose(rgb, (2, 0, 1))

            # crop height to 368 & width to 1224 since kitti is not consistent and these are multiples of 8
            rgb = rgb[:,-368:,-1224:]

            rgb = torch.tensor(rgb, dtype=torch.float).to(self.device)
        else:
            rgb = -1

        return item, sparse_depth, sparse_intensity, gt_depth, rgb, dirs, offsets, im_shape

    def test_projections(self,item):
        lidar_padding = self.lidar_padding
        self.lidar_padding = 0

        velo_grid, intensity_grid, dirs, offsets, im_shape = self.project_velo_points_to_velo_grid(item)
        velo_to_grid_to_image, intensity_to_grid_to_image = torch.tensor(velo_grid, device=self.device)[None,None,...], torch.tensor(intensity_grid, device=self.device)[None,None,...]
        dirs, offsets, im_shape = torch.tensor(dirs, device=self.device)[None,...], torch.tensor(offsets, device=self.device)[None,...], torch.tensor(im_shape, device=self.device)[None,...]
        d, cd, x = project_velo_grid_to_image(self.lidar_padding, velo_to_grid_to_image, torch.ones_like(velo_to_grid_to_image),intensity_to_grid_to_image, dirs, offsets, im_shape)
        velo_to_grid_to_image = d.cpu().numpy().squeeze()
        intensity_to_grid_to_image = x.cpu().numpy().squeeze()

        import matplotlib.pyplot as plt
        lower_quantile = np.quantile(velo_to_grid_to_image[velo_to_grid_to_image != 0], 0.05)
        upper_quantile = np.quantile(velo_to_grid_to_image[velo_to_grid_to_image != 0], 0.95)
        cmap = plt.cm.get_cmap('nipy_spectral', 256)
        cmap = np.ndarray.astype(np.array([cmap(i) for i in range(256)])[:, :3] * 255, np.uint8)

        # use different figures so you can toggle between them to spot the differences
        
        plt.figure()
        plt.suptitle('velo to grid')
        plt.subplot(211).title.set_text("depth")
        velo_grid_im = cmap[np.ndarray.astype(np.interp(velo_grid, (lower_quantile, upper_quantile), (0, 255)), np.int_),:]
        velo_grid_im[velo_grid == 0,:] = 128
        plt.imshow(velo_grid_im)
        plt.axis('off')

        plt.subplot(212).title.set_text("intensity")
        velo_grid_im = cmap[np.ndarray.astype(np.interp(intensity_grid, (0, 1), (0, 255)), np.int_),:]
        #velo_grid_im[velo_grid == 0,:] = 128
        plt.imshow(velo_grid_im)
        plt.axis('off')

        plt.figure()
        plt.subplot(211).title.set_text('velo to grid to image')
        velo_grid_to_image_img = cmap[np.ndarray.astype(np.interp(velo_to_grid_to_image, (lower_quantile, upper_quantile), (0, 255)), np.int_),:]
        velo_grid_to_image_img[velo_to_grid_to_image == 0,:] = 128
        plt.imshow(velo_grid_to_image_img)
        plt.axis('off')

        plt.subplot(212).title.set_text('intensity to grid to image')
        intensity_grid_to_image_img = cmap[np.ndarray.astype(np.interp(intensity_to_grid_to_image, (0, 1), (0, 255)), np.int_),:]
        intensity_grid_to_image_img[velo_to_grid_to_image == 0,:] = 128
        plt.imshow(intensity_grid_to_image_img)
        plt.axis('off')
        
        plt.figure()
        velo_to_image, intensity_to_image = self.project_velo_points_to_image(item)
        plt.subplot(211).title.set_text('velo to image')
        velo_to_image_img = cmap[np.ndarray.astype(np.interp(velo_to_image, (lower_quantile, upper_quantile), (0, 255)), np.int_),:]
        velo_to_image_img[velo_to_image == 0,:] = 128
        plt.imshow(velo_to_image_img)
        plt.axis('off')

        plt.subplot(212).title.set_text('intensity to image')
        intensity_to_image_img = cmap[np.ndarray.astype(np.interp(intensity_to_image, (0, 1), (0, 255)), np.int_),:]
        intensity_to_image_img[velo_to_image == 0,:] = 128
        plt.imshow(intensity_to_image_img)
        plt.axis('off')
        
        plt.figure()
        kitti_sparse = np.array(Image.open(str(self.sparse_depth_paths[item])))[-368:,-1224:] / 256
        plt.subplot(211).title.set_text('kitti sparse depth')
        kitti_sparse_img = cmap[np.ndarray.astype(np.interp(kitti_sparse, (lower_quantile, upper_quantile), (0, 255)), np.int_),:]
        kitti_sparse_img[kitti_sparse == 0,:] = 128
        plt.imshow(kitti_sparse_img)
        plt.axis('off')
            
        split = (self.gt_depth_paths[item].split(self.setname)[1]).split('/')
        drive_dir = split[1]
        day_dir = drive_dir.split('_drive')[0]
        img_source_dir = split[4]
        img_idx_dir = split[5]
        rgb_path = self.rgb_dir + '/' + day_dir + '/' + drive_dir + '/' + img_source_dir + '/data/' + img_idx_dir
        
        rgb = np.array(Image.open(rgb_path), dtype=np.float) / 255
        plt.figure()
        plt.suptitle('kitti rgb')
        plt.imshow(rgb)
        plt.axis('off')
        plt.show()

        self.lidar_padding = lidar_padding


    def project_velo_points_to_image(self, item):

        velo_filename, velo2im, im_shape, velo_start2end = self.get_velo_data(item)
        velo = np.fromfile(velo_filename, dtype=np.float32).reshape(-1, 4) # forward, left, up, intensity

        forward = velo[:, 0] >= 0
        velo = velo[forward, :] # forward only

        # untwist egomotion
        velo[:,:3] += (get_lidar_cols(velo, 2000) / 1999 - 0.5)[:,None] * velo_start2end[None,:]

        # project the points to the camera
        intensity = velo[:,3].copy()
        velo[:,3] = 1  # homogeneous coodinates
        velo_pts_im = np.einsum('i v, n v -> n i', velo2im, velo)
        # use minus 1 to get the exact same value as KITTI matlab code
        velo_pts_im[:, :2] = np.round(velo_pts_im[:, :2] / velo_pts_im[:, 2,None])
        
        lidar_padding = 0 if self.lidar_padding is None else self.lidar_padding
        velo_pts_im[:,0] += lidar_padding + 1224 - im_shape[1]
        velo_pts_im[:,1] += 368 - im_shape[0]

        # check if in bounds
        val_inds = ((velo_pts_im[:, 0] >= 0) #
                    & (velo_pts_im[:, 1] >= 0) #
                    & (velo_pts_im[:, 0] < 1224 + 2 * lidar_padding) #
                    & (velo_pts_im[:, 1] < 368) #
                    & (velo_pts_im[:, 2] > 0))  # positive depth

        # project to image
        sparse_depth_map = np.zeros((368, 1224 + 2 * lidar_padding), np.float)
        intensity_map = np.zeros((368, 1224 + 2 * lidar_padding), np.float)
        velo_pts_im = velo_pts_im[val_inds, :]
        intensity = intensity[val_inds]

        for idx in range(velo_pts_im.shape[0]):
            px = int(velo_pts_im[idx, 0])
            py = int(velo_pts_im[idx, 1])
            depth = velo_pts_im[idx, 2]
            if sparse_depth_map[py, px] == 0 or sparse_depth_map[py, px] > depth:
                # for conflicts, use closer point
                sparse_depth_map[py, px] = depth
                intensity_map[py,px] = intensity[idx]

        return sparse_depth_map, intensity_map

    def project_velo_points_to_velo_grid(self, item):
        min_col, max_col = 744 - self.lidar_padding, 1256 + self.lidar_padding
        h, w = 48, max_col - min_col

        velo_filename, velo2im, im_shape, velo_start2end = self.get_velo_data(item)
        velo = np.fromfile(velo_filename, dtype=np.float32).reshape(-1, 4) # forward, left, up, intensity
        forward = velo[:, 0] >= 0
        velo = velo[forward, :] # forward only

        cols = get_lidar_cols(velo)
        front = (cols >= min_col) & (cols < max_col)
        velo, cols = velo[front], cols[front] - min_col
        rows = get_lidar_rows(velo)
        upper = rows < h
        velo, rows, cols = velo[upper], rows[upper], cols[upper]
        d = np.linalg.norm(velo[:,:3], ord=2,axis=1)
        
        lidar_proj = np.zeros((h,w))
        lidar_proj[rows, cols] = d
        lidar_proj_i = np.zeros((h,w))
        lidar_proj_i[rows, cols] = velo[:,3]

        # linearly interpolate angles to calculate directions for missing data
        # for rho, all cols should be similar so try those first
        # for phi, do the opposite
        rho = np.zeros((h, w))
        rho[rows, cols] = np.arcsin(velo[:,2] / d)
        rows_known = []
        rows_skipped = []
        for row in range(h):
            cols_known = np.arange(w)[lidar_proj[row,:] != 0]
            if len(cols_known) <= 1:
                rows_skipped.append(row)
            elif len(cols_known) == w:
                rows_known.append(row)
            else:
                rows_known.append(row)
                rhos_known = rho[row,cols_known]
                cols_interp = np.arange(w)[lidar_proj[row,:] == 0]
                interp = interp1d(cols_known, rhos_known, kind='linear', fill_value='extrapolate', assume_sorted=True)
                rho[row, cols_interp] = interp(cols_interp)
        if len(rows_skipped) > 0:
            rows_known = np.array(rows_known)
            rows_interp = np.array(rows_skipped)
            for col in range(w):
                rhos_known = rho[rows_known,col]
                interp = interp1d(rows_known, rhos_known, kind='linear', fill_value='extrapolate', assume_sorted=True)
                rho[rows_interp,col] = interp(rows_interp)

        phi = np.zeros((h, w))
        phi[rows, cols] = np.arctan2(velo[:,1],velo[:,0])
        cols_known = []
        cols_skipped = []
        for col in range(w):
            rows_known = np.arange(h)[lidar_proj[:,col] != 0]
            if len(rows_known) <= 1:
                cols_skipped.append(row)
            elif len(rows_known) == h:
                cols_known.append(row)
            else:
                cols_known.append(row)
                phis_known = phi[rows_known,col]
                rows_interp = np.arange(h)[lidar_proj[:,col] == 0]
                interp = interp1d(rows_known, phis_known, kind='linear', fill_value='extrapolate', assume_sorted=True)
                phi[rows_interp,col] = interp(rows_interp)
        if len(cols_skipped) > 0:
            cols_known = np.array(cols_known)
            cols_interp = np.array(cols_skipped)
            for row in range(h):
                phis_known = phi[row,cols_known]
                interp = interp1d(cols_known, phis_known, kind='linear', fill_value='extrapolate', assume_sorted=True)
                phi[row, cols_interp] = interp(cols_interp)

        dirs = np.stack((np.cos(phi) * np.cos(rho),
                       np.sin(phi) * np.cos(rho),
                       np.sin(rho)), axis=2)
        # override known elements to reduce back and forth conversion errors
        dirs[rows,cols,:] = velo[:,:3] / d[:,None]
               

        # untwist egomotion
        # [744, 1256) instead of [0, 2000) (assuming no lidar padding)
        offsets = np.arange(min_col * 0.0005 - 0.5, max_col * 0.0005 - 0.5, 0.0005)[:,None] * velo_start2end[None,:]
        
        dirs = np.einsum('h w v, i v -> h w i', dirs, velo2im[:,:3])
        offsets = np.einsum('w v, i v -> w i', offsets, velo2im[:,:3]) + velo2im[None,:,3]

        return lidar_proj, lidar_proj_i, dirs, offsets, im_shape


    def get_velo_data(self, item):
        split = (self.gt_depth_paths[item].split(self.setname)[1]).split('/')
        drive_dir = split[1]
        day_dir = drive_dir.split('_drive')[0]
        img_source_dir = split[4]
        img_idx_dir = split[5].split('.png')[0]
        cam = img_source_dir.split('0')[1]

        calib_dir = day_dir = os.path.join(self.rgb_dir, day_dir)
        drive_dir = os.path.join(day_dir, drive_dir)
        velo_filename = os.path.join(drive_dir, 'velodyne_points', 'data', img_idx_dir) + ".bin"

        # load calibration files
        cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))

        velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
        velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
        velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))

        # get image shape
        im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)

        # compute projection matrix velodyne->image plane
        R_cam2rect = np.eye(4)
        R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
        P_rect = cam2cam['P_rect_0' + str(cam)].reshape(3, 4)
        velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

        return velo_filename, velo2im, im_shape, self.get_velo_start2end(calib_dir, drive_dir, img_idx_dir)

    def get_velo_start2end(self, calib_dir, drive_dir, img_idx_dir):

        timestamp_start_filename = os.path.join(drive_dir, 'velodyne_points', 'timestamps_start.txt')
        timestamp_end_filename = os.path.join(drive_dir, 'velodyne_points', 'timestamps_end.txt')

        with open(timestamp_start_filename, 'r') as f:
            for i in range(int(img_idx_dir)):
                f.readline()
            timestamp_start = timestamp2sec(f.readline()[:-1])
        with open(timestamp_end_filename, 'r') as f:
            for i in range(int(img_idx_dir)):
                f.readline()
            timestamp_end = timestamp2sec(f.readline()[:-1])

        if drive_dir in self.poses:
            timestamps, velo2world = self.poses[drive_dir]
        else:
            imu2velo = read_calib_file(os.path.join(calib_dir, 'calib_imu_to_velo.txt'))
            imu2velo = np.hstack((imu2velo['R'].reshape(3, 3), imu2velo['T'][..., np.newaxis]))
            imu2velo = np.vstack((imu2velo, np.array([0, 0, 0, 1.0])))

            oxts_drive_dir = os.path.join(drive_dir, 'oxts')
            timestamps, velo2world = oxts_2_poses(oxts_drive_dir, imu2velo)
            self.poses[drive_dir] = timestamps, velo2world

        # find poses bracing the true movement
        start_idx = 0
        while timestamps[start_idx + 1] < timestamp_start:
            start_idx+=1
        end_idx = start_idx + 1
        while timestamps[end_idx] < timestamp_end:
            end_idx+=1

        # calculate movement from start to end pose from both perspectives
        # output = weighted mean of both, with weights corresponding to how close the respective timestamps are to the middle of the scan
        # ignore rotation for easier interpolation
        timestamp_mid = (timestamp_start + timestamp_end) / 2
        w_start = (timestamps[end_idx] - timestamp_mid) / (timestamps[end_idx] - timestamps[start_idx])
        velo_start2end = w_start * np.matmul(np.linalg.inv(velo2world[start_idx])[:3,:], velo2world[end_idx][:,3]).squeeze() + \
           (w_start - 1) * np.matmul(np.linalg.inv(velo2world[end_idx])[:3,:], velo2world[start_idx][:,3]).squeeze()
        return velo_start2end / (timestamps[end_idx] - timestamps[start_idx]) * (timestamp_end - timestamp_start)
    

def project_velo_grid_to_image(lidar_padding, d, cd, x, dirs, offsets, im_shape, filter_occlusion=True):

    # reproject depth 
    velo_im = (d[:,0,:,:,None] * dirs + offsets[:,None,:,:])    
    cd = cd[:,0,:,:]
    d = velo_im[...,2]
    shape = cd.shape
        
    # calculate image indices
    b = torch.arange(shape[0], dtype=torch.long, device=d.device)[:,None,None]
    h = (velo_im[...,1] / d).detach().round().long() + 368 - im_shape[:,0,None,None]
    w = (velo_im[...,0] / d).detach().round().long() + lidar_padding + 1224 - im_shape[:,1,None,None]
    velo_im = torch.stack(torch.broadcast_tensors(b, h, w), 0)
    d = d.float()
        
    # filter visible points
    val_inds = ((w >= 0) & (w < 1224 + 2 * lidar_padding)  #
                & (h >= 0) & (h < 368) #
                & (d > 0) & (cd > 0))
    
    
    # filter occluded points
    if filter_occlusion:
        # for each 2 by 2 square of valid pixels in the lidar grid, project them into image perspective and observe the covered area
        # a depth value in this area is valid, if its depth is <= the maximum depth of the four pixels 
        # occluded pixels fall inside the area of occluding pixels and will be filtered out

        # to implement this efficiently
        # 1) define the covered area by its the outer rect
        # 2) use max depth of 4x4 pixels to filter less points by mistake because of approximation errors
        left, right, top, bot, valid_quads, max_d = -torch.max_pool2d(-w.float(), 2, 1, 0), torch.max_pool2d(w.float(), 2, 1, 0).long(), \
            -torch.max_pool2d(-h.float(), 2, 1, 0), torch.max_pool2d(h.float(), 2, 1, 0).long(), \
           (val_inds[...,1:,1:] & val_inds[...,:-1,1:] & val_inds[...,1:,:-1] & val_inds[...,:-1,:-1]), torch.max_pool2d(d.detach(), 4, 1, 1)

            
        for batch in range(left.shape[0]):  
            # 3) project into the bottom right corner of this rectangle, represent the full rectangle via width and height channels instead of projecting into each pixel
            # 4) in case multiple sets of points project into the same pixel, let closer points overwrite distant ones by sorting by descending depth beforehand
            #    sort is partially parallelizable, sequential minimum depth calculations are not
            window = (17,21)
            max_ds = torch.zeros((3, 368 + window[0] - 1, 1224 + 2 * lidar_padding + window[1] - 1),device= dirs.device)
            max_ds[2,...].fill_(np.inf)
            val_ = valid_quads[batch,...]
            max_d_, sorted = torch.sort(max_d[batch,val_], descending=True)
            left_, right_, top_, bot_ = left[batch,val_][sorted], right[batch,val_][sorted], top[batch,val_][sorted], bot[batch,val_][sorted]
            max_ds[:,bot_, right_] = torch.stack((right_ - left_, bot_ - top_, max_d_), 0)
        
            # 5) for each point to test, observe a window to the bottom right (approximation which can miss larger areas)
            #    don't filter the point if all areas in this window are too small or have a higher depth
            max_ds_window = torch.nn.Unfold(window)(max_ds.unsqueeze(0)).view(3, window[0], window[1], 368, 1224 + 2 * lidar_padding)[..., h[batch, val_inds[batch,...]], w[batch, val_inds[batch,...]]].unbind(0)
            val_inds[batch, val_inds[batch,...].clone()] = ((max_ds_window[0] < torch.arange(window[0],device=dirs.device)[:,None,None]) 
                                                            | (max_ds_window[1] < torch.arange(window[1], device=dirs.device)[None,:,None])
                                                            | (max_ds_window[2] >= d[batch, val_inds[batch,...]][None,None,:])).all(1).all(0)
               
            # there might be more efficient aproaches (e.g. using KD trees, doing the calculation on cpu in c++ without approximation, ...) i have not looked into; it should not take as long as it does
                    
    velo_im, d, cd = velo_im[:,val_inds], d[val_inds], cd[val_inds]
    if x is not None:
        x = rearrange(x, 'b c h w -> b h w c')[val_inds,:]

    # currently, the gradient is only propagated through the depth itself
    # to propagate via signal locations, project each point to 4 locations, using floor and ceil of h and w (still detached) and multiply cd by weights
    # cd_(i, floor, floor) = cd_i * (1 + floor(w_i) -w_i) * (1 + floor(h_i) - h_i)

    # for duplicate indices torch will sum the values
    # this is used to handle occlusion via an 1x1 nconv instead of hardcoding it => use dcd
    # the model is expected to predict low confidences for occluded pixels and receives a better signal this way
    # during inference this could be handled differently
    dcd = torch.sparse_coo_tensor(indices=velo_im, values=d * cd, size=(shape[0], 368, 1224 + 2 * lidar_padding)).to_dense()[:,None,:,:]
    if x is not None:
        xcd = torch.sparse_coo_tensor(indices=velo_im, values=x * cd[...,None], size=(shape[0], 368, 1224 + 2 * lidar_padding, x.shape[-1])).to_dense()
    cd = torch.sparse_coo_tensor(indices=velo_im, values=cd,     size=(shape[0], 368, 1224 + 2 * lidar_padding)).to_dense()[:,None,:,:]
    if x is not None:
        x = rearrange(xcd, 'b h w c -> b c h w') / (cd + 1e-20)
    d = dcd / (cd + 1e-20)
    
    # rescale confidences
    cd = cd / reduce(cd.detach(), 'b c h w -> b 1 1 1', 'max')

    return d, cd, x

def timestamp2sec(timestamp):
    # ignore date
    timestamp = timestamp.split(' ')[1]
    h, min, sec = timestamp.split(':')
    return (int(h) * 60 + int(min)) * 60 + float(sec)

# based on convertOxtsToPose from kitti raw devkit
# because i only care about relative poses, i do not multiply by the inverse first pose
def oxts_2_poses(oxts_drive_dir, imu2velo):
    timestamps = [timestamp2sec(x[:-1]) for x in open('{}/timestamps.txt'.format(oxts_drive_dir),'r').readlines()]

    velo2worlds = []
    velo2imu = np.linalg.inv(imu2velo)

    oxt_labels = ['lat',  'lon',  'alt',  'roll',  'pitch',  'yaw']#,  'vn',  've',  'vf', 'vl',  'vu',  'ax',  'ay',  'az',  'af',  'al',  'au',  'wx',  'wy',  'wz',  'wf', 'wl',  'wu',  'posacc',  'velacc',  'navstat',  'numsats',  'posmode',  'velmode',  'orimode']
    for idx in range(len(timestamps)):
        oxt_filename = '{}/data/{:010d}.txt'.format(oxts_drive_dir, idx)
        oxt_datapoints = open(oxt_filename,'r').readline().split(' ')
        oxt_datapoints[-1] = oxt_datapoints[-1][:-1]
        oxt_data = {}
        for i in range(len(oxt_labels)):
            oxt_data[oxt_labels[i]] = float(oxt_datapoints[i])

        if idx == 0:
            scale = np.cos(oxt_data['lat'] * np.pi / 180.0)

        T = np.zeros((4,4))

        er = 6378137
        T[:,3] = [scale * er * oxt_data['lon'] * np.pi / 180,
                  scale * er * np.log(np.tan((90 + oxt_data['lat']) * np.pi / 360)),
                  oxt_data['alt'],
                  1]

        rx = oxt_data['roll']
        ry = oxt_data['pitch']
        rz = oxt_data['yaw']
        Rx = np.array([[1, 0, 0,],
                      [0, np.cos(rx), -np.sin(rx)],
                      [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                     [0, 1, 0],
                     [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                     [np.sin(rz), np.cos(rz), 0],
                     [0, 0, 1]])
        T[:3,:3] = np.matmul(np.matmul(Rz,Ry),Rx)

        if idx == 0:
            T0inv = np.linalg.inv(T)
            velo2worlds.append(velo2imu)
        else:
            velo2worlds.append(np.matmul(np.matmul(T0inv, T), velo2imu))
    return timestamps, velo2worlds

# adapted from https://github.com/dendisuhubdy/kitti_scan_unfolding/blob/master/unfolding/scan.py
def get_lidar_cols(points: np.array, num_cols: int=2000) -> np.array:
    """ Returns the grid indices for unfolding one or more raw KITTI scans """
    azi = np.arctan2(points[..., 1], points[..., 0])
    cols = num_cols * (0.5 - 0.5 * azi / np.pi)
    return np.int32(np.minimum(cols, num_cols - 1))

# adapted from https://github.com/dendisuhubdy/kitti_scan_unfolding/blob/master/unfolding/scan.py
def get_lidar_rows(points: np.array, threshold: float=-0.005) -> np.array:
    azimuth_flipped = -np.arctan2(points[..., 1], -points[..., 0])
    azi_diffs = azimuth_flipped[..., 1:] - azimuth_flipped[..., :-1]

    rows = np.zeros(points.shape[:-1], dtype=np.int32)
    rows[..., 1:] = np.cumsum(threshold > azi_diffs, axis=-1)
    return rows

def read_calib_file(filepath):
    ''' Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    '''
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data