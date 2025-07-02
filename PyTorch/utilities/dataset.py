"""
 Dataloader
 Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 If you use this code, please cite the following paper:
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import cv2
import numpy as np
import glob
import torch
from torch.utils.data import Dataset
import logging
import random
from utilities.utils import construct_mosaic
from patchify import patchify
import pickle
import os
import random
import torch.nn.functional as F


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, split, pattern, remosaic, iso=3200, patch_size=120, patch_num_per_image=1, max_trdata=0, hard_patches_percentile = 0, noisy_viz=False, dropout=0, mask_interpolation=False):
        self.imgs_dir = imgs_dir
        self.patch_size = patch_size
        self.patch_num_per_image = patch_num_per_image
        self.split = split
        self.pattern = pattern
        self.remosaic = remosaic
        self.iso = iso
        self.noisy_viz=noisy_viz
        self.dropout = dropout
        self.mask_interpolation = mask_interpolation
        assert split in ['train', 'test', 'val', 'mining', 'test_full_size', 'all'], "Unknown split"
        assert pattern in ['single', 'quad', 'nona', 'single_quad', 'single_quad_nona', 'quad_single', 'nona_single', 'random'], "Unknown pattern"
        assert not (remosaic and (pattern == "single")) #can't do remosaic if the pattern is single bayer.

        if pattern == "single":
            self.filter_size = [1]
        elif pattern == "quad":
            self.filter_size = [2]
        elif pattern == "nona":
            self.filter_size = [3]
        elif pattern == "single_quad":
            self.filter_size = [1,2]
        elif pattern == "single_quad_nona":
            self.filter_size = [1,2,3]
        elif pattern == "quad_single":
            self.filter_size = [2,1]
        elif pattern == "nona_single":
            self.filter_size = [3,1]

        #leave 18 out due to the feathers
        split_info ={"train": [3,4,5,6,9,10,11,12,13,14], "val": [1,2], "test": [7, 8, 15, 16, 17], "test_full_size": [7, 8, 15, 16, 17], 'mining':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]}
        

        logging.info('Loading images information...')

        self.imgfiles = []
        if iso in [100, 6400, 12800, 100000]:
            for scene in split_info[split]:
                self.imgfiles.extend(glob.glob(f'{imgs_dir}/Scene{scene}/*/*[!_q]_gt.npy'))
        elif iso == 400:
            for scene in split_info[split]:
                self.imgfiles.extend(glob.glob(f'{imgs_dir}/Scene{scene}/*/*[!_q]_iso400.npy'))
        elif iso == 800:
            for scene in split_info[split]:
                self.imgfiles.extend(glob.glob(f'{imgs_dir}/Scene{scene}/*/*[!_q]_iso800.npy'))
        elif iso == 1600:
            for scene in split_info[split]:
                self.imgfiles.extend(glob.glob(f'{imgs_dir}/Scene{scene}/*/*[!_q]_iso1600.npy'))
        elif iso==3200:
            for scene in split_info[split]:
                self.imgfiles.extend(glob.glob(f'{imgs_dir}/Scene{scene}/*/*[!_q]_iso3200.npy'))
   

        if hard_patches_percentile != 0:
            # open a file, where you stored the pickled data
            hardpatches_file = os.path.join(imgs_dir, f'hardpatches{hard_patches_percentile:.2f}.pkl')
            with open(hardpatches_file, 'rb') as f:
                # dump information to that file
                hard_patches = set(pickle.load(f))

                #filter so I only keep hard_patches
                hard_patch_list = []
                for img in self.imgfiles:
                    scene_view = img.split('/')[-3:-1]
                    scene_view = '/'.join(scene_view)
                    img_row_col = img.split('/')[-1].split('_')[0:5]
                    img_row_col = '_'.join(img_row_col)
                    if f'{scene_view}/{img_row_col}' in hard_patches:
                            hard_patch_list.append(img)
                
                self.imgfiles = hard_patch_list
        
        self.imgfiles = sorted(self.imgfiles) #sort the files by name

        if split == "train" or split == "val": #this shuffle will be the same because I set the seed somewhere else
            random.shuffle(self.imgfiles)

        if max_trdata != 0 and len(self.imgfiles) > max_trdata:
            self.imgfiles = self.imgfiles[0:max_trdata]

        logging.info(f'Creating {split} dataset with {len(self.imgfiles)} examples')


    def __len__(self):
        return len(self.imgfiles)

    @classmethod
    def preprocess(cls, img, patch_size, patch_coords, remosaic, pattern):

        if patch_size != -1: #val/test is -1
            img = img[patch_coords[0]:patch_coords[0]+patch_size, patch_coords[1]:patch_coords[1]+patch_size, :]

        if remosaic and pattern == "quad": 
            #THESE ASSUME RGGB for quad
            remosaic_img = torch.empty(img.shape)
            remosaic_img[0::4,0::4] = img[0::4,0::4] #0,0 = R0,0
            remosaic_img[0::4,1::4] = img[0::4,2::4] #0,1 = G0,2
            remosaic_img[0::4,2::4] = img[0::4,1::4] #0,2 = R0,1
            remosaic_img[0::4,3::4] = img[0::4,3::4] #0,3 = G0,3

            remosaic_img[1::4,0::4] = img[2::4,0::4] #1,0 = G2,0
            remosaic_img[1::4,1::4] = img[2::4,2::4] #1,1 = B2,2
            remosaic_img[1::4,2::4] = img[1::4,2::4] #1,2 = G1,2
            remosaic_img[1::4,3::4] = img[2::4,3::4] #1,3 = B2,3

            remosaic_img[2::4,0::4] = img[1::4,0::4] #2,0 = R1,0
            remosaic_img[2::4,1::4] = img[2::4,1::4] #2,1 = G2,1
            remosaic_img[2::4,2::4] = img[1::4,1::4] #2,2 = R1,1
            remosaic_img[2::4,3::4] = img[1::4,3::4] #2,3 = G1,3

            remosaic_img[3::4,0::4] = img[3::4,0::4] #3,0 = G3,0
            remosaic_img[3::4,1::4] = img[3::4,2::4] #3,1 = B3,2
            remosaic_img[3::4,2::4] = img[3::4,1::4] #3,2 = G3,1
            remosaic_img[3::4,3::4] = img[3::4,3::4] #3,3 = B3,3
            img = remosaic_img

        elif remosaic and pattern == "nona":
            #THESE ASSUME RGGB for NONA
            #Remosaicing can be done arbirtrarily (I chose something reasonable)
            remosaic_img = torch.empty(img.shape)
            remosaic_img[0::6,0::6] = img[0::6,0::6] #0,0 = R0,0
            remosaic_img[0::6,1::6] = img[0::6,3::6] #0,1 = G0,3
            remosaic_img[0::6,2::6] = img[0::6,1::6] #0,2 = R0,1
            remosaic_img[0::6,3::6] = img[0::6,4::6] #0,3 = G0,4
            remosaic_img[0::6,4::6] = img[0::6,2::6] #0,4 = R0,2
            remosaic_img[0::6,5::6] = img[0::6,5::6] #0,5 = G0,5

            remosaic_img[1::6,0::6] = img[3::6,0::6] #1,0 = G3,0
            remosaic_img[1::6,1::6] = img[3::6,3::6] #1,1 = B3,3
            remosaic_img[1::6,2::6] = img[1::6,3::6] #1,2 = G1,3
            remosaic_img[1::6,3::6] = img[3::6,4::6] #1,3 = B3,4
            remosaic_img[1::6,4::6] = img[1::6,4::6] #1,4 = G1,4
            remosaic_img[1::6,5::6] = img[3::6,5::6] #1,5 = B3,5

            remosaic_img[2::6,0::6] = img[1::6,0::6] #2,0 = R1,0
            remosaic_img[2::6,1::6] = img[3::6,1::6] #2,1 = G3,1
            remosaic_img[2::6,2::6] = img[1::6,1::6] #2,2 = R1,1
            remosaic_img[2::6,3::6] = img[2::6,3::6] #2,3 = G2,3
            remosaic_img[2::6,4::6] = img[1::6,2::6] #2,4 = R1,2
            remosaic_img[2::6,5::6] = img[1::6,5::6] #2,5 = G1,5

            remosaic_img[3::6,0::6] = img[4::6,0::6] #3,0 = G4,0
            remosaic_img[3::6,1::6] = img[4::6,3::6] #3,1 = B4,3
            remosaic_img[3::6,2::6] = img[3::6,2::6] #3,2 = G3,2
            remosaic_img[3::6,3::6] = img[4::6,4::6] #3,3 = B4,4
            remosaic_img[3::6,4::6] = img[2::6,4::6] #3,4 = G2,4
            remosaic_img[3::6,5::6] = img[4::6,5::6] #3,5 = B4,5

            remosaic_img[4::6,0::6] = img[2::6,0::6] #4,0 = R2,0
            remosaic_img[4::6,1::6] = img[4::6,1::6] #4,1 = G4,1
            remosaic_img[4::6,2::6] = img[2::6,1::6] #4,2 = R2,1
            remosaic_img[4::6,3::6] = img[4::6,2::6] #4,3 = G4,2
            remosaic_img[4::6,4::6] = img[2::6,2::6] #4,4 = R2,2
            remosaic_img[4::6,5::6] = img[2::6,5::6] #4,5 = G2,5

            remosaic_img[5::6,0::6] = img[5::6,0::6] #5,0 = G5,0
            remosaic_img[5::6,1::6] = img[5::6,3::6] #5,1 = B5,3
            remosaic_img[5::6,2::6] = img[5::6,1::6] #5,2 = G5,1
            remosaic_img[5::6,3::6] = img[5::6,4::6] #5,3 = B5,4
            remosaic_img[5::6,4::6] = img[5::6,2::6] #5,4 = G5,2
            remosaic_img[5::6,5::6] = img[5::6,5::6] #5,5 = B5,5
            img = remosaic_img
            
        

        #DONT DO FLIPS
        #if flip_op == 1:
        #    img = torch.flip(img, dims=(1,)) #horizontal flip
        #elif flip_op == 2:
        #    img = torch.flip(img, dims=(0,)) #vertical flip


        
        # HWC to CHW
        img_trans = img.permute((2, 0, 1))
        img_trans = img_trans / (13496)
            

        img_trans = (img_trans-0.5)/0.5

        return img_trans

    def __getitem__(self, i):
        

        gt_ext = 'gt.npy'
        img_file = self.imgfiles[i]
        in_img_npy = np.load(img_file).astype(np.uint16)

        in_img = torch.from_numpy(in_img_npy.astype(np.float32))



        if self.noisy_viz:
            in_img = in_img
        else:
            in_img_all_mosaics = torch.empty((in_img.shape[0], in_img.shape[1], len(self.filter_size)*4), dtype=in_img.dtype)
            for i, fs in enumerate(self.filter_size):
                pattern, mosaic = construct_mosaic(in_img, order="RGGB", filter_size=fs)
                pattern = pattern.long()


                if self.split == "train":
                    dropout = torch.rand((1))*self.dropout #randomly choose a value between 0 and self.dropout
                    rand = torch.rand(mosaic.shape)
                    mask = rand < dropout
                else: #if test
                    dropout = self.dropout
                    torch.manual_seed(hash(img_file))
                    rand = torch.rand(mosaic.shape[0]*mosaic.shape[1])
                    num_dead_pixels = int(dropout * mosaic.shape[0] * mosaic.shape[1])
                    values, indices = torch.topk(rand, num_dead_pixels) #always get this number of dead pixels
                    
                    rows = torch.div(indices, mosaic.shape[0], rounding_mode='trunc')
                    cols = torch.div(indices, mosaic.shape[1], rounding_mode='trunc')

                    mask = torch.zeros(mosaic.shape, dtype=torch.bool)
                    mask[rows, cols] = True

                    if self.mask_interpolation:
                        for dp_num in range(num_dead_pixels):
                            #compute a 7x7 filter (this will be the same for everything)
                            R = torch.arange(-3, 4)
                            C = torch.arange(-3, 4)

                            mC, mR = torch.meshgrid(C,R, indexing='ij')
                            gauss_sigma = 3
                            filter = torch.exp(-(((mR.type(torch.float32)) ** 2 + (mC.type(torch.float32)) ** 2) / (2 * gauss_sigma ** 2))) * (1 / (2 * np.pi * gauss_sigma ** 2))
                            #get pixels around my central dead pixel
                            mosaic_pad = F.pad(input=mosaic, pad=(3, 3, 3, 3), mode='constant', value=0)
                            crop = mosaic_pad[rows[dp_num]: rows[dp_num]+7, cols[dp_num]: cols[dp_num]+7]

                            #mask pixels that are outside row/col boundaries to fitler weight of 0
                            R = torch.arange(rows[dp_num]-3, rows[dp_num]+4)
                            C = torch.arange(cols[dp_num]-3,cols[dp_num] + 4)
                            mR, mC = torch.meshgrid(R,C, indexing='ij')

                            filter[mR<0] = 0
                            filter[mR>mosaic.shape[0]-1] = 0
                            filter[mC<0] = 0
                            filter[mC>mosaic.shape[1]-1] = 0
                            #set filter weights for things that are not the same color to 0
                            
                            color = pattern[rows[dp_num], cols[dp_num], 0]

                            mR = torch.clamp(mR, 0, mosaic.shape[0] - 1)
                            mC = torch.clamp(mC, 0, mosaic.shape[1] - 1)
                            
                            filter[pattern[mR,mC,0] != color] = 0

                            #set other dead pixels filter to 0 (don't put other dead pixels in the filter), because I am updating the mosaic directly
                            #if this works this will also set the filter weight of the central pixel
                            for odp_num in range(num_dead_pixels):
                                if (rows[odp_num]-rows[dp_num] >= -3 and rows[odp_num]-rows[dp_num] <= 3):
                                    if (cols[odp_num]-cols[dp_num] >= -3 and cols[odp_num]-cols[dp_num] <=3):
                                        filter[rows[odp_num]-rows[dp_num]+3, cols[odp_num]-cols[dp_num]+3] = 0

                            #normalize filter

                            filter = filter/torch.sum(filter)

                            #compute interpolated color

                            out_color = torch.sum(crop*filter)
                            mosaic[rows[dp_num], cols[dp_num]] = out_color

                if not self.mask_interpolation:
                    mosaic[mask] = 0 #kill these pixels
                    
                pattern = pattern[:,:,0]
                if self.mask_interpolation:
                    R_pattern = (pattern == 0).long()*13496
                    G_pattern = (pattern == 1).long()*13496
                    B_pattern = (pattern == 2).long()*13496
                else:
                    R_pattern = torch.logical_and((pattern == 0), ~mask).long()*13496
                    G_pattern = torch.logical_and((pattern == 1), ~mask).long()*13496
                    B_pattern = torch.logical_and((pattern == 2), ~mask).long()*13496
                in_img_all_mosaics[:,:, i*4] = mosaic
                in_img_all_mosaics[:,:, i*4+1] = R_pattern
                in_img_all_mosaics[:,:, i*4+2] = G_pattern
                in_img_all_mosaics[:,:, i*4+3] = B_pattern
            in_img = in_img_all_mosaics

        # get image size
        w, h, _= in_img.shape

        # get ground truth images
        parts = img_file.split('_')
        base_name = ''
        for k in range(len(parts) - 1):
            base_name = base_name + parts[k] + '_'
        gt_file = base_name + gt_ext
        gt_img_npy = np.load(gt_file).astype(np.uint16)
        gt_img = torch.from_numpy(gt_img_npy.astype(np.float32))

        # if self.iso == 100000:
        #     gt_img = gt_img/low_light_scale # low light


        if self.split == "train" or self.split == "all":
            patch_x = 0
            patch_y = 0

            in_img_patches = self.preprocess(in_img, self.patch_size, (patch_x, patch_y), self.remosaic, self.pattern)
            gt_img_patches = self.preprocess(gt_img, self.patch_size, (patch_x, patch_y), False, self.pattern)

            return {'image': in_img_patches, 'gt': gt_img_patches, 'img_file': img_file}
        elif self.split == "test_full_size":
            in_img = self.preprocess(in_img, -1, (-1, -1), self.remosaic, self.pattern)
            gt_img = self.preprocess(gt_img, -1, (-1, -1), False, self.pattern) 
            return {'image': in_img, 'gt': gt_img, 'img_file': img_file}
        else:
            
            in_img = self.preprocess(in_img, -1, (-1, -1), self.remosaic, self.pattern)
            gt_img = self.preprocess(gt_img, -1, (-1, -1), False, self.pattern)
            return {'image': in_img, 'gt': gt_img, 'img_file': img_file}
