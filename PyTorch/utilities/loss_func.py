"""
 Loss function
 Copyright (c) 2019 Samsung Electronics Co., Ltd. All Rights Reserved
 If you use this code, please cite the following paper:
 Mahmoud Afifi and Michael S Brown. Deep White-Balance Editing. In CVPR, 2020.
"""
__author__ = "Mahmoud Afifi"
__credits__ = ["Mahmoud Afifi"]

import torch
from skimage.metrics import structural_similarity as ssim

from simple_camera_pipeline.python.pipeline import run_pipeline_v2, get_metadata
import cv2
import colour
import numpy as np
import os

# Get the directory of the current Python file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the new file path
dng_path = os.path.join(current_dir, "DSC01201_PSMS.dng")

# Use the new path in your function
metadata = get_metadata(dng_path)


class mse_loss():

    @staticmethod
    def compute(output, target):
        loss = torch.sum(torch.square(output - target))/output.shape[0]
        return loss

class psnr_per_image():
    @staticmethod
    def compute(output, target):
        mse_per_image = torch.sum(torch.square(output - target), dim=(1,2,3))/(output.shape[1]*output.shape[2]*output.shape[3])
        MAX = 13496
        psnrs_per_image = 10*(torch.log10(MAX**2/mse_per_image))
        return output.shape[0], torch.sum(psnrs_per_image), psnrs_per_image
    
class ssim_per_image():
    @staticmethod
    def compute(output, target):
        output = output.permute(0,2,3,1)
        target = target.permute(0,2,3,1)
    
        ssim_per_image = torch.zeros(output.shape[0])
        for i in range(output.shape[0]):
            ssim_per_image[i] = ssim(output[i].cpu().numpy(), target[i].cpu().numpy(), multichannel=True, channel_axis=2, data_range=13496)
        return output.shape[0], torch.sum(ssim_per_image), ssim_per_image
    
class deltae_per_image():
    @staticmethod
    def compute(output, target):
        output = output.permute(0,2,3,1)/13496
        target = target.permute(0,2,3,1)/13496
        #compute delta e
        deltae_per_image = torch.zeros(output.shape[0])
        imgs = torch.empty((output.shape[0], output.shape[1], output.shape[2], 3), dtype=torch.uint8)
        gts = torch.empty((output.shape[0], output.shape[1], output.shape[2], 3), dtype=torch.uint8)
        for i in range(output.shape[0]):
            img = output[i].cpu().numpy()
            params = {"input_stage": "demosaic", "output_stage": "tone"}
            img = run_pipeline_v2(img, params, metadata)
            #clip 
            img = np.clip(img, 0, 1)
            img = (img *255).astype(np.uint8)
            imgs[i] = torch.tensor(img[:,:,::-1].copy()) #RGB later need for FID
            gt = target[i].cpu().numpy()
            params = {"input_stage": "demosaic", "output_stage": "tone"}
            gt = run_pipeline_v2(gt, params, metadata)
            #clip
            gt = np.clip(gt, 0, 1)
            gt = (gt *255).astype(np.uint8)
            gts[i] = torch.tensor(gt[:,:,::-1].copy()) #RGB later for FID

            #delta e between gt and img
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            gt_lab = cv2.cvtColor(gt, cv2.COLOR_BGR2Lab)
            deltae_per_image[i] = np.mean(colour.delta_E(img_lab, gt_lab))

        #make imgs and gts channel second dimentions
        imgs = imgs.permute(0,3,1,2).to(output.device)
        gts = gts.permute(0,3,1,2).to(output.device)
        return output.shape[0], torch.sum(deltae_per_image), deltae_per_image, imgs, gts
        


class mae_loss():
    @staticmethod
    def compute(output, target):
        loss = torch.sum(output - target)/output.shape[0]
        return loss
    
class mse_validation_loss():

    @staticmethod
    def compute(output, target):
        loss = torch.mean(torch.square(output - target))
        return loss

#I did this to allow faster computation of validation PSNR
class mse_loss_sum_across_batch():

    @staticmethod
    def compute(output, target):
        loss = torch.sum(torch.square(output - target))/(output.shape[1]*output.shape[2]*output.shape[3])
        return output.shape[0], loss
