"""
Copyright (c) 2025 Samsung Electronics Co., Ltd.

Author(s):
SaiKiran Tedla
Abhijith Punnappurath
Luxi Zhao
Michael S. Brown

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""


import torch.nn as nn
import torch
from .jdndm_backbone import JDNDM_BACKBONE

class SIMP_MULTI_JD(nn.Module):

    def __init__(self, out_channels=3):
        super(SIMP_MULTI_JD, self).__init__()
        #TODO:missing kernel initializer
        filters = 64
        kernel_size = (3,3)
        self.in_channels = 12
        self.out_channels = out_channels

        self.first_conv = nn.Conv2d(4, filters, kernel_size, padding='same')
        self.relu = nn.ReLU()

        self.first_conv_upsample = nn.Conv2d(filters, filters, kernel_size, padding='same') #not really an upsample

        self.relu = nn.ReLU()
        self.backbone = JDNDM_BACKBONE(depth=2)

    def forward(self, x):


       
        mosaic_pattern_single = x[:,0:4,:,:]
        mosaic_pattern_quad = x[:,4:8,:,:]
        mosaic_pattern_nona = x[:,8:12,:,:]

        combined_in = torch.cat((mosaic_pattern_single, mosaic_pattern_quad, mosaic_pattern_nona), dim = 0)
        
        first_conv_out = self.relu(self.first_conv(combined_in))
        all_head_out = self.relu(self.first_conv_upsample(first_conv_out))

        out, = self.backbone(all_head_out)

        num_imgs = x.shape[0]
        return out[0:num_imgs], out[num_imgs*1:num_imgs*2], out[num_imgs*2:num_imgs*3] #out_single, out_quad, out_nona
