#impleemntation from https://github.com/xingwz/End-to-End-JDNDMSR/blob/main/network/model.py

import torch.nn as nn
from .rcan import RCAN

class JDNDM_BACKBONE(nn.Module):

    def __init__(self, depth = 4, out_channels=3):
        super(JDNDM_BACKBONE, self).__init__()
        #TODO:missing kernel initializer
        self.out_channels = out_channels
        kernel_size = (3, 3)
        filters = 64
        self.relu = nn.ReLU()
        self.rcan = RCAN(n_resgroups=depth, n_resblocks=20, n_feats=filters)

        self.final_conv = nn.Conv2d(filters, filters, kernel_size, padding='same')

        self.upsample = nn.Conv2d(filters, filters, kernel_size, padding='same')

        self.output_conv = nn.Conv2d(filters, 3, kernel_size, padding='same')
    
    
    def forward(self, x):

        
        features = self.rcan(x)
    
        final_conv_out = self.relu(self.final_conv(features))

        #long skip
        final_conv_out = final_conv_out + x

        upsample_out = self.relu(self.upsample(final_conv_out))

        output = self.output_conv(upsample_out)

        return output, 

        # return model