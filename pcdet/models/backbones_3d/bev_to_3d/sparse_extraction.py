import torch
import torch.nn as nn


class SparseExtraction(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        # self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        # (M, 4)[b_idx, z, x, y]. z always zero, i.e., assume pillar
        all_voxel_coords = batch_dict["voxel_coords"] 
        
        # (B, C, X, Y). BEV features
        spatial_features = batch_dict["spatial_features_2d"]
        
        # [batch_idx, y coord, x coord]. indices of orignal voxels
        slices = [all_voxel_coords[:, i].long() for i in [0, 2, 3]]

        # Sparse extraction: extract pixels that correspond to original voxels.  
        all_pyramid_voxel_features = spatial_features.permute(0, 2, 3, 1)[slices]
        batch_dict['point_features'] = all_pyramid_voxel_features
        return batch_dict

class SparseExtractionPointNet(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.in_channel = self.model_cfg.IN_CHANNEL
        self.out_channel = self.model_cfg.OUT_CHANNEL
        self.fc = nn.Sequential(
            nn.Linear(self.in_channel, self.out_channel),
            nn.BatchNorm1d(self.out_channel),
            nn.ReLU()
        )

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        # (M, 4)[b_idx, z, x, y]. z always zero, i.e., assume pillar
        all_voxel_coords = batch_dict["voxel_coords"] 
        
        # (B, C, X, Y). BEV features
        spatial_features = batch_dict["spatial_features_2d"]
        
        # [batch_idx, y coord, x coord]. indices of orignal voxels
        slices = [all_voxel_coords[:, i].long() for i in [0, 2, 3]]

        # Sparse extraction: extract pixels that correspond to original voxels.  
        all_pyramid_voxel_features = spatial_features.permute(0, 2, 3, 1)[slices]

        # point wise features 
        pp_feature = batch_dict["voxel_features"] # (M, D)
        x = torch.cat([pp_feature, all_pyramid_voxel_features], dim=1)
        x = self.fc(x)
        batch_dict['point_features'] = x
        return batch_dict