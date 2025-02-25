from functools import partial
import random
import numpy as np
import torch
import torch.nn as nn
import kornia

from ...utils.spconv_utils import replace_feature, spconv
from ...utils import common_utils
from ...utils import loss_utils
from .spconv_backbone import post_act_block


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        # bias = norm_fn is not None
        # bias = False
        bias = True
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out

class BEV_MAE_res_expand(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        self.mask_ratio = model_cfg.MASKED_RATIO
        self.grid = model_cfg.GRID
        
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.base_channel = 16
        self.width_mul = model_cfg.WIDTH_MUL
        self.num_blocks = model_cfg.NUM_BLOCKS

        self.out_input = self.base_channel * self.width_mul
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, self.out_input, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(self.out_input),
            nn.ReLU(),
        )
        block = post_act_block

        ## Level 1
        self.out_1 = self.out_input
        block_list = []
        n_block = self.num_blocks[0]
        block_list.append(SparseBasicBlock(self.out_input, self.out_1, norm_fn=norm_fn, indice_key='res1'))
        for _ in range(n_block):
            block_list.append(SparseBasicBlock(self.out_1, self.out_1, norm_fn=norm_fn, indice_key='res1'))
        self.conv1 = spconv.SparseSequential(*block_list)
        

        self.out_2 = self.out_1 * 2
        block_list = []
        n_block = self.num_blocks[1]
        block_list.append(block(self.out_1, self.out_2, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'))
        for _ in range(n_block):
            block_list.append(SparseBasicBlock(self.out_2, self.out_2, norm_fn=norm_fn, indice_key='res2'))
        self.conv2 = spconv.SparseSequential(*block_list)
        


        self.out_3 = self.out_2 * 2
        block_list = []
        n_block = self.num_blocks[2]
        block_list.append(block(self.out_2, self.out_3, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'))
        for _ in range(n_block):
            block_list.append(SparseBasicBlock(self.out_3, self.out_3, norm_fn=norm_fn, indice_key='res3'))
        self.conv3 = spconv.SparseSequential(*block_list)
        


        self.out_4 = self.out_3 * 2
        block_list = []
        n_block = self.num_blocks[3]
        block_list.append(block(self.out_3, self.out_4, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'))
        for _ in range(n_block):
            block_list.append(SparseBasicBlock(self.out_4, self.out_4, norm_fn=norm_fn, indice_key='res4'))
        self.conv4 = spconv.SparseSequential(*block_list)
        

        self.out_last = self.out_4
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad) # if last pad doesnt exist then return 0
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(self.out_last, self.out_last, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.out_last),
            nn.ReLU(),
        )
        
        self.num_point_features = 16 # why? does not exist in this file          
        
        last_dim_z, last_dim_y, last_dim_x = self.get_last_dims(grid_size=grid_size[::-1])
        
        self.decoder = nn.Sequential(
            # nn.Conv2d(last_dim_z * 128, 256, 3, padding=1, stride=1),
            nn.Conv2d(last_dim_z * self.out_last, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.coor_conv = nn.Conv2d(256, 3*20, 1) # what is this 20? --> probably the number of points to predict. Chamfer distance can have differnet src dst #points.
        self.num_conv = nn.Conv2d(256, 1, 1)

        down_factor = 8 # final feature map is downsampled 8, so mask is also downsampled x8
        self.down_factor = down_factor
        self.unshuffle = torch.nn.PixelUnshuffle(down_factor)
        
        voxel_size = model_cfg.VOXEL_SIZE # model voxel size, not dataset.  
        point_cloud_range = model_cfg.POINT_CLOUD_RANGE  # the same as dataet one 
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        # point_cloud_range[0:3]: minx, miny, minz. defined in config
        self.x_offset = self.vx / 2 + point_cloud_range[0] # half voxel + the min range offsets that are used to move PC to origin. probably for spconv.
        self.y_offset = self.vy / 2 + point_cloud_range[1] # 
        self.z_offset = point_cloud_range[2]

        self.coor_loss = loss_utils.MaskChamferDistance()
        self.num_loss = nn.SmoothL1Loss(reduction='none', beta=1.0)

        self.mask_token = nn.Parameter(torch.zeros(1,3)) 

        self.forward_re_dict = {}

    def get_last_dims(self, grid_size):
        kernel_list = [
            (3,3,3),
            (3,3,3),
            (3,3,3),
            (3,1,1)
        ]
        padding_list = [
            (1,1,1),
            (1,1,1),
            (1,1,1),
            (0,0,0)
        ]
        stride_list = [
            (2,2,2),
            (2,2,2),
            (2,2,2),
            (2,1,1)
        ]
        def calculate_output_size(input_size, kernel_size, padding, stride):
            if isinstance(padding, tuple) and isinstance(stride, tuple) and isinstance(kernel_size, tuple):
                return tuple((input_size[i] + 2 * padding[i] - kernel_size[i]) // stride[i] + 1 for i in range(3))
            else:
                raise ValueError("Padding, stride, and kernel size must be tuples of length 3.")
        D, H, W = grid_size
        n_conv = len(kernel_list)
        for i in range(n_conv):
            D, H, W = calculate_output_size((D, H, W), kernel_list[i], padding_list[i], stride_list[i])
        return D, H, W

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        # pred = self.forward_re_dict['pred']
        # target = self.forward_re_dict['target']
        pred_coor = self.forward_re_dict['pred_coor']
        gt_coor = self.forward_re_dict['gt_coor'].detach()
        chamfer_mask = self.forward_re_dict['chamfer_mask'].detach()

        pred_num = self.forward_re_dict['pred_num']
        gt_num = self.forward_re_dict['gt_num'].detach()

        gt_mask = self.forward_re_dict['gt_mask'].detach()
        # loss = self.criterion(pred, target)
        
        loss_num = self.get_num_loss(pred_num, gt_num, gt_mask)

        loss_coor = self.get_coor_loss(pred_coor, gt_coor, gt_mask, chamfer_mask)

        loss = loss_num + loss_coor # added a weight to loss_num because it causes the exploding gradient.

        tb_dict = {
            'loss_num': loss_num.item(),
            'loss_coor': loss_coor.item(),
        }



        return loss, tb_dict
    
    def get_num_loss(self, pred, target, mask):
        bs = pred.shape[0]
        loss = self.num_loss(pred, target).squeeze()
        
        if bs == 1:
            loss = loss.unsqueeze(dim=0)
        assert loss.size() == mask.size(), f'{loss.size()} != {mask.size()}'
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
    
    def get_coor_loss(self, pred, target, mask, chamfer_mask):
        bs, d, _, h, w = target.shape
        target = target.reshape(bs, -1, h, w)
        target = target.permute(0, 2, 3, 1)

        pred = pred.permute(0, 2, 3, 1) 
        chamfer_mask = chamfer_mask.permute(0, 2, 3, 1)

        mask = mask.squeeze().bool()
        if bs == 1:
            mask = mask.unsqueeze(dim=0)

        pred = pred[mask]
        target = target[mask]

        chamfer_mask = chamfer_mask[mask]


        pred = pred.reshape(-1, 3, 20).permute(0, 2, 1)
        target = target.reshape(-1, d, 3)

        loss_source, loss_target = self.coor_loss(pred, target, chamfer_mask)

        loss = loss_source + loss_target
        return loss
    
    def decode_feat(self, feats, mask=None):
        # feats = feats[mask]
        if mask is not None:
            bs, c, h, w = feats.shape
            # print(mask.shape)
            mask_tokens = self.mask_token.view(1, -1, 1, 1).expand(bs, -1, h, w)
            w = mask.unsqueeze(dim=1).expand_as(mask_tokens)
            feats = feats + w * mask_tokens

        x = self.decoder(feats)
        bs, c, h, w = x.shape
        # x = feats
        coor = self.coor_conv(x)
        num = self.num_conv(x)
        # x = x.reshape(bs, )
        return coor, num


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        # voxel_features:(bs * max_point, c_in), coors: (bs * max_point, [batch_idx,z_idx, y_idx, x_idx]); num_points: (b * max_point, )
        voxel_features, coors, num_points = batch_dict['voxel_features'], batch_dict['voxel_coords'], batch_dict['voxel_num_points']
        # print(coors.shape)
        coor_down_sample = coors.int().detach().clone() # (N, (b, z, y, x))
        coor_down_sample = coors.int().clone() # (N, (b, z, y, x))
        coor_down_sample[:, 1:] = coor_down_sample[:, 1:]//(self.down_factor * self.grid) # draw a grid with (down_factor x grid)=(8x1) as unit. probably BEV cell size   
        coor_down_sample[:, 1] = coor_down_sample[:, 1]//(coor_down_sample[:, 1].max(dim=0)[0]*2) # all z made 0. projected to the ground.  
        
        # get a unique voxel for each BEV cell. In other words, occupied BEV cell. 
        # inverse index: an index that can reconstruct the original input.  
        # dim = 0 means each element along dim 0 (batch) is a unit to be compared (pytorch doc). So, the unique voxel is retained. 
        unique_coor_down_sample, inverse_index = torch.unique(coor_down_sample, return_inverse=True, dim=0) 

        select_ratio = 1 - self.mask_ratio # ratio for select bev voxel
        nums = unique_coor_down_sample.shape[0] # num of unique bev voxel. 
        
        len_keep = int(nums * select_ratio) # num of unique BEV cells to keep 

        noise = torch.rand(nums, device=voxel_features.device)  # noise in [0, 1] # random noise for creating a bev mask

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise) # a random mask for shuffling. noise random, so argsort noise is a random order.
        ids_restore = torch.argsort(ids_shuffle) # recover mask for a shuffled point by rearranding the indice

        keep = ids_shuffle[:len_keep] # random sampling from unique bev cells

        unique_keep_bool = torch.zeros(nums).to(voxel_features.device).detach() # unique bev cell zero mask
        unique_keep_bool[keep] = 1 # the kept bev cells assigned to 1. bev binary mask
        # unique_mask_bool = unique_mask_bool.bool()
        ids_keep = torch.gather(unique_keep_bool, 0, inverse_index) # propagate the bev binary mask to all bev poitns (coor_down_sample) (b * max_point,)
        ids_keep = ids_keep.bool() # from bianry to bool 

        ids_mask = ~ids_keep

        batch_size = batch_dict['batch_size']
        ### mask
        voxel_features_mask, voxel_coords_mask = voxel_features[ids_mask,:], coors[ids_mask,:] # bev masked original feature and coords

        voxel_fratures_mask_one = torch.ones(voxel_features_mask.shape[0],1).to(voxel_features_mask.device).detach() # 1 as feature for masked positions
        
        # a dense 3d tensor (entire grid defined by point cloud range and voxel size) that masked positions have one unmasked have zeros
        pts_mask = spconv.SparseConvTensor(
            voxel_fratures_mask_one,
            voxel_coords_mask.int(),
            self.sparse_shape,
            batch_size
        ).dense()  # (b, 1, d, w, h) 

        pts_mask = pts_mask.detach()
        # point_mask = pts_mask.clone()

        # pixel unshuffling: aggreate local neighbors to the single point. 
        # A kind of downsampling that does not lose info by moving nearby values to the channel dimension of the aggregated position. 
        pts_mask = self.unshuffle(pts_mask) # (b, 1, d * 8 * 8, w//8, h//8). 8: downsampling factor.
        # print(pts_mask.shape)
        
        # reduce channels by max. 
        # since the pts_mask is a binary dense mask, the resulting bev_mask (b, h, w) indicates the occupancy of the entire grid in the BEV plane. 
        # masked bev position has one, 0 elsewhere.
        bev_mask = pts_mask.squeeze(1).max(dim=1)[0] # (b, bev_w, bev_h)
        # bev_mask = bev_mask.max(dim=1)[0]
        
        # if len(bev_mask.shape) != 3:
        #     import pdb; pdb.set_trace()
        self.forward_re_dict['gt_mask'] = bev_mask
        
        #### gt num
        # dense tensor that has the number of point assigned to each voxel
        pts_gt_num = spconv.SparseConvTensor(
            num_points.view(-1, 1).detach(), # (b * max_point, 1) as feature
            coors.int(), # (b * max_point, 4)
            self.sparse_shape, # dense grid size (z, y, x)
            batch_size
        ).dense() # (b, 1, d, w, h). 
        bs, _, d, h, w = pts_gt_num.shape
        pts_gt_num = self.unshuffle(pts_gt_num.reshape(bs, d, h, w)) # downsample like above
        
        # sum the #point in a BEV cell and divided by the numebr unshuffled cells. 
        # essentially getting the average #point in local voxels, as a number of points inside a bev cell. 
        pts_gt_num = pts_gt_num.sum(dim=1, keepdim=True) / self.down_factor**2 
        # pts_gt_num = pts_gt_num / (torch.max(pts_gt_num.reshape(bs, -1), dim=1, keepdim=True)[0] + 1e-6) # sample wise normalization
        pts_gt_num = pts_gt_num.detach()
        self.forward_re_dict['gt_num'] = pts_gt_num

        ### input
        voxel_features_partial, voxel_coords_partial = voxel_features[ids_keep,:], coors[ids_keep,:] # kept orignal input

        average_features = self.mask_token.repeat(voxel_features_mask.size(0), 1) # masked tokens 
        
        # concat masked tokens to remaining featuers and voxel coords
        voxel_features_partial = torch.cat([voxel_features_partial, average_features], dim=0) # concat mask tokens. (b * num_sample, c_in)
        voxel_coords_partial = torch.cat([voxel_coords_partial, voxel_coords_mask], dim=0) # mask tokens have original grid positions (not feature).  

        # a sparse tensor whihc have orignal feature at unmasked position and maske token at masked positions
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features_partial,
            indices=voxel_coords_partial.int(),
            spatial_shape=self.sparse_shape, # grid size 
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor) # (b * num_sample, 16)
        x_conv1 = self.conv1(x) # (b * num_sample, 16)
        x_conv2 = self.conv2(x_conv1) # (downsampled x 2, 32)
        x_conv3 = self.conv3(x_conv2) # (downsampled x 4, 64)
        x_conv4 = self.conv4(x_conv3) # (downsampled x 8, 128)
        out = self.conv_out(x_conv4) # (xy same while z further downsampled x 16, 128)
        feats = out.dense() # converted to dense tensor. (bs, 128, d=grid_z / 16, w=grid_x/8, grid_y/8) 
        bs, c, d, h, w = feats.shape
        # depth and channel merged. in other words, depth is flattened to make a 2d feature map  
        feats = feats.reshape(bs, -1, h, w) # roughly z / 16 * 128 for channels

        pred_coor, pred_num = self.decode_feat(feats) # predict coordinates + number of points in the BEV cell
        self.forward_re_dict['pred_coor'] = pred_coor  # (b, 3 * 20, bev_h, bev_w) # prediction of 20 coordinates 
        self.forward_re_dict['pred_num'] = pred_num # (b, 1, bev_h, bev_w) # prediction of average number in a voxel
        # import pdb; pdb.set_trace()

        # original points/features in BEV plane extracted during preprocessing. these attributes are defined in the data_processor.py. 
        # voxels_large: (b * n_bev cell, max_point, c_in). coors_large: (b * n_bev, 4)
        voxels_large, num_points_large, coors_large = batch_dict['voxels_bev'], batch_dict['voxel_num_points_bev'], batch_dict['voxel_coords_bev'], 
        f_center = torch.zeros_like(voxels_large[:, :, :3]) # (n_bev, max_points_per_bev, 3). absolute xyz in a voxel position. 
        
        # make global coords relative to BEV cell
        f_center[:, :, 0] = (voxels_large[:, :, 0] - (coors_large[:, 3].unsqueeze(dim=1) * self.vx + self.x_offset)) / self.vx # (n_bev, k) - (n_bev, 1)
        f_center[:, :, 1] = (voxels_large[:, :, 1] - (coors_large[:, 2].unsqueeze(dim=1) * self.vy + self.y_offset)) / self.vy
        f_center[:, :, 2] = (voxels_large[:, :, 2]) / self.vz

        voxel_count = f_center.shape[1] # number of point per voxel
        mask_num = get_paddings_indicator(num_points_large, voxel_count, axis=0) # (b, max_num). boolean
        mask_num = torch.unsqueeze(mask_num, -1).type_as(f_center) # (b, max_num, 1). binary mask that marks the valid points within a voxel.
        f_center *= mask_num # zero those padded points (b * n_bev, max_num, 3)

        sparse_shape = [1, self.sparse_shape[1]//self.down_factor, self.sparse_shape[2]//self.down_factor,] # (1, bev_cell_w, bev_cell_h)

        chamfer_mask = spconv.SparseConvTensor(# 0 at padded points, 1 at real points 
            mask_num.squeeze().detach(), # (b * nbev, max_num)
            coors_large.int(), # (b * nbev, 4)
            sparse_shape, # # (1, bev_cell_w, bev_cell_h) # grid size z y x
            batch_size
        ).dense() # (b, max_num, 1, bev_w, bev_h). # valid mask for chamfer calcuation

        self.forward_re_dict['chamfer_mask'] = chamfer_mask.sum(dim=2)

        n, m, _ = f_center.shape
        f_center = f_center.reshape(n, -1)

        pts_gt_coor = spconv.SparseConvTensor(
            f_center.detach(), # (b * n_bev, max_num *3) flattend coord
            coors_large.int(), # (b * n_bev, 4) # voxel coord
            sparse_shape,
            batch_size
        ).dense() # (b, max_num*3, 1, n_bev_w, n_nev_h)

        bs, _, d, h, w = pts_gt_coor.shape
        pts_gt_coor = pts_gt_coor.reshape(bs, m, -1, h, w) # unflattened coord. (b, max_num, 3, bev_h, bev_w)
        self.forward_re_dict['gt_coor'] = pts_gt_coor

        vis_save_path = batch_dict.get('vis_save_path', None)
        if vis_save_path is not None:

            ### for visualizatio only!
            vis_dict = {
                "coors_large": coors_large,
                "f_center": f_center,
                "forward_re_dict": self.forward_re_dict,
                "mask_num": mask_num,
                "voxels_large": voxels_large,
                "x_offset": self.x_offset,
                "vx": self.vx,
                "vy": self.vy,
                "vz": self.vz,
                "sparse_shape": sparse_shape,
            }
            torch.save(vis_dict, vis_save_path)
            ### for visualizatio only!

        return batch_dict


class BEV_MAE_res_expand_no_density(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        self.mask_ratio = model_cfg.MASKED_RATIO
        self.grid = model_cfg.GRID
        
        self.sparse_shape = grid_size[::-1] + [1, 0, 0] 

        self.base_channel = 16
        self.width_mul = model_cfg.WIDTH_MUL
        self.num_blocks = model_cfg.NUM_BLOCKS

        self.out_input = self.base_channel * self.width_mul
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, self.out_input, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(self.out_input),
            nn.ReLU(),
        )
        block = post_act_block

        ## Level 1
        self.out_1 = self.out_input
        block_list = []
        n_block = self.num_blocks[0]
        block_list.append(SparseBasicBlock(self.out_input, self.out_1, norm_fn=norm_fn, indice_key='res1'))
        for _ in range(n_block):
            block_list.append(SparseBasicBlock(self.out_1, self.out_1, norm_fn=norm_fn, indice_key='res1'))
        self.conv1 = spconv.SparseSequential(*block_list)
        
        self.out_2 = self.out_1 * 2
        block_list = []
        n_block = self.num_blocks[1]
        block_list.append(block(self.out_1, self.out_2, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'))
        for _ in range(n_block):
            block_list.append(SparseBasicBlock(self.out_2, self.out_2, norm_fn=norm_fn, indice_key='res2'))
        self.conv2 = spconv.SparseSequential(*block_list)
        


        self.out_3 = self.out_2 * 2
        block_list = []
        n_block = self.num_blocks[2]
        block_list.append(block(self.out_2, self.out_3, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'))
        for _ in range(n_block):
            block_list.append(SparseBasicBlock(self.out_3, self.out_3, norm_fn=norm_fn, indice_key='res3'))
        self.conv3 = spconv.SparseSequential(*block_list)
        


        self.out_4 = self.out_3 * 2
        block_list = []
        n_block = self.num_blocks[3]
        block_list.append(block(self.out_3, self.out_4, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'))
        for _ in range(n_block):
            block_list.append(SparseBasicBlock(self.out_4, self.out_4, norm_fn=norm_fn, indice_key='res4'))
        self.conv4 = spconv.SparseSequential(*block_list)
        

        self.out_last = self.out_4
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad) # if last pad doesnt exist then return 0
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(self.out_last, self.out_last, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.out_last),
            nn.ReLU(),
        )
        
        self.num_point_features = 16 # why? does not exist in this file          
        
        last_dim_z, last_dim_y, last_dim_x = self.get_last_dims(grid_size=grid_size[::-1])
        
        self.decoder = nn.Sequential(
            # nn.Conv2d(last_dim_z * 128, 256, 3, padding=1, stride=1),
            nn.Conv2d(last_dim_z * self.out_last, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.coor_conv = nn.Conv2d(256, 3*20, 1) # what is this 20? --> probably the number of points to predict. Chamfer distance can have differnet src dst #points.
        

        down_factor = 8 # final feature map is downsampled 8, so mask is also downsampled x8
        self.down_factor = down_factor
        self.unshuffle = torch.nn.PixelUnshuffle(down_factor)
        # self.vx = voxel_size[0] * down_factor
        # self.vy = voxel_size[1] * down_factor
        # self.vz = voxel_size[2] * down_factor
        voxel_size = model_cfg.VOXEL_SIZE # model voxel size, not dataset.  
        point_cloud_range = model_cfg.POINT_CLOUD_RANGE  # the same as dataet one 
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        # point_cloud_range[0:3]: minx, miny, minz. defined in config
        self.x_offset = self.vx / 2 + point_cloud_range[0] # half voxel + the min range offsets that are used to move PC to origin. probably for spconv.
        self.y_offset = self.vy / 2 + point_cloud_range[1] # 
        self.z_offset = point_cloud_range[2]

        self.coor_loss = loss_utils.MaskChamferDistance()
        # self.num_loss = nn.SmoothL1Loss(reduction='none', beta=1.0)

        self.mask_token = nn.Parameter(torch.zeros(1,3)) 

        self.forward_re_dict = {}

    def get_last_dims(self, grid_size):
        kernel_list = [
            (3,3,3),
            (3,3,3),
            (3,3,3),
            (3,1,1)
        ]
        padding_list = [
            (1,1,1),
            (1,1,1),
            (1,1,1),
            (0,0,0)
        ]
        stride_list = [
            (2,2,2),
            (2,2,2),
            (2,2,2),
            (2,1,1)
        ]
        def calculate_output_size(input_size, kernel_size, padding, stride):
            if isinstance(padding, tuple) and isinstance(stride, tuple) and isinstance(kernel_size, tuple):
                return tuple((input_size[i] + 2 * padding[i] - kernel_size[i]) // stride[i] + 1 for i in range(3))
            else:
                raise ValueError("Padding, stride, and kernel size must be tuples of length 3.")
        D, H, W = grid_size
        n_conv = len(kernel_list)
        for i in range(n_conv):
            D, H, W = calculate_output_size((D, H, W), kernel_list[i], padding_list[i], stride_list[i])
        return D, H, W

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        # pred = self.forward_re_dict['pred']
        # target = self.forward_re_dict['target']
        pred_coor = self.forward_re_dict['pred_coor']
        gt_coor = self.forward_re_dict['gt_coor'].detach()
        chamfer_mask = self.forward_re_dict['chamfer_mask'].detach()

        # pred_num = self.forward_re_dict['pred_num']
        # gt_num = self.forward_re_dict['gt_num'].detach()

        gt_mask = self.forward_re_dict['gt_mask'].detach()
        # loss = self.criterion(pred, target)
        
        # loss_num = self.get_num_loss(pred_num, gt_num, gt_mask)

        loss_coor = self.get_coor_loss(pred_coor, gt_coor, gt_mask, chamfer_mask)

        # loss = loss_num + loss_coor # added a weight to loss_num because it causes the exploding gradient when AMP
        loss =  loss_coor 

        tb_dict = {
            'loss_coor': loss_coor.item(),
        }


        return loss, tb_dict
    
    def get_coor_loss(self, pred, target, mask, chamfer_mask):
        bs, d, _, h, w = target.shape
        target = target.reshape(bs, -1, h, w)
        target = target.permute(0, 2, 3, 1)

        pred = pred.permute(0, 2, 3, 1) 
        chamfer_mask = chamfer_mask.permute(0, 2, 3, 1)

        mask = mask.squeeze().bool()
        if bs == 1:
            mask = mask.unsqueeze(dim=0)

        pred = pred[mask]
        target = target[mask]

        chamfer_mask = chamfer_mask[mask]


        pred = pred.reshape(-1, 3, 20).permute(0, 2, 1)
        target = target.reshape(-1, d, 3)

        loss_source, loss_target = self.coor_loss(pred, target, chamfer_mask)

        loss = loss_source + loss_target
        return loss
    
    def decode_feat(self, feats, mask=None):
        # feats = feats[mask]
        if mask is not None:
            bs, c, h, w = feats.shape
            # print(mask.shape)
            mask_tokens = self.mask_token.view(1, -1, 1, 1).expand(bs, -1, h, w)
            w = mask.unsqueeze(dim=1).expand_as(mask_tokens)
            feats = feats + w * mask_tokens

        x = self.decoder(feats)
        bs, c, h, w = x.shape
        # x = feats
        coor = self.coor_conv(x)
        # x = x.reshape(bs, )
        return coor


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        # voxel_features:(bs * max_point, c_in), coors: (bs * max_point, [batch_idx,z_idx, y_idx, x_idx]); num_points: (b * max_point, )
        voxel_features, coors, num_points = batch_dict['voxel_features'], batch_dict['voxel_coords'], batch_dict['voxel_num_points']
        # print(coors.shape)
        coor_down_sample = coors.int().detach().clone() # (N, (b, z, y, x))
        coor_down_sample = coors.int().clone() # (N, (b, z, y, x))
        coor_down_sample[:, 1:] = coor_down_sample[:, 1:]//(self.down_factor * self.grid) # draw a grid with (down_factor x grid)=(8x1) as unit. probably BEV cell size   
        coor_down_sample[:, 1] = coor_down_sample[:, 1]//(coor_down_sample[:, 1].max(dim=0)[0]*2) # all z made 0. projected to the ground.  
        
        # get a unique voxel for each BEV cell. In other words, occupied BEV cell. 
        # inverse index: an index that can reconstruct the original input.  
        # dim = 0 means each element along dim 0 (batch) is a unit to be compared (pytorch doc). So, the unique voxel is retained. 
        unique_coor_down_sample, inverse_index = torch.unique(coor_down_sample, return_inverse=True, dim=0) 

        select_ratio = 1 - self.mask_ratio # ratio for select bev voxel
        nums = unique_coor_down_sample.shape[0] # num of unique bev voxel. 
        
        len_keep = int(nums * select_ratio) # num of unique BEV cells to keep 

        noise = torch.rand(nums, device=voxel_features.device)  # noise in [0, 1] # random noise for creating a bev mask

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise) # a random mask for shuffling. noise random, so argsort noise is a random order.
        ids_restore = torch.argsort(ids_shuffle) # recover mask for a shuffled point by rearranding the indice

        keep = ids_shuffle[:len_keep] # random sampling from unique bev cells

        unique_keep_bool = torch.zeros(nums).to(voxel_features.device).detach() # unique bev cell zero mask
        unique_keep_bool[keep] = 1 # the kept bev cells assigned to 1. bev binary mask
        # unique_mask_bool = unique_mask_bool.bool()
        ids_keep = torch.gather(unique_keep_bool, 0, inverse_index) # propagate the bev binary mask to all bev poitns (coor_down_sample) (b * max_point,)
        ids_keep = ids_keep.bool() # from bianry to bool 

        ids_mask = ~ids_keep

        batch_size = batch_dict['batch_size']
        ### mask
        voxel_features_mask, voxel_coords_mask = voxel_features[ids_mask,:], coors[ids_mask,:] # bev masked original feature and coords

        voxel_fratures_mask_one = torch.ones(voxel_features_mask.shape[0],1).to(voxel_features_mask.device).detach() # 1 as feature for masked positions
        
        # a dense 3d tensor (entire grid defined by point cloud range and voxel size) that masked positions have one unmasked have zeros
        pts_mask = spconv.SparseConvTensor(
            voxel_fratures_mask_one,
            voxel_coords_mask.int(),
            self.sparse_shape,
            batch_size
        ).dense()  # (b, 1, d, w, h) 

        pts_mask = pts_mask.detach()
        # point_mask = pts_mask.clone()

        # pixel unshuffling: aggreate local neighbors to the single point. 
        # A kind of downsampling that does not lose info by moving nearby values to the channel dimension of the aggregated position. 
        # TODO big tensor 
        pts_mask = self.unshuffle(pts_mask) # (b, 1, d * 8 * 8, w//8, h//8). 8: downsampling factor.
        # print(pts_mask.shape)
        
        # reduce channels by max. 
        # since the pts_mask is a binary dense mask, the resulting bev_mask (b, h, w) indicates the occupancy of the entire grid in the BEV plane. 
        # masked bev position has one, 0 elsewhere.
        bev_mask = pts_mask.squeeze(1).max(dim=1)[0] # (b, bev_w, bev_h)
        # bev_mask = bev_mask.max(dim=1)[0]
        
        # if len(bev_mask.shape) != 3:
        #     import pdb; pdb.set_trace()
        self.forward_re_dict['gt_mask'] = bev_mask
        
        #### gt num
        # dense tensor that has the number of point assigned to each voxel
        # pts_gt_num = spconv.SparseConvTensor(
        #     num_points.view(-1, 1).detach(), # (b * max_point, 1) as feature
        #     coors.int(), # (b * max_point, 4)
        #     self.sparse_shape, # dense grid size (z, y, x)
        #     batch_size
        # ).dense() # (b, 1, d, w, h). 
        # bs, _, d, h, w = pts_gt_num.shape
        # pts_gt_num = self.unshuffle(pts_gt_num.reshape(bs, d, h, w)) # downsample like above
        
        # # sum the #point in a BEV cell and divided by the numebr unshuffled cells. 
        # # essentially getting the average #point in local voxels, as a number of points inside a bev cell. 
        # pts_gt_num = pts_gt_num.sum(dim=1, keepdim=True) / self.down_factor**2 
        # # pts_gt_num = pts_gt_num / (torch.max(pts_gt_num.reshape(bs, -1), dim=1, keepdim=True)[0] + 1e-6) # sample wise normalization
        # pts_gt_num = pts_gt_num.detach()
        # self.forward_re_dict['gt_num'] = pts_gt_num

        ### input
        voxel_features_partial, voxel_coords_partial = voxel_features[ids_keep,:], coors[ids_keep,:] # kept orignal input

        average_features = self.mask_token.repeat(voxel_features_mask.size(0), 1) # masked tokens 
        
        # concat masked tokens to remaining featuers and voxel coords
        voxel_features_partial = torch.cat([voxel_features_partial, average_features], dim=0) # concat mask tokens. (b * num_sample, c_in)
        voxel_coords_partial = torch.cat([voxel_coords_partial, voxel_coords_mask], dim=0) # mask tokens have original grid positions (not feature).  

        # a sparse tensor whihc have orignal feature at unmasked position and maske token at masked positions
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features_partial,
            indices=voxel_coords_partial.int(),
            spatial_shape=self.sparse_shape, # grid size 
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor) # (b * num_sample, 16)
        x_conv1 = self.conv1(x) # (b * num_sample, 16)
        x_conv2 = self.conv2(x_conv1) # (downsampled x 2, 32)
        x_conv3 = self.conv3(x_conv2) # (downsampled x 4, 64)
        x_conv4 = self.conv4(x_conv3) # (downsampled x 8, 128)
        out = self.conv_out(x_conv4) # (xy same while z further downsampled x 16, 128)
        feats = out.dense() # converted to dense tensor. (bs, 128, d=grid_z / 16, w=grid_x/8, grid_y/8) 
        bs, c, d, h, w = feats.shape
        # depth and channel merged. in other words, depth is flattened to make a 2d feature map  
        feats = feats.reshape(bs, -1, h, w) # roughly z / 16 * 128 for channels

        # pred_coor, pred_num = self.decode_feat(feats) # predict coordinates + number of points in the BEV cell
        pred_coor = self.decode_feat(feats) # predict coordinates + number of points in the BEV cell
        self.forward_re_dict['pred_coor'] = pred_coor  # (b, 3 * 20, bev_h, bev_w) # prediction of 20 coordinates 
        # self.forward_re_dict['pred_num'] = pred_num # (b, 1, bev_h, bev_w) # prediction of average number in a voxel
        # import pdb; pdb.set_trace()

        # original points/features in BEV plane extracted during preprocessing. these attributes are defined in the data_processor.py. 
        # voxels_large: (b * n_bev cell, max_point, c_in). coors_large: (b * n_bev, 4)
        voxels_large, num_points_large, coors_large = batch_dict['voxels_bev'], batch_dict['voxel_num_points_bev'], batch_dict['voxel_coords_bev'], 
        f_center = torch.zeros_like(voxels_large[:, :, :3]) # (n_bev, max_points_per_bev, 3). absolute xyz in a voxel position. 
        
        # make global coords relative to BEV cell
        # step by step (by arranging the equation)
        ## voxel_large - min_coord (part of offset): make the min value of each dim to 0
        ## - coors_large * v_size - v_size/2 (part of offset): move each point cluster relative to its voxel center. scaling by v_size by transforming voxel scale to the orignal point cloud scale.
        ## lastly scale by v_size to transform the values from abs to voxel space again. 
        ## in a nutshell, this step centralizes raw poitns by its corresponding voxel center. The last data is in the voxel space. 
        f_center[:, :, 0] = (voxels_large[:, :, 0] - (coors_large[:, 3].unsqueeze(dim=1) * self.vx + self.x_offset)) / self.vx # (n_bev, k) - (n_bev, 1)
        f_center[:, :, 1] = (voxels_large[:, :, 1] - (coors_large[:, 2].unsqueeze(dim=1) * self.vy + self.y_offset)) / self.vy
        f_center[:, :, 2] = (voxels_large[:, :, 2]) / self.vz

        voxel_count = f_center.shape[1] # number of point per voxel
        mask_num = get_paddings_indicator(num_points_large, voxel_count, axis=0) # (b, max_num). boolean
        mask_num = torch.unsqueeze(mask_num, -1).type_as(f_center) # (b, max_num, 1). binary mask that marks the valid points within a voxel.
        f_center *= mask_num # zero those padded points (b * n_bev, max_num, 3)

        sparse_shape = [1, self.sparse_shape[1]//self.down_factor, self.sparse_shape[2]//self.down_factor,] # (1, bev_cell_w, bev_cell_h)

        chamfer_mask = spconv.SparseConvTensor(# 0 at padded points, 1 at real points 
            mask_num.squeeze().detach(), # (b * nbev, max_num)
            coors_large.int(), # (b * nbev, 4)
            sparse_shape, # # (1, bev_cell_w, bev_cell_h) # grid size z y x
            batch_size
        ).dense() # (b, max_num, 1, bev_w, bev_h). # valid mask for chamfer calcuation

        self.forward_re_dict['chamfer_mask'] = chamfer_mask.sum(dim=2)

        n, m, _ = f_center.shape
        f_center = f_center.reshape(n, -1)

        pts_gt_coor = spconv.SparseConvTensor(
            f_center.detach(), # (b * n_bev, max_num *3) flattend coord
            coors_large.int(), # (b * n_bev, 4) # voxel coord
            sparse_shape,
            batch_size
        ).dense() # (b, max_num*3, 1, n_bev_w, n_nev_h)

        bs, _, d, h, w = pts_gt_coor.shape
        pts_gt_coor = pts_gt_coor.reshape(bs, m, -1, h, w) # unflattened coord. (b, max_num, 3, bev_h, bev_w)
        self.forward_re_dict['gt_coor'] = pts_gt_coor


        return batch_dict


class BEV_MAE_res_expand_no_density_prediction_normalized_reconstruction(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        self.mask_ratio = model_cfg.MASKED_RATIO
        self.grid = model_cfg.GRID
        
        self.sparse_shape = grid_size[::-1] + [1, 0, 0] 

        self.base_channel = 16
        self.width_mul = model_cfg.WIDTH_MUL
        self.num_blocks = model_cfg.NUM_BLOCKS

        self.out_input = self.base_channel * self.width_mul
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, self.out_input, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(self.out_input),
            nn.ReLU(),
        )
        block = post_act_block

        ## Level 1
        self.out_1 = self.out_input
        block_list = []
        n_block = self.num_blocks[0]
        block_list.append(SparseBasicBlock(self.out_input, self.out_1, norm_fn=norm_fn, indice_key='res1'))
        for _ in range(n_block):
            block_list.append(SparseBasicBlock(self.out_1, self.out_1, norm_fn=norm_fn, indice_key='res1'))
        self.conv1 = spconv.SparseSequential(*block_list)
        # self.conv1 = spconv.SparseSequential( # no downsampling
        #     SparseBasicBlock(self.out_input, self.out_1, norm_fn=norm_fn, indice_key='res1'),
        #     SparseBasicBlock(self.out_1, self.out_1, norm_fn=norm_fn, indice_key='res1'),
        # )

        self.out_2 = self.out_1 * 2
        block_list = []
        n_block = self.num_blocks[1]
        block_list.append(block(self.out_1, self.out_2, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'))
        for _ in range(n_block):
            block_list.append(SparseBasicBlock(self.out_2, self.out_2, norm_fn=norm_fn, indice_key='res2'))
        self.conv2 = spconv.SparseSequential(*block_list)
        # self.conv2 = spconv.SparseSequential( # downsampling by 2
        #     # [1600, 1408, 41] <- [800, 704, 21]
        #     block(self.out_1, self.out_2, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
        #     SparseBasicBlock(self.out_2, self.out_2, norm_fn=norm_fn, indice_key='res2'),
        #     SparseBasicBlock(self.out_2, self.out_2, norm_fn=norm_fn, indice_key='res2'),
        # )


        self.out_3 = self.out_2 * 2
        block_list = []
        n_block = self.num_blocks[2]
        block_list.append(block(self.out_2, self.out_3, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'))
        for _ in range(n_block):
            block_list.append(SparseBasicBlock(self.out_3, self.out_3, norm_fn=norm_fn, indice_key='res3'))
        self.conv3 = spconv.SparseSequential(*block_list)
        # self.conv3 = spconv.SparseSequential( # by 2
        #     # [800, 704, 21] <- [400, 352, 11] 
        #     block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
        #     SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        #     SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        # )


        self.out_4 = self.out_3 * 2
        block_list = []
        n_block = self.num_blocks[3]
        block_list.append(block(self.out_3, self.out_4, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'))
        for _ in range(n_block):
            block_list.append(SparseBasicBlock(self.out_4, self.out_4, norm_fn=norm_fn, indice_key='res4'))
        self.conv4 = spconv.SparseSequential(*block_list)
        # self.conv4 = spconv.SparseSequential(
        #     # [400, 352, 11] <- [200, 176, 5]
        #     block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        #     SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        #     SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        # )

        self.out_last = self.out_4
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad) # if last pad doesnt exist then return 0
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(self.out_last, self.out_last, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.out_last),
            nn.ReLU(),
        )
        
        self.num_point_features = 16 # why? does not exist in this file          
        
        last_dim_z, last_dim_y, last_dim_x = self.get_last_dims(grid_size=grid_size[::-1])
        
        self.decoder = nn.Sequential(
            # nn.Conv2d(last_dim_z * 128, 256, 3, padding=1, stride=1),
            nn.Conv2d(last_dim_z * self.out_last, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.n_point_recon = self.model_cfg.get('N_POINT_RECON', 20)
        self.coor_conv = nn.Conv2d(256, 3*self.n_point_recon, 1) # what is this 20? --> probably the number of points to predict. Chamfer distance can have differnet src dst #points.
        # self.num_conv = nn.Conv2d(256, 1, 1)
        # self.coor_conv = nn.Sequential(
        #     nn.Conv2d(256, 128, 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 3 * 20, 1)
        # )
        # self.num_conv = nn.Sequential(
        #     nn.Conv2d(256, 128, 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 1, 1)
        # )

        down_factor = 8 # final feature map is downsampled 8, so mask is also downsampled x8
        self.down_factor = down_factor
        self.unshuffle = torch.nn.PixelUnshuffle(down_factor)
        # self.vx = voxel_size[0] * down_factor
        # self.vy = voxel_size[1] * down_factor
        # self.vz = voxel_size[2] * down_factor
        voxel_size = model_cfg.VOXEL_SIZE # model voxel size, not dataset.  
        point_cloud_range = model_cfg.POINT_CLOUD_RANGE  # the same as dataet one 
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        # point_cloud_range[0:3]: minx, miny, minz. defined in config
        self.x_offset = self.vx / 2 + point_cloud_range[0] # half voxel + the min range offsets that are used to move PC to origin. probably for spconv.
        self.y_offset = self.vy / 2 + point_cloud_range[1] # 
        self.z_offset = point_cloud_range[2]

        self.coor_loss = loss_utils.MaskChamferDistance()
        # self.num_loss = nn.SmoothL1Loss(reduction='none', beta=1.0)

        self.mask_token = nn.Parameter(torch.zeros(1,3)) 

        self.forward_re_dict = {}

    def get_last_dims(self, grid_size):
        kernel_list = [
            (3,3,3),
            (3,3,3),
            (3,3,3),
            (3,1,1)
        ]
        padding_list = [
            (1,1,1),
            (1,1,1),
            (1,1,1),
            (0,0,0)
        ]
        stride_list = [
            (2,2,2),
            (2,2,2),
            (2,2,2),
            (2,1,1)
        ]
        def calculate_output_size(input_size, kernel_size, padding, stride):
            if isinstance(padding, tuple) and isinstance(stride, tuple) and isinstance(kernel_size, tuple):
                return tuple((input_size[i] + 2 * padding[i] - kernel_size[i]) // stride[i] + 1 for i in range(3))
            else:
                raise ValueError("Padding, stride, and kernel size must be tuples of length 3.")
        D, H, W = grid_size
        n_conv = len(kernel_list)
        for i in range(n_conv):
            D, H, W = calculate_output_size((D, H, W), kernel_list[i], padding_list[i], stride_list[i])
        return D, H, W

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        # pred = self.forward_re_dict['pred']
        # target = self.forward_re_dict['target']
        pred_coor = self.forward_re_dict['pred_coor']
        gt_coor = self.forward_re_dict['gt_coor'].detach()
        chamfer_mask = self.forward_re_dict['chamfer_mask'].detach()

        # pred_num = self.forward_re_dict['pred_num']
        # gt_num = self.forward_re_dict['gt_num'].detach()

        gt_mask = self.forward_re_dict['gt_mask'].detach()
        # loss = self.criterion(pred, target)
        
        # loss_num = self.get_num_loss(pred_num, gt_num, gt_mask)

        loss_coor = self.get_coor_loss(pred_coor, gt_coor, gt_mask, chamfer_mask)

        # loss = loss_num + loss_coor # added a weight to loss_num because it causes the exploding gradient when AMP
        loss =  loss_coor 

        tb_dict = {
            'loss_coor': loss_coor.item(),
        }


        return loss, tb_dict
    
    def get_coor_loss(self, pred, target, mask, chamfer_mask):
        bs, d, _, h, w = target.shape
        target = target.reshape(bs, -1, h, w)
        target = target.permute(0, 2, 3, 1)

        pred = pred.permute(0, 2, 3, 1) 
        chamfer_mask = chamfer_mask.permute(0, 2, 3, 1)

        mask = mask.squeeze().bool()
        if bs == 1:
            mask = mask.unsqueeze(dim=0)

        pred = pred[mask]
        target = target[mask]

        chamfer_mask = chamfer_mask[mask]


        pred = pred.reshape(-1, 3, self.n_point_recon).permute(0, 2, 1) # (-1, 20, 3)
        h_per_bev_min = torch.min(pred[:, :, 2], 1, keepdim=True)[0]
        h_per_bev_max = torch.max(pred[:, :, 2], 1, keepdim=True)[0]
        pred[:, :, 2] = (pred[:, :, 2] - h_per_bev_min) / (h_per_bev_max - h_per_bev_min + 1e-6) # min max scale [0, 1]
        target = target.reshape(-1, d, 3)

        loss_source, loss_target = self.coor_loss(pred, target, chamfer_mask)

        loss = loss_source + loss_target
        return loss
    
    def decode_feat(self, feats, mask=None):
        # feats = feats[mask]
        if mask is not None:
            bs, c, h, w = feats.shape
            # print(mask.shape)
            mask_tokens = self.mask_token.view(1, -1, 1, 1).expand(bs, -1, h, w)
            w = mask.unsqueeze(dim=1).expand_as(mask_tokens)
            feats = feats + w * mask_tokens

        x = self.decoder(feats)
        bs, c, h, w = x.shape
        # x = feats
        coor = self.coor_conv(x)
        # x = x.reshape(bs, )
        return coor


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        # voxel_features:(bs * max_point, c_in), coors: (bs * max_point, [batch_idx,z_idx, y_idx, x_idx]); num_points: (b * max_point, )
        voxel_features, coors, num_points = batch_dict['voxel_features'], batch_dict['voxel_coords'], batch_dict['voxel_num_points']
        # print(coors.shape)
        coor_down_sample = coors.int().detach().clone() # (N, (b, z, y, x))
        coor_down_sample = coors.int().clone() # (N, (b, z, y, x))
        coor_down_sample[:, 1:] = coor_down_sample[:, 1:]//(self.down_factor * self.grid) # draw a grid with (down_factor x grid)=(8x1) as unit. probably BEV cell size   
        coor_down_sample[:, 1] = coor_down_sample[:, 1]//(coor_down_sample[:, 1].max(dim=0)[0]*2) # all z made 0. projected to the ground.  
        
        # get a unique voxel for each BEV cell. In other words, occupied BEV cell. 
        # inverse index: an index that can reconstruct the original input.  
        # dim = 0 means each element along dim 0 (batch) is a unit to be compared (pytorch doc). So, the unique voxel is retained. 
        unique_coor_down_sample, inverse_index = torch.unique(coor_down_sample, return_inverse=True, dim=0) 

        select_ratio = 1 - self.mask_ratio # ratio for select bev voxel
        nums = unique_coor_down_sample.shape[0] # num of unique bev voxel. 
        
        len_keep = int(nums * select_ratio) # num of unique BEV cells to keep 

        noise = torch.rand(nums, device=voxel_features.device)  # noise in [0, 1] # random noise for creating a bev mask

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise) # a random mask for shuffling. noise random, so argsort noise is a random order.
        ids_restore = torch.argsort(ids_shuffle) # recover mask for a shuffled point by rearranding the indice

        keep = ids_shuffle[:len_keep] # random sampling from unique bev cells

        unique_keep_bool = torch.zeros(nums).to(voxel_features.device).detach() # unique bev cell zero mask
        unique_keep_bool[keep] = 1 # the kept bev cells assigned to 1. bev binary mask
        # unique_mask_bool = unique_mask_bool.bool()
        ids_keep = torch.gather(unique_keep_bool, 0, inverse_index) # propagate the bev binary mask to all bev poitns (coor_down_sample) (b * max_point,)
        ids_keep = ids_keep.bool() # from bianry to bool 

        ids_mask = ~ids_keep

        batch_size = batch_dict['batch_size']
        ### mask
        voxel_features_mask, voxel_coords_mask = voxel_features[ids_mask,:], coors[ids_mask,:] # bev masked original feature and coords

        voxel_fratures_mask_one = torch.ones(voxel_features_mask.shape[0],1).to(voxel_features_mask.device).detach() # 1 as feature for masked positions
        
        # a dense 3d tensor (entire grid defined by point cloud range and voxel size) that masked positions have one unmasked have zeros
        pts_mask = spconv.SparseConvTensor(
            voxel_fratures_mask_one,
            voxel_coords_mask.int(),
            self.sparse_shape,
            batch_size
        ).dense()  # (b, 1, d, w, h) 

        pts_mask = pts_mask.detach()
        # point_mask = pts_mask.clone()

        # pixel unshuffling: aggreate local neighbors to the single point. 
        # A kind of downsampling that does not lose info by moving nearby values to the channel dimension of the aggregated position. 
        pts_mask = self.unshuffle(pts_mask) # (b, 1, d * 8 * 8, w//8, h//8). 8: downsampling factor.
        # print(pts_mask.shape)
        
        # reduce channels by max. 
        # since the pts_mask is a binary dense mask, the resulting bev_mask (b, h, w) indicates the occupancy of the entire grid in the BEV plane. 
        # masked bev position has one, 0 elsewhere.
        bev_mask = pts_mask.squeeze(1).max(dim=1)[0] # (b, bev_w, bev_h)
        # bev_mask = bev_mask.max(dim=1)[0]
        
        # if len(bev_mask.shape) != 3:
        #     import pdb; pdb.set_trace()
        self.forward_re_dict['gt_mask'] = bev_mask
        
        #### gt num
        # dense tensor that has the number of point assigned to each voxel
        # pts_gt_num = spconv.SparseConvTensor(
        #     num_points.view(-1, 1).detach(), # (b * max_point, 1) as feature
        #     coors.int(), # (b * max_point, 4)
        #     self.sparse_shape, # dense grid size (z, y, x)
        #     batch_size
        # ).dense() # (b, 1, d, w, h). 
        # bs, _, d, h, w = pts_gt_num.shape
        # pts_gt_num = self.unshuffle(pts_gt_num.reshape(bs, d, h, w)) # downsample like above
        
        # # sum the #point in a BEV cell and divided by the numebr unshuffled cells. 
        # # essentially getting the average #point in local voxels, as a number of points inside a bev cell. 
        # pts_gt_num = pts_gt_num.sum(dim=1, keepdim=True) / self.down_factor**2 
        # # pts_gt_num = pts_gt_num / (torch.max(pts_gt_num.reshape(bs, -1), dim=1, keepdim=True)[0] + 1e-6) # sample wise normalization
        # pts_gt_num = pts_gt_num.detach()
        # self.forward_re_dict['gt_num'] = pts_gt_num

        ### input
        voxel_features_partial, voxel_coords_partial = voxel_features[ids_keep,:], coors[ids_keep,:] # kept orignal input

        average_features = self.mask_token.repeat(voxel_features_mask.size(0), 1) # masked tokens 
        
        # concat masked tokens to remaining featuers and voxel coords
        voxel_features_partial = torch.cat([voxel_features_partial, average_features], dim=0) # concat mask tokens. (b * num_sample, c_in)
        voxel_coords_partial = torch.cat([voxel_coords_partial, voxel_coords_mask], dim=0) # mask tokens have original grid positions (not feature).  

        # a sparse tensor whihc have orignal feature at unmasked position and maske token at masked positions
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features_partial,
            indices=voxel_coords_partial.int(),
            spatial_shape=self.sparse_shape, # grid size 
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor) # (b * num_sample, 16)
        x_conv1 = self.conv1(x) # (b * num_sample, 16)
        x_conv2 = self.conv2(x_conv1) # (downsampled x 2, 32)
        x_conv3 = self.conv3(x_conv2) # (downsampled x 4, 64)
        x_conv4 = self.conv4(x_conv3) # (downsampled x 8, 128)
        out = self.conv_out(x_conv4) # (xy same while z further downsampled x 16, 128)
        feats = out.dense() # converted to dense tensor. (bs, 128, d=grid_z / 16, w=grid_x/8, grid_y/8) 
        bs, c, d, h, w = feats.shape
        # depth and channel merged. in other words, depth is flattened to make a 2d feature map  
        feats = feats.reshape(bs, -1, h, w) # roughly z / 16 * 128 for channels

        # pred_coor, pred_num = self.decode_feat(feats) # predict coordinates + number of points in the BEV cell
        pred_coor = self.decode_feat(feats) # predict coordinates + number of points in the BEV cell
        self.forward_re_dict['pred_coor'] = pred_coor  # (b, 3 * 20, bev_h, bev_w) # prediction of 20 coordinates 
        # self.forward_re_dict['pred_num'] = pred_num # (b, 1, bev_h, bev_w) # prediction of average number in a voxel
        # import pdb; pdb.set_trace()

        # original points/features in BEV plane extracted during preprocessing. these attributes are defined in the data_processor.py. 
        # voxels_large: (b * n_bev cell, max_point, c_in). coors_large: (b * n_bev, 4)
        voxels_large, num_points_large, coors_large = batch_dict['voxels_bev'], batch_dict['voxel_num_points_bev'], batch_dict['voxel_coords_bev'], 
        f_center = torch.zeros_like(voxels_large[:, :, :3]) # (n_bev, max_points_per_bev, 3). absolute xyz in a voxel position. 
        
        # make global coords relative to BEV cell
        # step by step (by arranging the equation)
        ## voxel_large - min_coord (part of offset): make the min value of each dim to 0
        ## - coors_large * v_size - v_size/2 (part of offset): move each point cluster relative to its voxel center. scaling by v_size by transforming voxel scale to the orignal point cloud scale.
        ## lastly scale by v_size to transform the values from abs to voxel space again. 
        ## in a nutshell, this step centralizes raw poitns by its corresponding voxel center. The last data is in the voxel space. 
        f_center[:, :, 0] = (voxels_large[:, :, 0] - (coors_large[:, 3].unsqueeze(dim=1) * self.vx + self.x_offset)) / self.vx # (n_bev, k) - (n_bev, 1)
        f_center[:, :, 1] = (voxels_large[:, :, 1] - (coors_large[:, 2].unsqueeze(dim=1) * self.vy + self.y_offset)) / self.vy
        # f_center[:, :, 2] = (voxels_large[:, :, 2]) / self.vz
        h_per_bev_min = torch.min(voxels_large[:, :, 2], 1, keepdim=True)[0]
        h_per_bev_max = torch.max(voxels_large[:, :, 2], 1, keepdim=True)[0]
        f_center[:, :, 2] = (voxels_large[:, :, 2] - h_per_bev_min) / (h_per_bev_max - h_per_bev_min + 1e-6) # min max scale [0, 1]

        voxel_count = f_center.shape[1] # number of point per voxel
        mask_num = get_paddings_indicator(num_points_large, voxel_count, axis=0) # (b, max_num). boolean
        mask_num = torch.unsqueeze(mask_num, -1).type_as(f_center) # (b, max_num, 1). binary mask that marks the valid points within a voxel.
        f_center *= mask_num # zero those padded points (b * n_bev, max_num, 3)

        sparse_shape = [1, self.sparse_shape[1]//self.down_factor, self.sparse_shape[2]//self.down_factor,] # (1, bev_cell_w, bev_cell_h)

        chamfer_mask = spconv.SparseConvTensor(# 0 at padded points, 1 at real points 
            mask_num.squeeze().detach(), # (b * nbev, max_num)
            coors_large.int(), # (b * nbev, 4)
            sparse_shape, # # (1, bev_cell_w, bev_cell_h) # grid size z y x. downsampled by self.down_factor(def 8)
            batch_size
        ).dense() # (b, max_num, 1, bev_w, bev_h). # valid mask for chamfer calcuation

        self.forward_re_dict['chamfer_mask'] = chamfer_mask.sum(dim=2)

        n, m, _ = f_center.shape
        f_center = f_center.reshape(n, -1)

        # gt coord for reconstruction. the bev cell size is 8 times larger than input, and input is downsampled by 8 by the network
        # to match this bev cell resolution. Though the network downsamples 8 times the point clouds, each bev cell is required to 
        # reconstruct orignal points defined in the coarser BEV grid cells.  
        pts_gt_coor = spconv.SparseConvTensor(
            f_center.detach(), # (b * n_bev, max_num *3) flattend coord
            coors_large.int(), # (b * n_bev, 4) # voxel coord
            sparse_shape, # (1, bev_cell_w, bev_cell_h) # grid size z y x. downsampled by self.down_factor(def 8)
            batch_size
        ).dense() # (b, max_num*3, 1, n_bev_w, n_nev_h)

        bs, _, d, h, w = pts_gt_coor.shape
        pts_gt_coor = pts_gt_coor.reshape(bs, m, -1, h, w) # unflattened coord. (b, max_num, 3, bev_h, bev_w)
        self.forward_re_dict['gt_coor'] = pts_gt_coor


        return batch_dict



class BEV_MAE_res_expand_no_density_decoder_more_conv(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        self.mask_ratio = model_cfg.MASKED_RATIO
        self.grid = model_cfg.GRID
        
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.base_channel = 16
        self.width_mul = model_cfg.WIDTH_MUL
        self.num_blocks = model_cfg.NUM_BLOCKS

        self.out_input = self.base_channel * self.width_mul
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, self.out_input, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(self.out_input),
            nn.ReLU(),
        )
        block = post_act_block

        ## Level 1
        self.out_1 = self.out_input
        block_list = []
        n_block = self.num_blocks[0]
        block_list.append(SparseBasicBlock(self.out_input, self.out_1, norm_fn=norm_fn, indice_key='res1'))
        for _ in range(n_block):
            block_list.append(SparseBasicBlock(self.out_1, self.out_1, norm_fn=norm_fn, indice_key='res1'))
        self.conv1 = spconv.SparseSequential(*block_list)
        # self.conv1 = spconv.SparseSequential( # no downsampling
        #     SparseBasicBlock(self.out_input, self.out_1, norm_fn=norm_fn, indice_key='res1'),
        #     SparseBasicBlock(self.out_1, self.out_1, norm_fn=norm_fn, indice_key='res1'),
        # )

        self.out_2 = self.out_1 * 2
        block_list = []
        n_block = self.num_blocks[1]
        block_list.append(block(self.out_1, self.out_2, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'))
        for _ in range(n_block):
            block_list.append(SparseBasicBlock(self.out_2, self.out_2, norm_fn=norm_fn, indice_key='res2'))
        self.conv2 = spconv.SparseSequential(*block_list)
        # self.conv2 = spconv.SparseSequential( # downsampling by 2
        #     # [1600, 1408, 41] <- [800, 704, 21]
        #     block(self.out_1, self.out_2, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
        #     SparseBasicBlock(self.out_2, self.out_2, norm_fn=norm_fn, indice_key='res2'),
        #     SparseBasicBlock(self.out_2, self.out_2, norm_fn=norm_fn, indice_key='res2'),
        # )


        self.out_3 = self.out_2 * 2
        block_list = []
        n_block = self.num_blocks[2]
        block_list.append(block(self.out_2, self.out_3, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'))
        for _ in range(n_block):
            block_list.append(SparseBasicBlock(self.out_3, self.out_3, norm_fn=norm_fn, indice_key='res3'))
        self.conv3 = spconv.SparseSequential(*block_list)
        # self.conv3 = spconv.SparseSequential( # by 2
        #     # [800, 704, 21] <- [400, 352, 11] 
        #     block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
        #     SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        #     SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        # )


        self.out_4 = self.out_3 * 2
        block_list = []
        n_block = self.num_blocks[3]
        block_list.append(block(self.out_3, self.out_4, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'))
        for _ in range(n_block):
            block_list.append(SparseBasicBlock(self.out_4, self.out_4, norm_fn=norm_fn, indice_key='res4'))
        self.conv4 = spconv.SparseSequential(*block_list)
        # self.conv4 = spconv.SparseSequential(
        #     # [400, 352, 11] <- [200, 176, 5]
        #     block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
        #     SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        #     SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        # )

        self.out_last = self.out_4
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad) # if last pad doesnt exist then return 0
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(self.out_last, self.out_last, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.out_last),
            nn.ReLU(),
        )
        
        self.num_point_features = 16 # why? does not exist in this file          
        
        last_dim_z, last_dim_y, last_dim_x = self.get_last_dims(grid_size=grid_size[::-1])
        
        self.decoder = nn.Sequential(
            # nn.Conv2d(last_dim_z * 128, 256, 3, padding=1, stride=1),
            nn.Conv2d(last_dim_z * self.out_last, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.n_point_recon = self.model_cfg.get('N_POINT_RECON', 20)
        self.coor_conv = nn.Conv2d(256, 3*self.n_point_recon, 1) # what is this 20? --> probably the number of points to predict. Chamfer distance can have differnet src dst #points.
        # self.num_conv = nn.Conv2d(256, 1, 1)
        # self.coor_conv = nn.Sequential(
        #     nn.Conv2d(256, 128, 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 3 * 20, 1)
        # )
        # self.num_conv = nn.Sequential(
        #     nn.Conv2d(256, 128, 1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 1, 1)
        # )

        down_factor = 8 # final feature map is downsampled 8, so mask is also downsampled x8
        self.down_factor = down_factor
        self.unshuffle = torch.nn.PixelUnshuffle(down_factor)
        # self.vx = voxel_size[0] * down_factor
        # self.vy = voxel_size[1] * down_factor
        # self.vz = voxel_size[2] * down_factor
        voxel_size = model_cfg.VOXEL_SIZE # model voxel size, not dataset.  
        point_cloud_range = model_cfg.POINT_CLOUD_RANGE  # the same as dataet one 
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        # point_cloud_range[0:3]: minx, miny, minz. defined in config
        self.x_offset = self.vx / 2 + point_cloud_range[0] # half voxel + the min range offsets that are used to move PC to origin. probably for spconv.
        self.y_offset = self.vy / 2 + point_cloud_range[1] # 
        self.z_offset = point_cloud_range[2]

        self.coor_loss = loss_utils.MaskChamferDistance()
        # self.num_loss = nn.SmoothL1Loss(reduction='none', beta=1.0)

        self.mask_token = nn.Parameter(torch.zeros(1,3)) 

        self.forward_re_dict = {}

    def get_last_dims(self, grid_size):
        kernel_list = [
            (3,3,3),
            (3,3,3),
            (3,3,3),
            (3,1,1)
        ]
        padding_list = [
            (1,1,1),
            (1,1,1),
            (1,1,1),
            (0,0,0)
        ]
        stride_list = [
            (2,2,2),
            (2,2,2),
            (2,2,2),
            (2,1,1)
        ]
        def calculate_output_size(input_size, kernel_size, padding, stride):
            if isinstance(padding, tuple) and isinstance(stride, tuple) and isinstance(kernel_size, tuple):
                return tuple((input_size[i] + 2 * padding[i] - kernel_size[i]) // stride[i] + 1 for i in range(3))
            else:
                raise ValueError("Padding, stride, and kernel size must be tuples of length 3.")
        D, H, W = grid_size
        n_conv = len(kernel_list)
        for i in range(n_conv):
            D, H, W = calculate_output_size((D, H, W), kernel_list[i], padding_list[i], stride_list[i])
        return D, H, W

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        # pred = self.forward_re_dict['pred']
        # target = self.forward_re_dict['target']
        pred_coor = self.forward_re_dict['pred_coor']
        gt_coor = self.forward_re_dict['gt_coor'].detach()
        chamfer_mask = self.forward_re_dict['chamfer_mask'].detach()

        # pred_num = self.forward_re_dict['pred_num']
        # gt_num = self.forward_re_dict['gt_num'].detach()

        gt_mask = self.forward_re_dict['gt_mask'].detach()
        # loss = self.criterion(pred, target)
        
        # loss_num = self.get_num_loss(pred_num, gt_num, gt_mask)

        loss_coor = self.get_coor_loss(pred_coor, gt_coor, gt_mask, chamfer_mask)

        # loss = loss_num + loss_coor # added a weight to loss_num because it causes the exploding gradient when AMP
        loss =  loss_coor 

        tb_dict = {
            'loss_coor': loss_coor.item(),
        }


        return loss, tb_dict
    
    def get_coor_loss(self, pred, target, mask, chamfer_mask):
        bs, d, _, h, w = target.shape
        target = target.reshape(bs, -1, h, w)
        target = target.permute(0, 2, 3, 1)

        pred = pred.permute(0, 2, 3, 1) 
        chamfer_mask = chamfer_mask.permute(0, 2, 3, 1)

        mask = mask.squeeze().bool()
        if bs == 1:
            mask = mask.unsqueeze(dim=0)

        pred = pred[mask]
        target = target[mask]

        chamfer_mask = chamfer_mask[mask]


        pred = pred.reshape(-1, 3, self.n_point_recon).permute(0, 2, 1) # (-1, 20, 3)
        h_per_bev_min = torch.min(pred[:, :, 2], 1, keepdim=True)[0]
        h_per_bev_max = torch.max(pred[:, :, 2], 1, keepdim=True)[0]
        pred[:, :, 2] = (pred[:, :, 2] - h_per_bev_min) / (h_per_bev_max - h_per_bev_min + 1e-6) # min max scale [0, 1]
        target = target.reshape(-1, d, 3)

        loss_source, loss_target = self.coor_loss(pred, target, chamfer_mask)

        loss = loss_source + loss_target
        return loss
    
    def decode_feat(self, feats, mask=None):
        # feats = feats[mask]
        if mask is not None:
            bs, c, h, w = feats.shape
            # print(mask.shape)
            mask_tokens = self.mask_token.view(1, -1, 1, 1).expand(bs, -1, h, w)
            w = mask.unsqueeze(dim=1).expand_as(mask_tokens)
            feats = feats + w * mask_tokens

        x = self.decoder(feats)
        bs, c, h, w = x.shape
        # x = feats
        coor = self.coor_conv(x)
        # x = x.reshape(bs, )
        return coor


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
                point_features: (N, C)
        """
        # voxel_features:(bs * max_point, c_in), coors: (bs * max_point, [batch_idx,z_idx, y_idx, x_idx]); num_points: (b * max_point, )
        voxel_features, coors, num_points = batch_dict['voxel_features'], batch_dict['voxel_coords'], batch_dict['voxel_num_points']
        # print(coors.shape)
        coor_down_sample = coors.int().detach().clone() # (N, (b, z, y, x))
        coor_down_sample = coors.int().clone() # (N, (b, z, y, x))
        coor_down_sample[:, 1:] = coor_down_sample[:, 1:]//(self.down_factor * self.grid) # draw a grid with (down_factor x grid)=(8x1) as unit. probably BEV cell size   
        coor_down_sample[:, 1] = coor_down_sample[:, 1]//(coor_down_sample[:, 1].max(dim=0)[0]*2) # all z made 0. projected to the ground.  
        
        # get a unique voxel for each BEV cell. In other words, occupied BEV cell. 
        # inverse index: an index that can reconstruct the original input.  
        # dim = 0 means each element along dim 0 (batch) is a unit to be compared (pytorch doc). So, the unique voxel is retained. 
        unique_coor_down_sample, inverse_index = torch.unique(coor_down_sample, return_inverse=True, dim=0) 

        select_ratio = 1 - self.mask_ratio # ratio for select bev voxel
        nums = unique_coor_down_sample.shape[0] # num of unique bev voxel. 
        
        len_keep = int(nums * select_ratio) # num of unique BEV cells to keep 

        noise = torch.rand(nums, device=voxel_features.device)  # noise in [0, 1] # random noise for creating a bev mask

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise) # a random mask for shuffling. noise random, so argsort noise is a random order.
        ids_restore = torch.argsort(ids_shuffle) # recover mask for a shuffled point by rearranding the indice

        keep = ids_shuffle[:len_keep] # random sampling from unique bev cells

        unique_keep_bool = torch.zeros(nums).to(voxel_features.device).detach() # unique bev cell zero mask
        unique_keep_bool[keep] = 1 # the kept bev cells assigned to 1. bev binary mask
        # unique_mask_bool = unique_mask_bool.bool()
        ids_keep = torch.gather(unique_keep_bool, 0, inverse_index) # propagate the bev binary mask to all bev poitns (coor_down_sample) (b * max_point,)
        ids_keep = ids_keep.bool() # from bianry to bool 

        ids_mask = ~ids_keep

        batch_size = batch_dict['batch_size']
        ### mask
        voxel_features_mask, voxel_coords_mask = voxel_features[ids_mask,:], coors[ids_mask,:] # bev masked original feature and coords

        voxel_fratures_mask_one = torch.ones(voxel_features_mask.shape[0],1).to(voxel_features_mask.device).detach() # 1 as feature for masked positions
        
        # a dense 3d tensor (entire grid defined by point cloud range and voxel size) that masked positions have one unmasked have zeros
        pts_mask = spconv.SparseConvTensor(
            voxel_fratures_mask_one,
            voxel_coords_mask.int(),
            self.sparse_shape,
            batch_size
        ).dense()  # (b, 1, d, w, h) 

        pts_mask = pts_mask.detach()
        # point_mask = pts_mask.clone()

        # pixel unshuffling: aggreate local neighbors to the single point. 
        # A kind of downsampling that does not lose info by moving nearby values to the channel dimension of the aggregated position. 
        # TODO big tensor 
        pts_mask = self.unshuffle(pts_mask) # (b, 1, d * 8 * 8, w//8, h//8). 8: downsampling factor.
        # print(pts_mask.shape)
        
        # reduce channels by max. 
        # since the pts_mask is a binary dense mask, the resulting bev_mask (b, h, w) indicates the occupancy of the entire grid in the BEV plane. 
        # masked bev position has one, 0 elsewhere.
        bev_mask = pts_mask.squeeze(1).max(dim=1)[0] # (b, bev_w, bev_h)
        # bev_mask = bev_mask.max(dim=1)[0]
        
        # if len(bev_mask.shape) != 3:
        #     import pdb; pdb.set_trace()
        self.forward_re_dict['gt_mask'] = bev_mask
        
        #### gt num
        # dense tensor that has the number of point assigned to each voxel
        # pts_gt_num = spconv.SparseConvTensor(
        #     num_points.view(-1, 1).detach(), # (b * max_point, 1) as feature
        #     coors.int(), # (b * max_point, 4)
        #     self.sparse_shape, # dense grid size (z, y, x)
        #     batch_size
        # ).dense() # (b, 1, d, w, h). 
        # bs, _, d, h, w = pts_gt_num.shape
        # pts_gt_num = self.unshuffle(pts_gt_num.reshape(bs, d, h, w)) # downsample like above
        
        # # sum the #point in a BEV cell and divided by the numebr unshuffled cells. 
        # # essentially getting the average #point in local voxels, as a number of points inside a bev cell. 
        # pts_gt_num = pts_gt_num.sum(dim=1, keepdim=True) / self.down_factor**2 
        # # pts_gt_num = pts_gt_num / (torch.max(pts_gt_num.reshape(bs, -1), dim=1, keepdim=True)[0] + 1e-6) # sample wise normalization
        # pts_gt_num = pts_gt_num.detach()
        # self.forward_re_dict['gt_num'] = pts_gt_num

        ### input
        voxel_features_partial, voxel_coords_partial = voxel_features[ids_keep,:], coors[ids_keep,:] # kept orignal input

        average_features = self.mask_token.repeat(voxel_features_mask.size(0), 1) # masked tokens 
        
        # concat masked tokens to remaining featuers and voxel coords
        voxel_features_partial = torch.cat([voxel_features_partial, average_features], dim=0) # concat mask tokens. (b * num_sample, c_in)
        voxel_coords_partial = torch.cat([voxel_coords_partial, voxel_coords_mask], dim=0) # mask tokens have original grid positions (not feature).  

        # a sparse tensor whihc have orignal feature at unmasked position and maske token at masked positions
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features_partial,
            indices=voxel_coords_partial.int(),
            spatial_shape=self.sparse_shape, # grid size 
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor) # (b * num_sample, 16)
        x_conv1 = self.conv1(x) # (b * num_sample, 16)
        x_conv2 = self.conv2(x_conv1) # (downsampled x 2, 32)
        x_conv3 = self.conv3(x_conv2) # (downsampled x 4, 64)
        x_conv4 = self.conv4(x_conv3) # (downsampled x 8, 128)
        out = self.conv_out(x_conv4) # (xy same while z further downsampled x 16, 128)
        feats = out.dense() # converted to dense tensor. (bs, 128, d=grid_z / 16, w=grid_x/8, grid_y/8) 
        bs, c, d, h, w = feats.shape
        # depth and channel merged. in other words, depth is flattened to make a 2d feature map  
        feats = feats.reshape(bs, -1, h, w) # roughly z / 16 * 128 for channels

        # pred_coor, pred_num = self.decode_feat(feats) # predict coordinates + number of points in the BEV cell
        pred_coor = self.decode_feat(feats) # predict coordinates + number of points in the BEV cell
        self.forward_re_dict['pred_coor'] = pred_coor  # (b, 3 * 20, bev_h, bev_w) # prediction of 20 coordinates 
        # self.forward_re_dict['pred_num'] = pred_num # (b, 1, bev_h, bev_w) # prediction of average number in a voxel
        # import pdb; pdb.set_trace()

        # original points/features in BEV plane extracted during preprocessing. these attributes are defined in the data_processor.py. 
        # voxels_large: (b * n_bev cell, max_point, c_in). coors_large: (b * n_bev, 4)
        voxels_large, num_points_large, coors_large = batch_dict['voxels_bev'], batch_dict['voxel_num_points_bev'], batch_dict['voxel_coords_bev'], 
        f_center = torch.zeros_like(voxels_large[:, :, :3]) # (n_bev, max_points_per_bev, 3). absolute xyz in a voxel position. 
        
        # make global coords relative to BEV cell
        # step by step (by arranging the equation)
        ## voxel_large - min_coord (part of offset): make the min value of each dim to 0
        ## - coors_large * v_size - v_size/2 (part of offset): move each point cluster relative to its voxel center. scaling by v_size by transforming voxel scale to the orignal point cloud scale.
        ## lastly scale by v_size to transform the values from abs to voxel space again. 
        ## in a nutshell, this step centralizes raw poitns by its corresponding voxel center. The last data is in the voxel space. 
        f_center[:, :, 0] = (voxels_large[:, :, 0] - (coors_large[:, 3].unsqueeze(dim=1) * self.vx + self.x_offset)) / self.vx # (n_bev, k) - (n_bev, 1)
        f_center[:, :, 1] = (voxels_large[:, :, 1] - (coors_large[:, 2].unsqueeze(dim=1) * self.vy + self.y_offset)) / self.vy
        # f_center[:, :, 2] = (voxels_large[:, :, 2]) / self.vz
        h_per_bev_min = torch.min(voxels_large[:, :, 2], 1, keepdim=True)[0]
        h_per_bev_max = torch.max(voxels_large[:, :, 2], 1, keepdim=True)[0]
        f_center[:, :, 2] = (voxels_large[:, :, 2] - h_per_bev_min) / (h_per_bev_max - h_per_bev_min + 1e-6) # min max scale [0, 1]

        voxel_count = f_center.shape[1] # number of point per voxel
        mask_num = get_paddings_indicator(num_points_large, voxel_count, axis=0) # (b, max_num). boolean
        mask_num = torch.unsqueeze(mask_num, -1).type_as(f_center) # (b, max_num, 1). binary mask that marks the valid points within a voxel.
        f_center *= mask_num # zero those padded points (b * n_bev, max_num, 3)

        sparse_shape = [1, self.sparse_shape[1]//self.down_factor, self.sparse_shape[2]//self.down_factor,] # (1, bev_cell_w, bev_cell_h)

        chamfer_mask = spconv.SparseConvTensor(# 0 at padded points, 1 at real points 
            mask_num.squeeze().detach(), # (b * nbev, max_num)
            coors_large.int(), # (b * nbev, 4)
            sparse_shape, # # (1, bev_cell_w, bev_cell_h) # grid size z y x. downsampled by self.down_factor(def 8)
            batch_size
        ).dense() # (b, max_num, 1, bev_w, bev_h). # valid mask for chamfer calcuation

        self.forward_re_dict['chamfer_mask'] = chamfer_mask.sum(dim=2)

        n, m, _ = f_center.shape
        f_center = f_center.reshape(n, -1)

        # gt coord for reconstruction. the bev cell size is 8 times larger than input, and input is downsampled by 8 by the network
        # to match this bev cell resolution. Though the network downsamples 8 times the point clouds, each bev cell is required to 
        # reconstruct orignal points defined in the coarser BEV grid cells.  
        pts_gt_coor = spconv.SparseConvTensor(
            f_center.detach(), # (b * n_bev, max_num *3) flattend coord
            coors_large.int(), # (b * n_bev, 4) # voxel coord
            sparse_shape, # (1, bev_cell_w, bev_cell_h) # grid size z y x. downsampled by self.down_factor(def 8)
            batch_size
        ).dense() # (b, max_num*3, 1, n_bev_w, n_nev_h)

        bs, _, d, h, w = pts_gt_coor.shape
        pts_gt_coor = pts_gt_coor.reshape(bs, m, -1, h, w) # unflattened coord. (b, max_num, 3, bev_h, bev_w)
        self.forward_re_dict['gt_coor'] = pts_gt_coor



        vis_save_path = batch_dict.get('vis_save_path', None)
        if vis_save_path is not None:
            # from datetime import datetime
            # import os.path as osp
            # # Get the current date and time and device id as an idnetifier
            # current_datetime = datetime.now()
            # formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
            # current_device = f_center.device

            ### for visualizatio only!
            vis_dict = {
                "coors_large": coors_large,
                "f_center": f_center,
                "forward_re_dict": self.forward_re_dict,
                "mask_num": mask_num,
                "voxels_large": voxels_large,
                "x_offset": self.x_offset,
                "vx": self.vx,
                "vy": self.vy,
                "vz": self.vz,
                "sparse_shape": sparse_shape,
            }
            torch.save(vis_dict, vis_save_path)
            # import pdb; pdb.set_trace()
            ### for visualizatio only!

        # ### for visualizatio only!
        # vis_dict = {
        #     "coors_large": coors_large,
        #     "f_center": f_center,
        #     "forward_re_dict": self.forward_re_dict,
        #     "mask_num": mask_num,
        #     "voxels_large": voxels_large
        # }
        # import pdb; pdb.set_trace()
        # ### for visualizatio only!

        return batch_dict

def get_paddings_indicator(actual_num, max_num, axis=0):
    """Create boolean mask by actually number of a padded tensor.

    Args:
        actual_num (torch.Tensor): Actual number of points in each voxel.
        max_num (int): Max number of points in each voxel

    Returns:
        torch.Tensor: Mask indicates which points are valid inside a voxel.
    """
    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(
        max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator