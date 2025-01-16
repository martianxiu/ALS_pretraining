from .detector3d_template_bev_mae import Detector3DTemplate_bev_mae
import torch

class BEV_MAE(Detector3DTemplate_bev_mae):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    # def forward(self, batch_dict):
    #     for cur_module in self.module_list:
    #         batch_dict = cur_module(batch_dict)

    #     if self.training:
    #         loss, tb_dict, disp_dict = self.get_training_loss()

    #         ret_dict = {
    #             'loss': loss
    #         }
    #         return ret_dict, tb_dict, disp_dict
    #     else:
    #         pred_dicts, recall_dicts = self.post_processing(batch_dict)
    #         return pred_dicts, recall_dicts
        
    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        
        loss, tb_dict, disp_dict = self.get_training_loss()

        ret_dict = {
            'loss': loss
        }
        return ret_dict, tb_dict, disp_dict

    def get_training_loss(self):
        disp_dict = {} # for visualizaiton/debugging purpose.

        loss_rpn, tb_dict = self.backbone_3d.get_loss() # loss_rpn: total loss, tb_dict: separate loss for TB logging
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

class BEV_MAE_return(Detector3DTemplate_bev_mae):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    # def forward(self, batch_dict):
    #     for cur_module in self.module_list:
    #         batch_dict = cur_module(batch_dict)

    #     if self.training:
    #         loss, tb_dict, disp_dict = self.get_training_loss()

    #         ret_dict = {
    #             'loss': loss
    #         }
    #         return ret_dict, tb_dict, disp_dict
    #     else:
    #         pred_dicts, recall_dicts = self.post_processing(batch_dict)
    #         return pred_dicts, recall_dicts
        
    def forward(self, batch_dict):
        # compute return number / num_return for input and BEV target
        ## cannot be done because of zero filling of PointTOVoxel. always produce 0s, leadning to nan.
        # points = batch_dict["voxels"]
        # n, m, c = points.shape
        # import pdb; pdb.set_trace()
        # return_num, num_return = points[:,:, -2:-1], points[:,:, -1:] # assume x,y,z,return_num, num_return
        # assert torch.any(~torch.isfinite(return_num.reshape(-1)/num_return.reshape(-1)))
        # return_ratio = return_num / num_return 
        # new_points = torch.cat([points[:, :, :3], return_ratio.reshape(n, m, 1)], dim=2)
        # batch_dict['voxels'] = new_points

        # points = batch_dict["voxels_bev"]
        # n, m, c = points.shape
        # return_num, num_return = points[:,:, -2:-1], points[:,:, -1:] # assume x,y,z,return_num, num_return
        # return_ratio = return_num / num_return 
        # new_points = torch.cat([points[:, :, :3], return_ratio.reshape(n, m, 1)], dim=2)
        # batch_dict['voxels_bev'] = new_points

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        
        loss, tb_dict, disp_dict = self.get_training_loss()

        ret_dict = {
            'loss': loss
        }
        return ret_dict, tb_dict, disp_dict

    def get_training_loss(self):
        disp_dict = {} # for visualizaiton/debugging purpose.

        loss_rpn, tb_dict = self.backbone_3d.get_loss() # loss_rpn: total loss, tb_dict: separate loss for TB logging
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict