from .segmentor3d_template_bev_mae import Segmenter3DTemplateBEVMAE


class BEV_MAE_UNet(Segmenter3DTemplateBEVMAE):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            iou_dict = self.get_scores(batch_dict) # "intersection", "union", "target", "batch_mIoU"
            ret_dict = {
                'loss': loss
            }
            ret_dict.update(iou_dict)
            disp_dict.update(iou_dict)
            tb_dict["batch_mIoU"] = iou_dict["batch_mIoU"]

            return ret_dict, tb_dict, disp_dict
        else:
            loss, tb_dict, disp_dict = self.get_training_loss()
            # tb_dict, disp_dict, ret_dict = {}, {}, {}
            score_dict = self.get_scores(batch_dict) # "intersection", "union", "target", "batch_mIoU"
            ret_dict = {
                'loss': loss
            }
            ret_dict.update(score_dict)
            disp_dict.update(score_dict)
            tb_dict["batch_mIoU"] = score_dict["batch_mIoU"]
            tb_dict["allAcc"] = score_dict["allAcc"]
            pred_dicts = {
                "pred": batch_dict["point_seg_logits"],
                "seg": batch_dict["point_seg_labels"]
            }
            disp_dict.update(pred_dicts)
            return ret_dict, tb_dict, disp_dict

    def get_training_loss(self):
        disp_dict = {} # for visualizaiton/debugging purpose.

        # loss_rpn, tb_dict = self.backbone_3d.get_loss() # loss_rpn: total loss, tb_dict: separate loss for TB logging
        # loss_seg, tb_dict = self.dense_head.get_loss() # deprecated as moved to seg_head
        loss_seg, tb_dict = self.seg_head.get_loss() # loss_rpn: total loss, tb_dict: separate loss for TB logging
        tb_dict = {
            'loss_seg': loss_seg.item(),
            **tb_dict
        }

        loss = loss_seg
        return loss, tb_dict, disp_dict
