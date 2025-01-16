# import torch

# from ...utils import box_utils
from .segmentation_head_template import SegmentationHeadTemplate


class SegmentationHead(SegmentationHeadTemplate):
    
    def __init__(self, num_class, input_channels, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=input_channels,
            output_channels=num_class # len(CLASS_NAMES) in config.
        )

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        point_loss_seg, tb_dict_1 = self.get_seg_layer_loss()

        point_loss = point_loss_seg
        tb_dict.update(tb_dict_1)
        return point_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                point_features: (N1 + N2 + N3 + ..., C) or (B, N, C)
                point_features_before_fusion: (N1 + N2 + N3 + ..., C)
                point_coords: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
                point_labels (optional): (N1 + N2 + N3 + ...)
                gt_boxes (optional): (B, M, 8)
        Returns:
            batch_dict:
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        """
        if self.model_cfg.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            point_features = batch_dict['point_features_before_fusion']
        else:
            point_features = batch_dict['point_features']
        point_cls_preds = self.cls_layers(point_features)  # (total_points, num_class)

        ret_dict = {
            'point_seg_preds': point_cls_preds,
        }
        batch_dict['point_seg_logits'] = point_cls_preds
        batch_dict['point_seg_preds'] = point_cls_preds.max(1)[1] # prediction

        
        # point_cls_scores = torch.sigmoid(point_cls_preds)
        # batch_dict['point_cls_scores'], _ = point_cls_scores.max(dim=-1)

        # if self.training:
            # targets_dict = self.assign_targets(batch_dict)
            # ret_dict['point_cls_labels'] = targets_dict['point_cls_labels']
        ret_dict['point_seg_labels'] = batch_dict['point_seg_labels'].view(-1)
        self.forward_ret_dict = ret_dict
        return batch_dict
