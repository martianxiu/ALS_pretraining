import torch

# from ...utils import box_utils
from .classification_head_template import ClassificationHeadTemplate


class ClassificationHead(ClassificationHeadTemplate):
    
    def __init__(self, num_class, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=model_cfg.IN_CHANNEL,
            output_channels=num_class # len(CLASS_NAMES) in config.
        )

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        # point_loss_seg, tb_dict_1 = self.get_seg_layer_loss()
        point_loss_seg, tb_dict_1 = self.get_cls_layer_loss()

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
        
        logits = self.cls_layers(point_features)  # (total_samples, num_class)

        ret_dict = {
            'logits': logits,
        }
        batch_dict['logits'] = logits
        batch_dict['preds'] = logits.max(1)[1] # prediction
        
        ret_dict['labels'] = batch_dict['label'].view(-1)
        
        
        self.forward_ret_dict = ret_dict
        
        return batch_dict

class GlobalMeanMaxClassificationHead(ClassificationHeadTemplate):
    
    def __init__(self, num_class, model_cfg, **kwargs):
        super().__init__(model_cfg=model_cfg, num_class=num_class)
        self.cls_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.CLS_FC,
            input_channels=model_cfg.IN_CHANNEL,
            output_channels=num_class # len(CLASS_NAMES) in config.
        )

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        # point_loss_seg, tb_dict_1 = self.get_seg_layer_loss()
        point_loss_seg, tb_dict_1 = self.get_cls_layer_loss()

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
        point_features = batch_dict['point_features'] # (N (batched), C)
        point_indices = batch_dict['point_indices'] # (N, 4 (b, z, y, x))
        
        # per-batch mean and max 
        batch_size = batch_dict['batch_size']
        mean_feat = []
        max_feat = []
        for b_i in range(batch_size):
            batch_mask = point_indices[:, 0] == b_i
            sample_features = point_features[batch_mask]
            mean_feat.append(torch.mean(sample_features, dim=0, keepdim=True))
            max_feat.append(torch.max(sample_features, dim=0, keepdim=True)[0])

        mean_features = torch.cat(mean_feat, 0)
        max_features = torch.cat(max_feat, 0)
        mean_max_features = torch.cat([mean_features, max_features], dim=1) # b, 2C
        logits = self.cls_layers(mean_max_features)  # (total_samples, num_class)

        ret_dict = {
            'logits': logits,
        }
        batch_dict['logits'] = logits
        batch_dict['preds'] = logits.max(1)[1] # prediction
        
        ret_dict['labels'] = batch_dict['labels'].view(-1)
        
        
        self.forward_ret_dict = ret_dict
        
        return batch_dict
