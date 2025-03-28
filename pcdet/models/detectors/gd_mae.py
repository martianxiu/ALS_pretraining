from .detector3d_template import Detector3DTemplate


class GDMAE(Detector3DTemplate):
    # def __init__(self, model_cfg, num_class, dataset, logger):
    #     super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, logger=logger)
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def post_processing(self, batch_dict):
        return {}, {}

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.backbone_3d.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
