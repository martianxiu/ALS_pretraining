from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .centerpoint import CenterPoint
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .mppnet import MPPNet
from .mppnet_e2e import MPPNetE2E
from .pillarnet import PillarNet
from .voxelnext import VoxelNeXt
from .transfusion import TransFusion
from .bevfusion import BevFusion

from .bev_mae_net import BEV_MAE, BEV_MAE_return
from .bev_mae_net_unet import BEV_MAE_UNet
from .segmentor3d_template_bev_mae import Segmenter3DTemplateBEVMAE
from .gd_mae import GDMAE
from .gdmae_seg import GDMAESEG
from .classifier3d_template_bev_mae import Classifier3DTemplateBEVMAE
from .bev_mae_net_cls import BEV_MAE_Cls

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'Segmenter3DTemplateBEVMAE': Segmenter3DTemplateBEVMAE,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'CenterPoint': CenterPoint,
    'PillarNet': PillarNet,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'MPPNet': MPPNet,
    'MPPNetE2E': MPPNetE2E,
    'VoxelNeXt': VoxelNeXt,
    'TransFusion': TransFusion,
    'BevFusion': BevFusion,
    'BEV_MAE': BEV_MAE,
    'BEV_MAE_return': BEV_MAE_return,
    'BEV_MAE_UNet': BEV_MAE_UNet,
    'GDMAE':GDMAE,
    'GDMAESEG': GDMAESEG,
    'Classifier3DTemplateBEVMAE': Classifier3DTemplateBEVMAE,
    'BEV_MAE_Cls': BEV_MAE_Cls
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
