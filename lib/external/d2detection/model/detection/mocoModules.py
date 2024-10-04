from detectron2.layers import get_norm
from detectron2.modeling.roi_heads import Res5ROIHeads, ROI_HEADS_REGISTRY


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeadsExtraNorm(Res5ROIHeads):
    """
    As described in the MOCO paper, there is an extra BN layer
    following the res5 stage.
    """
    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        norm = cfg.MODEL.RESNETS.NORM
        norm = get_norm(norm, self.res5[-1].out_channels)
        self.res5.add_module("norm", norm)