import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

# --- exactly as in lib/pvt.py from Polyp-PVT :contentReference[oaicite:0]{index=0} ---
from lib.pvt import BasicConv2d, CFM, ChannelAttention, SpatialAttention, SAM
from lib.pvtv2 import pvt_v2_b2

class PolypPVTEncoder(nn.Module):
    """
    PVTv2-B2 → CIM → CFM → SAM
    Returns a dict of four feature maps at strides 4,8,16,32.
    """
    def __init__(self, channel: int = 32, pretrained_path: str = './pretrained_pth/pvt_v2_b2.pth'):
        super().__init__()
        # 1) PVTv2-B2 backbone
        self.backbone = pvt_v2_b2()                  # embed dims [64,128,320,512] :contentReference[oaicite:1]{index=1}
        state = torch.load(pretrained_path, map_location='cpu')
        m = self.backbone.state_dict()
        m.update({k:v for k,v in state.items() if k in m})
        self.backbone.load_state_dict(m)

        # 2) Translayers to unify to `channel` dims
        self.trans1 = BasicConv2d(64,  channel, 1)   # for CIM edge
        self.trans2 = BasicConv2d(128, channel, 1)
        self.trans3 = BasicConv2d(320, channel, 1)
        self.trans4 = BasicConv2d(512, channel, 1)

        # 3) CIM = ChannelAttention + SpatialAttention on x1
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()

        # 4) CFM on (x4_t, x3_t, x2_t)
        self.cfm = CFM(channel)

        # 5) SAM similarity aggregation
        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.sam    = SAM(num_in=channel)

        self.out_channels = channel

    def forward(self, x):
        # 1) PVT stages
        x1, x2, x3, x4 = self.backbone(x)   # strides 4,8,16,32

        # 2) CIM on x1
        ca = self.ca(x1) * x1
        cim_feat = self.sa(ca) * x1         # stays at stride 4

        # 3) Trans→CFM on x2,x3,x4
        t2 = self.trans2(x2)                # stride 8
        t3 = self.trans3(x3)                # stride 16
        t4 = self.trans4(x4)                # stride 32
        cfm_feat = self.cfm(t4, t3, t2)     # outputs stride 8

        # 4) SAM (cfm_feat @ edge)
        edge = self.trans1(cim_feat)        # stride 4
        edge = self.down05(edge)            # now stride 8
        sam_feat = self.sam(cfm_feat, edge) # stride 8

        # 5) collect for FPN
        return OrderedDict([
            ("p2", cim_feat      .new_zeros(1) or self.trans1(cim_feat)),  # stride 4 → trans1(cim) 
            ("p3", sam_feat),                                            # stride 8
            ("p4", t3),                                                   # stride 16
            ("p5", t4),                                                   # stride 32
        ])


def get_polyp_pvt_fpn_maskrcnn(num_classes: int = 2,
                               channel: int     = 32,
                               pretrained_path  = './pretrained_pth/pvt_v2_b2.pth'):
    # 1) our encoder
    encoder = PolypPVTEncoder(channel, pretrained_path)

    # 2) build a small FPN
    in_channels_list = [channel] * 4
    fpn = FeaturePyramidNetwork(
        in_channels_list=in_channels_list,
        out_channels=channel,
        extra_blocks=None
    )

    # wrap encoder+fpn into one module
    class BackboneWithFPN(nn.Module):
        def __init__(self, body, fpn):
            super().__init__()
            self.body = body
            self.fpn  = fpn
            self.out_channels = channel

        def forward(self, x):
            feats = self.body(x)      # OrderedDict p2..p5
            return self.fpn(feats)    # OrderedDict p2..p5 → p2..p5 (FPNet)

    backbone = BackboneWithFPN(encoder, fpn)

    # 3) anchors: one size per P2–P5
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5,1.0,2.0),) * 4
    )

    # 4) ROI Aligns for box & mask heads
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=["p2","p3","p4","p5"],
        output_size=7, sampling_ratio=2
    )
    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["p2","p3","p4","p5"],
        output_size=14, sampling_ratio=2
    )

    # 5) assemble Mask R-CNN
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        mask_roi_pool=mask_roi_pool
    )
    return model
