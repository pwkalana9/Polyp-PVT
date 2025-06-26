import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

# --- import Polyp-PVT core blocks from lib/pvt.py and lib/pvtv2.py ---
from lib.pvt import BasicConv2d, CFM, ChannelAttention, SpatialAttention, SAM
from lib.pvtv2 import pvt_v2_b2

class PolypPVTBackbone(nn.Module):
    """
    Exact PVT → CIM → CFM → SAM backbone from Polyp-PVT,
    producing 4 feature maps at strides 4,8,16,32 (all with 'channel' channels).
    """
    def __init__(self, channel: int = 32, pretrained_path: str = './pretrained_pth/pvt_v2_b2.pth'):
        super().__init__()
        # 1) PVTv2-B2 encoder
        self.backbone = pvt_v2_b2()               # embed dims: [64,128,320,512] :contentReference[oaicite:0]{index=0}
        # load pretrained weights (same as Polyp-PVT)
        state = torch.load(pretrained_path, map_location='cpu')
        m_dict = self.backbone.state_dict()
        pretrained = {k: v for k, v in state.items() if k in m_dict}
        m_dict.update(pretrained)
        self.backbone.load_state_dict(m_dict)

        # 2) Translayers to unify each PVT stage into 'channel' dims
        d1, d2, d3, d4 = [64, 128, 320, 512]      # from pvt_v2_b2 embed_dims :contentReference[oaicite:1]{index=1}
        self.trans1 = BasicConv2d(d1, channel, kernel_size=1)   # for CIM→SAM edge
        self.trans2 = BasicConv2d(d2, channel, kernel_size=1)   # P3
        self.trans3 = BasicConv2d(d3, channel, kernel_size=1)   # P4
        self.trans4 = BasicConv2d(d4, channel, kernel_size=1)   # P5

        # 3) CIM branch: Channel + Spatial attention on stage1
        self.ca = ChannelAttention(d1)           # :contentReference[oaicite:2]{index=2}
        self.sa = SpatialAttention()             # :contentReference[oaicite:3]{index=3}

        # 4) CFM on stages 2–4
        self.cfm = CFM(channel)                  # :contentReference[oaicite:4]{index=4}

        # 5) SAM similarity aggregation
        self.sam = SAM(num_in=channel)           # :contentReference[oaicite:5]{index=5}
        self.down05 = nn.Upsample(scale_factor=0.5,
                                  mode='bilinear',
                                  align_corners=True)

        # final channel count for MaskRCNN
        self.out_channels = channel

    def forward(self, x):
        # --- PVT encoder ---
        feats = self.backbone(x)   # list of 4 tensors: [x1,x2,x3,x4]
        x1, x2, x3, x4 = feats

        # --- CIM on x1 only ---
        ca = self.ca(x1) * x1
        cim = self.sa(ca) * ca

        # --- CFM on Trans2–4 ---
        t2 = self.trans2(x2)
        t3 = self.trans3(x3)
        t4 = self.trans4(x4)
        cfm_feat = self.cfm(t4, t3, t2)

        # --- SAM using cfm_feat + edge from CIM via Trans1→down0.5 ---
        edge = self.down05(self.trans1(cim))
        sam_feat = self.sam(cfm_feat, edge)

        # --- assemble FPN-style dict for MaskRCNN ---
        return {
            # stride 4  → from stage1 CIM
            "0": self.trans1(cim),
            # stride 8  → fused CFM+SAM
            "1": sam_feat,
            # stride 16 → pure Trans3
            "2": t3,
            # stride 32 → pure Trans4
            "3": t4,
        }


def get_polyp_pvt_maskrcnn(num_classes: int = 2,
                           channel: int = 32,
                           pretrained_path: str = './pretrained_pth/pvt_v2_b2.pth'):
    # build the backbone
    backbone = PolypPVTBackbone(channel, pretrained_path)
    backbone.out_channels = backbone.out_channels

    # RPN anchors & ROI-aligns
    anchor_gen = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    box_pool = MultiScaleRoIAlign(
        featmap_names=[str(i) for i in range(4)],
        output_size=7, sampling_ratio=2
    )
    mask_pool = MultiScaleRoIAlign(
        featmap_names=[str(i) for i in range(4)],
        output_size=14, sampling_ratio=2
    )

    # assemble MaskRCNN
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_gen,
        box_roi_pool=box_pool,
        mask_roi_pool=mask_pool
    )
    return model


if __name__ == "__main__":
    # quick shape check
    model = get_polyp_pvt_maskrcnn()
    imgs = [torch.randn(1, 3, 512, 512)]
    # dummy target required for training mode
    target = [{
        "boxes": torch.tensor([[50, 50, 200, 200]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "masks": torch.zeros((1, 512, 512), dtype=torch.uint8)
    }]
    model.train()
    loss_dict = model(imgs, target)
    print({k: v.item() for k, v in loss_dict.items()})
