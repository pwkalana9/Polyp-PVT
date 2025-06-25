import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

# --- bring in PVT, CFM, SAM, plus attention blocks for CIM ---
from lib.pvt import PVT, CFM, SAM, ChannelAttention, SpatialAttention

class PolypPVTBackbone(nn.Module):
    """
    PVT → CFM branch  +  Channel+Spatial Attention (CIM) branch  → fuse  → SAM.
    Returns a dict of 4 feature maps to feed MaskRCNN.
    """
    def __init__(self,
                 pvt_args=None,
                 cfm_args=None,
                 sam_args=None):
        super().__init__()
        # encoder + fusion
        self.pvt = PVT(**(pvt_args or {}))
        self.cfm = CFM(**(cfm_args or {}))
        # build the parallel CIM branch from low-level features
        # PVT exposes its embed dims (e.g. [64,128,320,512]) via `.embed_dims` :contentReference[oaicite:0]{index=0}
        dims = getattr(self.pvt, 'embed_dims', None)
        if dims is None:
            raise ValueError("PVT must expose `embed_dims` for CIM construction")
        self.ca_layers = nn.ModuleList([ChannelAttention(d) for d in dims])
        self.sa_layers = nn.ModuleList([SpatialAttention()    for _ in dims])
        # similarity aggregation
        self.sam = SAM(**(sam_args or {}))

        # after CFM+fuse+CIM+SAM, final channel dim:
        # we assume SAM preserves channel count = dims[-1]
        self.out_channels = dims[-1]

    def forward(self, x):
        # 1) original multi-scale features from PVT
        feats = self.pvt(x)                # list of 4 tensors :contentReference[oaicite:1]{index=1}

        # 2) CFM branch on those features
        feats_cfm = self.cfm(feats)

        # 3) CIM branch: for each PVT feature apply Channel→Spatial attention 
        feats_cim = []
        for f, ca, sa in zip(feats, self.ca_layers, self.sa_layers):
            # channel attention multiplies per-channel weights
            f_ca = ca(f) * f
            # spatial attention multiplies per-spatial weights
            f_sa = sa(f_ca) * f_ca
            feats_cim.append(f_sa)
        # note: CIM captures "camouflaged" cues in low-level features :contentReference[oaicite:2]{index=2}

        # 4) fuse CFM + CIM
        feats_fused = [fc + fci for fc, fci in zip(feats_cfm, feats_cim)]

        # 5) similarity aggregation
        feats_out = self.sam(feats_fused)

        # return as dict[str→Tensor] for MaskRCNN RoIAlign
        return {str(i): feats_out[i] for i in range(len(feats_out))}


def get_polyp_pvt_maskrcnn(num_classes=2,
                           pvt_args=None,
                           cfm_args=None,
                           sam_args=None):
    # build backbone
    backbone = PolypPVTBackbone(pvt_args, cfm_args, sam_args)
    # MaskRCNN needs .out_channels
    backbone.out_channels = backbone.out_channels

    # RPN anchor generator (tune sizes/aspect_ratios to your dataset)
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    # RoIAlign for box and mask heads
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=[str(i) for i in range(4)],
        output_size=7, sampling_ratio=2
    )
    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=[str(i) for i in range(4)],
        output_size=14, sampling_ratio=2
    )

    # assemble MaskRCNN
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        mask_roi_pool=mask_roi_pool
    )
    return model


if __name__ == "__main__":
    # sanity check
    model = get_polyp_pvt_maskrcnn(num_classes=2)
    model.train()
    img = torch.randn(1, 3, 512, 512)
    # dummy target
    targets = [{
        "boxes": torch.tensor([[100,100,400,400]], dtype=torch.float32),
        "labels": torch.tensor([1], dtype=torch.int64),
        "masks": torch.zeros((1,512,512), dtype=torch.uint8).index_fill_(1, torch.arange(100,400), 1)
    }]
    losses = model([img], targets)
    print({k: v.item() for k,v in losses.items()})
