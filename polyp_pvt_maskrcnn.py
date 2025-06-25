import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

# --- import the original Polyp‐PVT building blocks ---
from model import PVT, CFM, CIM  # adjust import paths if necessary

class PolypPVTBackbone(nn.Module):
    """
    Wraps PVT + CFM + CIM as a backbone producing a dict of multi-scale features
    to feed into torchvision MaskRCNN.
    """
    def __init__(self, pvt_args=None, cfm_args=None, cim_args=None):
        super().__init__()
        # initialize the original components
        self.pvt = PVT(**(pvt_args or {}))
        self.cfm = CFM(**(cfm_args or {}))
        self.cim = CIM(**(cim_args or {}))

        # determine number of output channels from CIM
        # assume CIM returns same channel dimensionality as its input feature maps
        # you may need to inspect CFM/CIM code to set this correctly:
        self.out_channels = self.cim.out_channels  

    def forward(self, x):
        # x: [B,3,H,W]
        feats = self.pvt(x)           # -> list of 4 tensors, shapes: [B,C_i,H/2^i,W/2^i]
        feats = self.cfm(feats)       # fuse high-level
        feats = self.cim(feats)       # refine low-level

        # return as dict[str->Tensor] for MaskRCNN
        # keys must match what you pass to RoIAlign (below)
        return {str(i): feats[i] for i in range(len(feats))}


def get_polyp_pvt_maskrcnn(num_classes=2,
                           pvt_args=None, cfm_args=None, cim_args=None):
    """
    Constructs a MaskRCNN model using PolypPVTBackbone.
    num_classes: includes background (so for polyp detection set to 2).
    """
    # 1) build the custom backbone
    backbone = PolypPVTBackbone(pvt_args, cfm_args, cim_args)
    # torchvision's MaskRCNN expects .out_channels attribute
    backbone.out_channels = backbone.out_channels

    # 2) RPN anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),       # one tuple per feature map
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    # 3) RoI Align for box heads
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=[str(i) for i in range(4)],  # matches backbone output keys
        output_size=7,
        sampling_ratio=2
    )

    # 4) RoI Align for mask heads
    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=[str(i) for i in range(4)],
        output_size=14,
        sampling_ratio=2
    )

    # 5) build the MaskRCNN model
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        mask_roi_pool=mask_roi_pool
    )

    return model


if __name__ == "__main__":
    # quick sanity check
    model = get_polyp_pvt_maskrcnn(num_classes=2)
    model.train()

    # dummy input and dummy target
    img = torch.randn(2, 3, 512, 512)
    targets = []
    for _ in range(2):
        # one object per image, box & binary mask
        boxes = torch.tensor([[50, 50, 300, 300]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)
        masks = torch.zeros((1, 512, 512), dtype=torch.uint8)
        masks[:, 50:300, 50:300] = 1
        targets.append({"boxes": boxes, "labels": labels, "masks": masks})

    # forward (returns losses in train mode)
    losses = model([img[0]], [targets[0]])
    print({k: v.item() for k, v in losses.items()})
