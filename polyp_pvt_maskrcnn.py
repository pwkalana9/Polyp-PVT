import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

# predictors to swap in
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn  import MaskRCNNPredictor

# your Polyp-PVT + FPN backbone from before:
from polyp_pvt_maskrcnn_backbone import get_polyp_pvt_fpn_backbone  

def get_polyp_pvt_instance_segmenter(num_classes=2,  # background + polyp
                                     hidden_layer=256,
                                     **backbone_kwargs):
    # 1) Build the backbone (with FPN) exactly as before
    backbone = get_polyp_pvt_fpn_backbone(**backbone_kwargs)
    
    # 2) Define RPN anchors & RoIAlign for box and mask:
    anchor_generator = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,)),  # one size per level
        aspect_ratios=((0.5,1.0,2.0),)*4
    )
    box_roi_pool = MultiScaleRoIAlign(
        featmap_names=["p2","p3","p4","p5"],
        output_size=7, sampling_ratio=2
    )
    mask_roi_pool = MultiScaleRoIAlign(
        featmap_names=["p2","p3","p4","p5"],
        output_size=14, sampling_ratio=2
    )

    # 3) Instantiate MaskRCNN (will include a default box & mask head)
    model = MaskRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=box_roi_pool,
        mask_roi_pool=mask_roi_pool
    )

    # 4) Replace the box predictor (class + bbox regression) to match `num_classes`
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box,
                                                     num_classes)

    # 5) Replace the mask predictor to match `num_classes`
    #    - conv5_mask.in_channels is the number of channels out of the mask head trunk
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

if __name__ == "__main__":
    # sanity check with a dummy image
    model = get_polyp_pvt_instance_segmenter(
        num_classes=2,
        backbone_kwargs=dict(channel=32,
                             pretrained_path='./pretrained_pth/pvt_v2_b2.pth')
    )
    model.train()
    imgs = [torch.randn(1,3,288,64)]
    targets = [{
        "boxes":  torch.tensor([[10, 10, 50, 80]], dtype=torch.float32),
        "labels": torch.tensor([1],           dtype=torch.int64),
        "masks":  torch.zeros((1,288,64),     dtype=torch.uint8)
    }]
    losses = model(imgs, targets)
    print({k: v.item() for k,v in losses.items()})
