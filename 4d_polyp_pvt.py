import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

# -----------------------------------------------------------------------------
# 1) True 4D convolution (Conv4d)
# -----------------------------------------------------------------------------
def _quad(x):
    return (x,)*4 if isinstance(x, int) else tuple(x)

class Conv4d(nn.Module):
    """
    A minimal Conv4d: input [B,C_in,T,D,H,W],
    weight [C_out,C_in,kT,kD,kH,kW].
    """
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.kT, self.kD, self.kH, self.kW = _quad(kernel_size)
        self.sT, self.sD, self.sH, self.sW = _quad(stride)
        self.pT, self.pD, self.pH, self.pW = _quad(padding)

        self.weight = nn.Parameter(torch.randn(
            out_c, in_c, self.kT, self.kD, self.kH, self.kW))
        self.bias = nn.Parameter(torch.zeros(out_c)) if bias else None

    def forward(self, x):
        # x: [B,C,T,D,H,W]
        B,C,T,D,H,W = x.shape
        # pad in (W,H,D,T) order for F.pad
        x = F.pad(x, (self.pW,)*2 + (self.pH,)*2 + (self.pD,)*2 + (self.pT,)*2)
        # unfold each spatial dim
        x = x.unfold(2, self.kT, self.sT)  # -> [B,C,outT,kT,D',H',W']
        x = x.unfold(3, self.kD, self.sD)
        x = x.unfold(4, self.kH, self.sH)
        x = x.unfold(5, self.kW, self.sW)
        # collapse kernel dims
        B,C,oT,oD,oH,oW,kT,kD,kH,kW = x.shape
        x = x.contiguous().view(B, C, oT, oD, oH, oW, -1)    # C * kT*kD*kH*kW
        x = x.permute(0,2,3,4,5,1,6).reshape(-1, C * kT*kD*kH*kW)
        w = self.weight.view(self.weight.size(0), -1)
        out = x @ w.t()
        if self.bias is not None:
            out += self.bias
        # reshape back to [B,oC,oT,oD,oH,oW]
        oC = self.weight.size(0)
        out = out.view(B, oT, oD, oH, oW, oC).permute(0,5,1,2,3,4)
        return out

# -----------------------------------------------------------------------------
# 2) 4D RoI Align (crop + adaptive pooling)
# -----------------------------------------------------------------------------
class ROIAlign4D(nn.Module):
    """
    Crops each 4D box (t1,d1,h1,w1,t2,d2,h2,w2) and adaptively pools to (Ot,Od,Oh,Ow).
    boxes list: one Tensor[K,8] per image in the batch.
    """
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size  # tuple of 4 ints

    def forward(self, features, boxes_list):
        # features: [B,C,T,D,H,W]
        # boxes_list: list of length B, each [K,8] in feature-map coordinates
        B, C, T, D, H, W = features.shape
        Ot, Od, Oh, Ow = self.output_size
        pooled = []
        for b, boxes in enumerate(boxes_list):
            f = features[b:b+1]  # [1,C,T,D,H,W]
            for box in boxes:
                t1,d1,h1,w1,t2,d2,h2,w2 = box.round().long().tolist()
                region = f[..., t1:t2, d1:d2, h1:h2, w1:w2]  # [1,C,dt,dd,dh,dw]
                # adaptive pool along last 4 dims
                # 1) pool (d,h,w) → (Od,Oh,Ow)
                region = F.adaptive_max_pool3d(
                    region.flatten(2),  # merge T into batch dim
                    (Od,Oh,Ow)
                ).view(1, C, T, Od, Oh, Ow)
                # 2) pool T → Ot
                if Ot != T:
                    region = F.adaptive_max_pool3d(
                        region.permute(0,1,3,4,5,2).reshape(1, C*Od*Oh*Ow, T,1,1),
                        (Ot,1,1)
                    ).reshape(1, C, Ot, Od, Oh, Ow)
                pooled.append(region)
        if not pooled:
            return torch.zeros(0, C, Ot, Od, Oh, Ow, device=features.device)
        return torch.cat(pooled, dim=0)

# -----------------------------------------------------------------------------
# 3) 4D Box Head & Predictor
# -----------------------------------------------------------------------------
class BoxHead4D(nn.Module):
    """
    Two‐layer MLP on flattened pooled 4D features.
    """
    def __init__(self, in_channels, Ot, Od, Oh, Ow, representation_size=256):
        super().__init__()
        dim = in_channels * Ot * Od * Oh * Ow
        self.fc1 = nn.Linear(dim, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [N, C, Ot, Od, Oh, Ow]
        x = x.flatten(1)            # [N, C*Ot*Od*Oh*Ow]
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x  # [N, representation_size]

class BoxPredictor4D(nn.Module):
    """
    Predicts class logits and 8-D box deltas per class.
    """
    def __init__(self, representation_size, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(representation_size, num_classes)
        # 8 coords: (t1_offset,d1_offset,h1_offset,w1_offset, t2_,d2_,h2_,w2_)
        self.bbox_pred = nn.Linear(representation_size, num_classes * 8)

    def forward(self, x):
        scores = self.cls_score(x)            # [N, num_classes]
        deltas = self.bbox_pred(x)            # [N, num_classes*8]
        return scores, deltas

# -----------------------------------------------------------------------------
# 4) 4D Mask Head & Predictor
# -----------------------------------------------------------------------------
class MaskHead4D(nn.Module):
    """
    A two‐layer 4D conv head.
    """
    def __init__(self, in_channels, hidden_dim=128):
        super().__init__()
        self.conv1 = Conv4d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = Conv4d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x  # [N, hidden_dim, Ot, Od, Oh, Ow]

class MaskPredictor4D(nn.Module):
    """
    Projects to per‐class mask logits at the pooled resolution.
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.mask_fcn = Conv4d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.mask_fcn(x)  # [N, num_classes, Ot, Od, Oh, Ow]

# -----------------------------------------------------------------------------
# 5) 4D RoI Heads tying it all together
# -----------------------------------------------------------------------------
class RoIHeads4D(nn.Module):
    def __init__(self,
                 out_channels,            # C
                 box_output_size,         # (Ot,Od,Oh,Ow)
                 mask_output_size,        # (Ot,Od,Oh,Ow)
                 num_classes):
        super().__init__()
        OtB,OdB,OhB,OwB = box_output_size
        OtM,OdM,OhM,OwM = mask_output_size

        self.box_roi_pool = ROIAlign4D(box_output_size)
        self.box_head     = BoxHead4D(out_channels, *box_output_size)
        self.box_predictor = BoxPredictor4D(self.box_head.fc2.out_features, num_classes)

        self.mask_roi_pool = ROIAlign4D(mask_output_size)
        self.mask_head     = MaskHead4D(out_channels)
        self.mask_predictor= MaskPredictor4D(128, num_classes)

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        features: [B,C,T,D,H,W]
        proposals: list of length B, each Tensor[K,8] with boxes in feature-map coords
        """
        # 1) Box branch
        pooled_boxes = self.box_roi_pool(features, proposals)      # [N,C,OtB,OdB,OhB,OwB]
        box_feats    = self.box_head(pooled_boxes)                # [N,rep]
        class_logits, box_deltas = self.box_predictor(box_feats)  # [N,cls], [N,cls*8]

        # Compute box losses or detections here (omitted for brevity)...
        # 2) Mask branch (only on positive RoIs in training/inference)
        pooled_masks = self.mask_roi_pool(features, proposals)     # [N,C,OtM,OdM,OhM,OwM]
        mask_feats   = self.mask_head(pooled_masks)               # [N,128,OtM,OdM,OhM,OwM]
        mask_logits  = self.mask_predictor(mask_feats)            # [N,cls,OtM,OdM,OhM,OwM]

        # Return raw outputs; a full implementation would compute:
        #   - classification + box regression losses
        #   - mask loss
        #   - final detections & masks in 4D
        return {
            "class_logits": class_logits,
            "box_deltas": box_deltas,
            "mask_logits": mask_logits
        }

# -----------------------------------------------------------------------------
# 6) Full 4D MaskRCNN
# -----------------------------------------------------------------------------
class MaskRCNN4D(nn.Module):
    def __init__(self,
                 backbone4d,            # e.g. your Polyp-PVT 4D encoder + FPN
                 num_classes,
                 anchor_sizes=((8,), (16,), (32,), (64,)),
                 aspect_ratios=((1.0,),)*4,
                 box_output_size=(2,2,7,7),
                 mask_output_size=(2,2,14,14)):
        super().__init__()
        # 1) backbone + FPN
        self.body = backbone4d
        in_channels_list = [backbone4d.out_channels]*4
        self.fpn = FeaturePyramidNetwork(in_channels_list, backbone4d.out_channels)
        self.out_channels = backbone4d.out_channels

        # 2) RPN
        self.anchor_generator = AnchorGenerator(sizes=anchor_sizes,
                                                aspect_ratios=aspect_ratios)
        # ... (build your RPN here, omitted for brevity)

        # 3) 4D RoI Heads
        self.roi_heads = RoIHeads4D(self.out_channels,
                                    box_output_size,
                                    mask_output_size,
                                    num_classes)

    def forward(self, x, targets=None):
        # x: list of 4D tensors [B, C_in, T, D, H, W]
        features = self.body(x[0])               # OrderedDict of 4D maps p2..p5
        features = self.fpn(features)            # same keys
        # 4) RPN to generate 4D proposals (omitted; produce list of B [K,8] tensors)
        proposals = rpn_4d(self.anchor_generator, features, x)  # you’d implement this
        # 5) RoI heads
        return self.roi_heads(features_to_tensor(features),
                              proposals, image_shapes=None, targets=targets)

# -----------------------------------------------------------------------------
# NOTE
#  - You’ll need to implement `rpn_4d` to produce 4D box proposals.
#  - `features_to_tensor` should stack your OrderedDict levels into one 4D tensor
#    or pass them separately to `ROIAlign4D`.
#  - Loss/detection post-processing are left as an exercise.
# -----------------------------------------------------------------------------
