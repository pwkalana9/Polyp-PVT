import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from collections import OrderedDict

# -----------------------------------------------------------------------------
# 1) True 4-D Conv (from previous)
# -----------------------------------------------------------------------------
def _quad(x):
    return (x,)*4 if isinstance(x, int) else tuple(x)

class Conv4d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.kT, self.kD, self.kH, self.kW = _quad(kernel_size)
        self.sT, self.sD, self.sH, self.sW = _quad(stride)
        self.pT, self.pD, self.pH, self.pW = _quad(padding)
        self.weight = nn.Parameter(torch.randn(
            out_c, in_c, self.kT, self.kD, self.kH, self.kW))
        self.bias = nn.Parameter(torch.zeros(out_c)) if bias else None

    def forward(self, x):
        B,C,T,D,H,W = x.shape
        x = F.pad(x, (self.pW,)*2 + (self.pH,)*2 + (self.pD,)*2 + (self.pT,)*2)
        x = x.unfold(2, self.kT, self.sT).unfold(3, self.kD, self.sD)
        x = x.unfold(4, self.kH, self.sH).unfold(5, self.kW, self.sW)
        B,C,oT,oD,oH,oW,kT,kD,kH,kW = x.shape
        x = x.contiguous().view(B, C, oT, oD, oH, oW, -1)
        x = x.permute(0,2,3,4,5,1,6).reshape(-1, C * kT*kD*kH*kW)
        w = self.weight.view(self.weight.size(0), -1)
        out = x @ w.t()
        if self.bias is not None: out += self.bias
        out = out.view(B, oT, oD, oH, oW, -1).permute(0,5,1,2,3,4)
        return out

# -----------------------------------------------------------------------------
# 2) 4-D Anchor Generator
# -----------------------------------------------------------------------------
class AnchorGenerator4D(nn.Module):
    def __init__(self, anchor_sizes, aspect_ratios):
        super().__init__()
        # anchor_sizes: list of tuples (t_size, d_size, h_size, w_size)
        # aspect_ratios: list of 4-tuples (t_ratio, d_ratio, h_ratio, w_ratio)
        self.sizes = anchor_sizes
        self.ratios = aspect_ratios

    def grid_anchors(self, feature_shape, stride):
        # feature_shape: (T,D,H,W), stride: (sT,sD,sH,sW)
        T,D,H,W = feature_shape
        sT,sD,sH,sW = stride
        # generate center coordinates
        t_centers = torch.arange(T, dtype=torch.float32, device=stride.device) * sT + sT/2
        d_centers = torch.arange(D, dtype=torch.float32, device=stride.device) * sD + sD/2
        h_centers = torch.arange(H, dtype=torch.float32, device=stride.device) * sH + sH/2
        w_centers = torch.arange(W, dtype=torch.float32, device=stride.device) * sW + sW/2
        grid = torch.meshgrid(t_centers, d_centers, h_centers, w_centers, indexing='ij')
        tc, dc, hc, wc = [g.reshape(-1) for g in grid]
        anchors = []
        for size in self.sizes:
            t_size, d_size, h_size, w_size = size
            for ratio in self.ratios:
                rt, rd, rh, rw = ratio
                anchors.append(torch.stack([
                    tc - t_size*rt/2, dc - d_size*rd/2,
                    hc - h_size*rh/2, wc - w_size*rw/2,
                    tc + t_size*rt/2, dc + d_size*rd/2,
                    hc + h_size*rh/2, wc + w_size*rw/2
                ], dim=1))
        return torch.cat(anchors, dim=0)

# -----------------------------------------------------------------------------
# 3) RPN Head 4-D
# -----------------------------------------------------------------------------
class RPNHead4D(nn.Module):
    def __init__(self, in_channels, num_anchors):
        super().__init__()
        self.conv = Conv4d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_logits = Conv4d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred  = Conv4d(in_channels, num_anchors*8, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B,C,T,D,H,W]
        t = self.relu(self.conv(x))
        logits = self.cls_logits(t)      # [B, A, T,D,H,W]
        bbox_d = self.bbox_pred(t)       # [B, A*8, T,D,H,W]
        return logits, bbox_d

# -----------------------------------------------------------------------------
# 4) Helpers: apply deltas, IoU, NMS in 4D
# -----------------------------------------------------------------------------
def apply_4d_deltas(boxes, deltas):
    # boxes: [N,8], deltas: [N,8]
    return boxes + deltas

def box_iou_4d(a, b):
    # both: [8] -> IoU scalar
    t1,d1,h1,w1,t2,d2,h2,w2 = a; T1,D1,H1,W1,T2,D2,H2,W2 = b
    i_t1, i_d1, i_h1, i_w1 = max(t1,T1), max(d1,D1), max(h1,H1), max(w1,W1)
    i_t2, i_d2, i_h2, i_w2 = min(t2,T2), min(d2,D2), min(h2,H2), min(w2,W2)
    if i_t2<=i_t1 or i_d2<=i_d1 or i_h2<=i_h1 or i_w2<=i_w1:
        return 0.0
    inter = (i_t2-i_t1)*(i_d2-i_d1)*(i_h2-i_h1)*(i_w2-i_w1)
    volA = (t2-t1)*(d2-d1)*(h2-h1)*(w2-w1)
    volB = (T2-T1)*(D2-D1)*(H2-H1)*(W2-W1)
    return inter / (volA + volB - inter + 1e-6)

def nms_4d(boxes, scores, iou_threshold):
    idxs = scores.argsort(descending=True)
    keep = []
    while idxs.numel() > 0:
        i = idxs[0].item(); keep.append(i)
        if idxs.numel()==1: break
        rest = idxs[1:]
        ious = torch.tensor([box_iou_4d(boxes[i], boxes[j]) for j in rest], device=boxes.device)
        idxs = rest[ious<=iou_threshold]
    return keep

# -----------------------------------------------------------------------------
# 5) 4-D RPN & Proposal Generation
# -----------------------------------------------------------------------------
def rpn_4d(anchor_gen, rpn_head, features, strides, pre_nms_top_n=1000, nms_thresh=0.7):
    # features: OrderedDict level->[B,C,T,D,H,W]
    batch_size = next(iter(features.values())).shape[0]
    all_anchors = []
    obj_logits = []
    bbox_deltas = []
    for lvl, feat in features.items():
        stride = strides[lvl]
        anchors = anchor_gen.grid_anchors(feat.shape[-4:], torch.tensor(stride, device=feat.device))
        anchors = anchors.unsqueeze(0).repeat(batch_size,1,1)  # [B,N,8]
        logits, deltas = rpn_head(feat)
        B,A,T,D,H,W = logits.shape
        logits = logits.permute(0,1,2,3,4,5).reshape(B, -1)
        deltas = deltas.permute(0,1,2,3,4,5).reshape(B, -1, 8)
        all_anchors.append(anchors)
        obj_logits.append(logits)
        bbox_deltas.append(deltas)
    all_anchors = torch.cat(all_anchors, dim=1)   # [B, M,8]
    obj_logits  = torch.cat(obj_logits, dim=1)    # [B, M]
    bbox_deltas = torch.cat(bbox_deltas, dim=1)   # [B, M,8]

    proposals = []
    for b in range(batch_size):
        scores = obj_logits[b].sigmoid()
        deltas = bbox_deltas[b]
        anchors = all_anchors[b]
        # top-k
        num_ = min(pre_nms_top_n, scores.size(0))
        topk = scores.topk(num_, sorted=True)
        idx = topk.indices
        anchors_t = anchors[idx]
        deltas_t  = deltas[idx]
        scores_t  = scores[idx]
        # decode
        props = apply_4d_deltas(anchors_t, deltas_t)
        # NMS
        keep = nms_4d(props, scores_t, nms_thresh)
        proposals.append(props[keep])
    return proposals  # list of [K,8]

# -----------------------------------------------------------------------------
# 6) Post-processing predictions
# -----------------------------------------------------------------------------
def postprocess_detections_4d(class_logits, box_deltas, mask_logits, proposals, score_thresh=0.5, iou_thresh=0.5):
    # class_logits: [N,cls], box_deltas: [N,cls*8], mask_logits: [N,cls,Ot,Od,Oh,Ow]
    device = class_logits.device
    N, C = class_logits.shape
    num_classes = C
    results = []
    scores = F.softmax(class_logits, -1)
    for cls in range(1, num_classes):
        cls_scores = scores[:, cls]
        keep = cls_scores > score_thresh
        if keep.sum()==0: continue
        cls_boxes  = box_deltas.view(N, num_classes,8)[keep, cls]
        cls_scores = cls_scores[keep]
        cls_masks  = mask_logits[keep, cls]  # [M,Ot,Od,Oh,Ow]
        # apply deltas to proposals
        idxs = keep.nonzero().view(-1)
        props = proposals[idxs]
        boxes = apply_4d_deltas(props, cls_boxes)
        # nms
        keep2 = nms_4d(boxes, cls_scores, iou_thresh)
        for i in keep2:
            mask = cls_masks[i].sigmoid() > 0.5
            results.append({
                "class": cls,
                "score": cls_scores[i].item(),
                "box4d": boxes[i].tolist(),
                "mask4d": mask.cpu().numpy()
            })
    return results

# -----------------------------------------------------------------------------
# Usage within MaskRCNN4D:
#   proposals = rpn_4d(anchor_gen, rpn_head, features, strides)
#   detections = postprocess_detections_4d(...)
# -----------------------------------------------------------------------------
