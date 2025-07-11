import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from collections import OrderedDict

# -----------------------------------------------------------------------------
# 1) True 4D convolution (Conv4d)
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
# 2) 4D Anchor Generator
# -----------------------------------------------------------------------------
class AnchorGenerator4D(nn.Module):
    def __init__(self, anchor_sizes, aspect_ratios):
        super().__init__()
        self.sizes = anchor_sizes
        self.ratios = aspect_ratios

    def grid_anchors(self, feature_shape, stride):
        T,D,H,W = feature_shape
        sT,sD,sH,sW = stride
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
        t = self.relu(self.conv(x))
        logits = self.cls_logits(t)      # [B, A, T,D,H,W]
        bbox_d = self.bbox_pred(t)       # [B, A*8, T,D,H,W]
        return logits, bbox_d

# -----------------------------------------------------------------------------
# 4) Helpers: apply deltas, IoU, NMS in 4D
# -----------------------------------------------------------------------------
def apply_4d_deltas(boxes, deltas):
    return boxes + deltas

def box_iou_4d(a, b):
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
# 5) RPN Proposal Generation
# -----------------------------------------------------------------------------
def rpn_4d(anchor_gen, rpn_head, features, strides, pre_nms_top_n=500, nms_thresh=0.7):
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
        logits = logits.view(B, -1)
        deltas = deltas.view(B, -1, 8)
        all_anchors.append(anchors)
        obj_logits.append(logits)
        bbox_deltas.append(deltas)
    all_anchors = torch.cat(all_anchors, dim=1)
    obj_logits  = torch.cat(obj_logits, dim=1)
    bbox_deltas = torch.cat(bbox_deltas, dim=1)

    proposals = []
    for b in range(batch_size):
        scores = obj_logits[b].sigmoid()
        deltas = bbox_deltas[b]
        anchors= all_anchors[b]
        num_ = min(pre_nms_top_n, scores.size(0))
        topk = scores.topk(num_, sorted=True)
        idx = topk.indices
        anchors_t = anchors[idx]
        deltas_t  = deltas[idx]
        scores_t  = scores[idx]
        props = apply_4d_deltas(anchors_t, deltas_t)
        keep = nms_4d(props, scores_t, nms_thresh)
        proposals.append(props[keep])
    return proposals

# -----------------------------------------------------------------------------
# 6) Post-processing predictions
# -----------------------------------------------------------------------------
def postprocess_detections_4d(class_logits, box_deltas, mask_logits, proposals, score_thresh=0.5, iou_thresh=0.5):
    device = class_logits.device
    N, C = class_logits.shape
    results = []
    scores = F.softmax(class_logits, -1)
    for cls in range(1, C):
        cls_scores = scores[:, cls]
        keep = cls_scores > score_thresh
        if keep.sum()==0: continue
        cls_boxes = box_deltas.view(N, C, 8)[keep, cls]
        cls_scores= cls_scores[keep]
        cls_masks = mask_logits[keep, cls]
        idxs = keep.nonzero().view(-1)
        props = torch.vstack([p for p in proposals])
        boxes = apply_4d_deltas(props[idxs], cls_boxes)
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
# 7) Simplified Dataset for 4D Radar
# -----------------------------------------------------------------------------
class Radar4DDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')])
        self.data_dir = data_dir
        self.transforms = transforms

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        arr = torch.from_numpy(np.load(os.path.join(self.data_dir, self.files[idx]))).float()
        # arr shape: [T,D,H,W]
        # load annotations: boxes_list and masks_list per time-step
        ann = load_annotations(self.files[idx])
        images, targets = [], []
        for t in range(arr.shape[0]):
            img = arr[t]  # [D,H,W]
            boxes, masks = ann[t]
            targets.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.ones(len(boxes),dtype=torch.int64),
                "masks": torch.tensor(masks, dtype=torch.uint8)
            })
            images.append(img)
        # stack as [C,D,H,W] for batch=1
        return torch.stack(images).permute(1,0,2,3), targets

# -----------------------------------------------------------------------------
# 8) Complete MaskRCNN4D Model
# -----------------------------------------------------------------------------
class MaskRCNN4D(nn.Module):
    def __init__(self, backbone4d, num_classes):
        super().__init__()
        self.body = backbone4d
        in_ch_list = [backbone4d.out_channels]*4
        self.fpn = FeaturePyramidNetwork(in_ch_list, backbone4d.out_channels)
        self.rpn_head = RPNHead4D(backbone4d.out_channels, num_anchors=len(anchor_sizes)*len(anchor_ratios))
        self.anchor_generator = AnchorGenerator4D(anchor_sizes, anchor_ratios)
        self.roi_heads = RoIHeads4D(backbone4d.out_channels, box_output_size, mask_output_size, num_classes)

    def forward(self, x, targets=None):
        features = self.body(x)           # OrderedDict p2..p5
        features = self.fpn(features)
        proposals= rpn_4d(self.anchor_generator, self.rpn_head, features, strides)
        raw = self.roi_heads(features, proposals, None, targets)
        if self.training:
            return raw  # dict of losses
        return postprocess_detections_4d(**raw, proposals=proposals)

# -----------------------------------------------------------------------------
# 9) Training Loop
# -----------------------------------------------------------------------------
def train_one_epoch(model, optimizer, loader, device):
    model.train()
    for i, (images, targets) in enumerate(loader):
        images = images.to(device)     # [C,T,D,H,W]
        # wrap as list
        images = [images]
        targets = [{k:v.to(device) for k,v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Iter {i}: total_loss={losses.item():.4f}, " +
                  ", ".join([f"{k}={v.item():.4f}" for k,v in loss_dict.items()]))

# -----------------------------------------------------------------------------
# 10) Putting it all together
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # hyperparams
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 2
    epochs = 10

    # instantiate backbone4d (e.g. Simple4DPVTBackbone)
    backbone4d = Simple4DPVTBackbone(in_chans=D, embed_dim=32)
    model = MaskRCNN4D(backbone4d, num_classes).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # dataset & dataloader
    dataset = Radar4DDataset('data/radar_cubes')
    loader  = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        train_one_epoch(model, optimizer, loader, device)
        # optionally save checkpoint
        torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')
