import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# -------------------------------------------------------------------
# 1) A naive Conv4D implemented via unfold on each axis + einsum
# -------------------------------------------------------------------
class Conv4d(nn.Module):
    """
    A minimal Conv4d: input shape [B, C_in, T, D, H, W],
    weight shape [C_out, C_in, kT, kD, kH, kW].
    """
    def __init__(self, 
                 in_channels, out_channels,
                 kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        # allow ints or 4-tuples
        self.kT, self.kD, self.kH, self.kW = _quad(kernel_size)
        self.sT, self.sD, self.sH, self.sW = _quad(stride)
        self.pT, self.pD, self.pH, self.pW = _quad(padding)

        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels,
                        self.kT, self.kD, self.kH, self.kW)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x):
        # x: [B,C_in,T,D,H,W]
        B, C, T, D, H, W = x.shape
        # pad
        x = F.pad(x, (self.pW,)*2 + (self.pH,)*2 + (self.pD,)*2 + (self.pT,)*2)
        # unfold each dim
        # after unfold_T: [B,C, outT, kT, D_pad, H_pad, W_pad]
        x = x.unfold(2, self.kT, self.sT)
        # unfold D
        x = x.unfold(3, self.kD, self.sD)
        # unfold H
        x = x.unfold(4, self.kH, self.sH)
        # unfold W
        x = x.unfold(5, self.kW, self.sW)
        # now x.shape = [B, C, outT, outD, outH, outW, kT, kD, kH, kW]
        B, C, outT, outD, outH, outW, kT, kD, kH, kW = x.shape
        # flatten the kernel dims into one
        x = x.contiguous().view(B, C, outT, outD, outH, outW, -1)  # C * kT*kD*kH*kW
        # reshape for matmul
        x = x.permute(0, 2, 3, 4, 5, 1, 6)  # [B, outT, outD, outH, outW, C, K]
        x = x.reshape(-1, C * kT * kD * kH * kW)                  # [B*outT*outD*outH*outW, C*K]

        w = self.weight.view(self.weight.size(0), -1)            # [C_out, C*K]
        out = x @ w.t()                                           # [B*... , C_out]
        if self.bias is not None:
            out = out + self.bias

        # reshape back
        out = out.view(B, outT, outD, outH, outW, -1)            # [B, T', D', H', W', C_out]
        out = out.permute(0, 5, 1, 2, 3, 4).contiguous()         # [B, C_out, T', D', H', W']
        return out

def _quad(x):
    if isinstance(x, int):
        return (x,)*4
    assert len(x) == 4
    return x

# -------------------------------------------------------------------
# 2) A toy “4D‐PVT” backbone replacing 2D patch‐embed with Conv4d
# -------------------------------------------------------------------
class PatchEmbed4D(nn.Module):
    """
    4D patch embed: reduces (T,D,H,W) → tokens + optional pooling
    """
    def __init__(self, in_chans, embed_dim,
                 kernel_size=(2,2,4,4), stride=(2,2,4,4), padding=(0,0,1,1)):
        super().__init__()
        self.proj = Conv4d(in_chans, embed_dim,
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, C, T, D, H, W]
        x = self.proj(x)  # [B, E, T', D', H', W']
        B, E, T, D, H, W = x.shape
        # flatten spatio-temporal dims to tokens
        x = x.flatten(2).transpose(1,2)  # [B, N, E], N=T'*D'*H'*W'
        x = self.norm(x)
        # (optionally) reshape back to 4D feature map
        x = x.transpose(1,2).view(B, E, T, D, H, W)
        return x

class Simple4DPVTBackbone(nn.Module):
    """
    A stub: one patch-embed + one Conv4d block + simple attention.
    """
    def __init__(self, in_chans=1, embed_dim=32):
        super().__init__()
        self.patch = PatchEmbed4D(in_chans, embed_dim)
        # a single 4D convolutional “block”
        self.conv4d = Conv4d(embed_dim, embed_dim,
                             kernel_size=(3,3,3,3),
                             stride=1, padding=1)
        # simple channel‐wise attention across 4D features
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool3d((None,1,1)),   # avg over D,H,W but keep T
            nn.Conv1d(embed_dim, embed_dim, 1),
            nn.Sigmoid()
        )
        self.out_channels = embed_dim

    def forward(self, x):
        # x: [B, C_in, T, D, H, W]
        x = self.patch(x)    # [B, E, T',D',H',W']
        f = self.conv4d(x)   # [B, E, T',D',H',W']
        # channel‐wise scale
        # collapse spatial dims to feed into Conv1d
        B, E, T, D, H, W = f.shape
        ctx = f.mean(dim=[3,4,5])            # [B, E, T]
        ctx = self.attn(ctx)                 # [B, E, T]
        ctx = ctx.view(B, E, T, 1,1,1)
        f = f * ctx
        return f

# -------------------------------------------------------------------
# 3) A stub “4D Mask R-CNN” API binding the 4D backbone
# -------------------------------------------------------------------
class MaskRCNN4D(nn.Module):
    """
    Sketch: takes a 4D backbone, then 4D→2D flatten & 2D Mask RCNN head.
    (Full 4D ROIAlign is out of scope here.)
    """
    def __init__(self, backbone4d, num_classes=2):
        super().__init__()
        self.backbone4d = backbone4d
        # after backbone → collapse T,D dims by pooling to get 2D maps
        self.pool4d = nn.AdaptiveAvgPool3d((None,1,1))  
        # now use a standard 2D MaskRCNN head
        from torchvision.models.detection import MaskRCNN
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        # dummy 2D backbone: will accept [B, C, H, W]
        # we wrap it so its forward(x) → feature dict for MaskRCNN
        class _FakeBack2d(nn.Module):
            def __init__(self, in_channels):
                super().__init__()
                self.out_channels = in_channels
            def forward(self, x):
                return OrderedDict([("0", x)])
        fake2d = _FakeBack2d(backbone4d.out_channels)
        self.maskrcnn2d = MaskRCNN(fake2d, num_classes=num_classes)
        # replace predictors (same as before)
        in_feat = self.maskrcnn2d.roi_heads.box_predictor.cls_score.in_features
        self.maskrcnn2d.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
        in_feat_mask = self.maskrcnn2d.roi_heads.mask_predictor.conv5_mask.in_channels
        self.maskrcnn2d.roi_heads.mask_predictor = MaskRCNNPredictor(in_feat_mask, 256, num_classes)

    def forward(self, imgs, targets=None):
        # imgs: list[Tensors] each of shape [B,C,T,D,H,W]
        # for simplicity assume batch of 1
        x4d = imgs[0]                      # [B,C,T,D,H,W]
        f4d = self.backbone4d(x4d)         # [B,E,T',D',H',W']
        # collapse T',D' via pooling → get [B,E,H',W']
        f2d = self.pool4d(f4d).squeeze(-1).squeeze(-1)  # [B,E,H',W']
        # feed into 2D Mask R-CNN
        return self.maskrcnn2d([f2d], targets)

# -------------------------------------------------------------------
# 4) Example usage
# -------------------------------------------------------------------
if __name__ == "__main__":
    # synthetic 4D radar + time input: [B=1, C=1, T=10, D=8, H=64, W=64]
    inp = torch.randn(1,1,10,8,64,64)
    backbone4d = Simple4DPVTBackbone(in_chans=1, embed_dim=32)
    model4d    = MaskRCNN4D(backbone4d, num_classes=2)

    # dummy target
    tgt = [{
      "boxes": torch.tensor([[10,10,50,50]], dtype=torch.float32),
      "labels":torch.tensor([1],dtype=torch.int64),
      "masks": torch.zeros((1,64,64),dtype=torch.uint8)
    }]
    # forward
    losses = model4d([inp], tgt)
    print(losses)
