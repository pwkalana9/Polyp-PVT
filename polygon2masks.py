import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

class PolypPolygonDataset(Dataset):
    def __init__(self, imgs_dir, ann_file, transforms=None):
        """
        imgs_dir: directory with your .jpg/.png images
        ann_file: path to a JSON file containing a list of annotations, e.g.:
            [
              {
                "image_id": "IMG_001.jpg",
                "height": 288,
                "width": 64,
                "objects": [
                  { "polygon": [[x1,y1],[x2,y2],…,[xn,yn]] },
                  …
                ]
              },
              …
            ]
        """
        self.imgs_dir   = imgs_dir
        self.transforms = transforms

        # Load all annotations into a dict keyed by image_id
        with open(ann_file) as f:
            records = json.load(f)
        self.anns = { r["image_id"]: r for r in records }
        self.ids  = list(self.anns.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        # 1) load image
        image_id = self.ids[idx]
        ann      = self.anns[image_id]
        img_path = os.path.join(self.imgs_dir, image_id)
        img      = Image.open(img_path).convert("RGB")

        # prepare target containers
        boxes  = []
        masks  = []
        labels = []

        # 2) for each polygon → box + mask
        for obj in ann["objects"]:
            poly = obj["polygon"]  # list of [x, y]
            xs   = [p[0] for p in poly]
            ys   = [p[1] for p in poly]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            boxes.append([xmin, ymin, xmax, ymax])

            # rasterize polygon to a (H, W) mask
            mask_img = Image.new("L", (ann["width"], ann["height"]), 0)
            ImageDraw.Draw(mask_img).polygon(poly, outline=1, fill=1)
            mask = np.array(mask_img, dtype=np.uint8)
            masks.append(mask)

            # all labels = 1 (polyp); extend if you have multiple classes
            labels.append(1)

        # convert lists to tensors
        boxes  = torch.as_tensor(boxes, dtype=torch.float32)      # [N,4]
        masks  = torch.as_tensor(masks, dtype=torch.uint8)        # [N,H,W]
        labels = torch.as_tensor(labels, dtype=torch.int64)       # [N]

        image_id_tensor = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes":       boxes,
            "labels":      labels,
            "masks":       masks,
            "image_id":    image_id_tensor,
            "area":        area,
            "iscrowd":     iscrowd,
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        # convert PIL→Tensor
        img = torch.as_tensor(np.array(img), dtype=torch.float32).permute(2,0,1) / 255.0

        return img, target
