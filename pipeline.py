import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from backbone import VGGBackbone, FPN, device
from head import OBBHead
from loss import gwd_loss
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

DOTA_CLASSES = [
    'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    'basketball-court', 'storage-tank', 'soccer-ball-field',
    'roundabout', 'harbor', 'swimming-pool', 'helicopter'
]
CLASS_TO_IDX = {name: i for i, name in enumerate(DOTA_CLASSES)}

SPLIT_DIRS = {
    'train':      {'images': 'images',  'labels': 'labelTxt'},
    'validation': {'images': 'images',  'labels': os.path.join('labelTxt', 'labelTxt')},
    'test':       {'images': 'images',  'labels': None},
}


def corners_to_obb(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Convert 4 corner coordinates to (cx, cy, w, h, theta).
    Uses OpenCV's minAreaRect for a robust conversion.
    """
    pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], dtype=np.float32)
    (cx, cy), (w, h), angle_deg = cv2_minAreaRect(pts)
    # OpenCV angle is in degrees, convert to radians and clamp to (-π/2, 0]
    theta = math.radians(angle_deg)
    # Normalise θ into (-π/2, 0]
    while theta > 0:
        theta -= math.pi
    while theta <= -math.pi / 2:
        theta += math.pi
    return cx, cy, w, h, theta


def cv2_minAreaRect(pts):
    """Pure-numpy minimum area rect (avoids cv2 dependency at import time)."""
    try:
        import cv2
        return cv2.minAreaRect(pts)
    except ImportError:
        # Fallback: simple bounding approach
        cx, cy = pts.mean(axis=0)
        d1 = np.linalg.norm(pts[1] - pts[0])
        d2 = np.linalg.norm(pts[2] - pts[1])
        dx, dy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
        angle = math.degrees(math.atan2(dy, dx))
        return (cx, cy), (d1, d2), angle


def parse_dota_annotation(ann_path, orig_w, orig_h, target_size=1024):
    """
    Parse a DOTA label .txt file.
    Returns:
        boxes  – (N, 5) tensor  (cx, cy, w, h, theta) in resized coords
        labels – (N,)   tensor  class indices
    """
    boxes, labels = [], []
    sx, sy = target_size / orig_w, target_size / orig_h

    with open(ann_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip header lines (imagesource, gsd, or empty)
            if not line or ':' in line:
                continue
            parts = line.split()
            if len(parts) < 10:
                continue
            coords = list(map(float, parts[:8]))
            class_name = parts[8]
            # difficulty = int(parts[9])

            if class_name not in CLASS_TO_IDX:
                continue

            # Scale corners to the resized image
            x1, y1 = coords[0] * sx, coords[1] * sy
            x2, y2 = coords[2] * sx, coords[3] * sy
            x3, y3 = coords[4] * sx, coords[5] * sy
            x4, y4 = coords[6] * sx, coords[7] * sy

            cx, cy, w, h, theta = corners_to_obb(x1, y1, x2, y2, x3, y3, x4, y4)
            boxes.append([cx, cy, w, h, theta])
            labels.append(CLASS_TO_IDX[class_name])

    if boxes:
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)
    else:
        return torch.zeros((0, 5), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)



class DOTADataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, target_size=1024):
        """
        root_dir: path to DOTA root (e.g. 'DOTA/')
        split: one of 'train', 'validation', 'test'
        Expected structure:
            DOTA/train/images/       DOTA/train/labelTxt/
            DOTA/validation/images/  DOTA/validation/labelTxt/labelTxt/
            DOTA/test/images/
        """
        dirs = SPLIT_DIRS[split]
        self.img_dir = os.path.join(root_dir, split, dirs['images'])
        self.ann_dir = os.path.join(root_dir, split, dirs['labels']) if dirs['labels'] else None
        self.split = split
        self.transform = transform
        self.target_size = target_size
        self.img_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size

        if self.transform:
            image = self.transform(image)

        # Load annotations (if available)
        if self.ann_dir is not None:
            ann_name = os.path.splitext(img_name)[0] + '.txt'
            ann_path = os.path.join(self.ann_dir, ann_name)
            if os.path.exists(ann_path):
                boxes, labels = parse_dota_annotation(
                    ann_path, orig_w, orig_h, self.target_size
                )
            else:
                boxes = torch.zeros((0, 5), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.zeros((0, 5), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        return image, boxes, labels


def run_epoch(dataloader, backbone, fpn, head, img_size, optimizer=None, split="train"):
    """Run one epoch of training, validation, or testing."""
    is_train = (split == "train")
    if is_train:
        backbone.train()
        fpn.train()
        head.train()
    else:
        backbone.eval()
        fpn.eval()
        head.eval()

    total_loss = 0.0
    num_batches = 0

    context = torch.no_grad() if not is_train else torch.enable_grad()
    with context:
        for images, gt_boxes_list, gt_labels_list in dataloader:
            images = images.to(device)
            gt_boxes_list = [b.to(device) for b in gt_boxes_list]
            gt_labels_list = [l.to(device) for l in gt_labels_list]

            c3, c4, c5 = backbone(images)
            p3, p4, p5, p6 = fpn(c3, c4, c5)
            features = [p3, p4, p5, p6]
            out_cls, out_ctr, out_reg = head(features)

            if split in ["test", "validation"]:
                batch_pred_boxes = []
                batch_pred_labels = []
                batch_pred_scores = []
                
                # We need to process each image in the batch individually
                for b in range(B):
                    all_boxes, all_cls, all_ctr = [], [], []
                    
                    # 1. Loop through all 4 FPN levels (P3, P4, P5, P6)
                    for i in range(4):
                        stride = head.STRIDES[i]
                        _, _, H, W = out_reg[i].shape
                        
                        # Generate the grid centers for this feature map
                        grid_y, grid_x = torch.meshgrid(
                            torch.arange(H, device=device), 
                            torch.arange(W, device=device),
                            indexing='ij'
                        )
                        grid_cx = (grid_x.float() + 0.5) * stride
                        grid_cy = (grid_y.float() + 0.5) * stride
                        
                        # Flatten and extract predictions for this image
                        reg_flat = out_reg[i][b].permute(1, 2, 0).reshape(-1, 5)
                        cls_flat = out_cls[i][b].permute(1, 2, 0).reshape(-1, len(DOTA_CLASSES))
                        ctr_flat = out_ctr[i][b].permute(1, 2, 0).reshape(-1, 1)
                        
                        # Decode (l, t, r, b, theta) into (cx, cy, w, h, theta)
                        l, t, r, b_var, theta = reg_flat.unbind(dim=-1)
                        cx_pred = grid_cx.flatten() + (r - l) / 2.0
                        cy_pred = grid_cy.flatten() + (b_var - t) / 2.0
                        w_pred = l + r
                        h_pred = t + b_var
                        
                        decoded_boxes = torch.stack([cx_pred, cy_pred, w_pred, h_pred, theta], dim=-1)
                        
                        all_boxes.append(decoded_boxes)
                        all_cls.append(cls_flat)
                        all_ctr.append(ctr_flat)
                        
                    # 2. Concatenate all levels together
                    all_boxes = torch.cat(all_boxes, dim=0)
                    all_cls = torch.cat(all_cls, dim=0)
                    all_ctr = torch.cat(all_ctr, dim=0)
                    
                    # 3. Apply the NMS and Filtering (from Step 2)
                    from visualize import apply_nms_and_filter # Assuming you put it in visualize.py
                    clean_boxes, clean_labels, clean_scores = apply_nms_and_filter(
                        all_boxes, all_cls, all_ctr, conf_thresh=0.6, iou_thresh=0.01
                    ) # conf_thresh = 0.05 iou_thresh = 0.1
                    
                    batch_pred_boxes.append(clean_boxes)
                    batch_pred_labels.append(clean_labels)
                    batch_pred_scores.append(clean_scores)
                    
                continue 
            
            reg0 = out_reg[0]  
            B, C, H, W = reg0.shape
            
            reg_flat = reg0.permute(0, 2, 3, 1).reshape(B, -1, 5)  
            
            stride = img_size // H  # P3 stride
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=device), 
                torch.arange(W, device=device),
                indexing='ij'
            )
            grid_centers = torch.stack([
                (grid_x.float() + 0.5) * stride,
                (grid_y.float() + 0.5) * stride
            ], dim=-1).reshape(-1, 2)  # (H*W, 2)
            
            batch_loss = 0.0
            valid_images = 0

            # Process loss per image to avoid cross-batch index mixing
            for b in range(B):
                gt_boxes = gt_boxes_list[b]
                if gt_boxes.shape[0] == 0:
                    continue
                
                valid_images += 1
                gt_centers = gt_boxes[:, :2]  # (n_gt, 2)
                
                # Match Ground Truth to nearest grid center
                dists = torch.cdist(gt_centers, grid_centers)
                nearest_idx = dists.argmin(dim=1)  # (n_gt,)
                
                # Extract predictions and matched anchor coordinates
                pred_raw = reg_flat[b][nearest_idx]  # (n_gt, 5) -> [l, t, r, b, theta]
                matched_centers = grid_centers[nearest_idx]  # (n_gt, 2) -> [grid_cx, grid_cy]
                
                # --- FIX 1: Decode (l, t, r, b) into (cx, cy, w, h) ---
                l = pred_raw[:, 0]
                t = pred_raw[:, 1]
                r = pred_raw[:, 2]
                b_var = pred_raw[:, 3]
                theta = pred_raw[:, 4]
                
                grid_cx = matched_centers[:, 0]
                grid_cy = matched_centers[:, 1]
                
                cx_pred = grid_cx + (r - l) / 2.0
                cy_pred = grid_cy + (b_var - t) / 2.0
                w_pred = l + r
                h_pred = t + b_var
                
                # Stack back into format expected by loss.py: [cx, cy, w, h, theta]
                pred_boxes_obb = torch.stack([cx_pred, cy_pred, w_pred, h_pred, theta], dim=-1)
                
                batch_loss += gwd_loss(pred_boxes_obb, gt_boxes)

            if valid_images > 0:
                loss = batch_loss / valid_images
                
                if is_train and optimizer is not None:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                total_loss += loss.item()
                num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    print(f"[{split.upper()}] Avg Loss: {avg_loss:.4f}")
    return avg_loss

def collate_fn(batch):
    """Custom collate: images are stacked, boxes/labels stay as lists (variable length)."""
    images, boxes, labels = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(boxes), list(labels)


def main():

    import os
    dota_root = 'DOTA'
    print(f"Using dataset: {dota_root}")
    
    num_epochs = 100
    lr = 1e-4

    # Use smaller images on CPU for testing, full size on GPU
    img_size = 1024 if torch.cuda.is_available() else 256
    batch_size = 32 if torch.cuda.is_available() else 1
    num_workers = 2 if torch.cuda.is_available() else 0
    print(f"Image size: {img_size}, Batch size: {batch_size}, Device: {device}")


    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    
    train_dataset = DOTADataset(dota_root, split="train",      transform=transform, target_size=img_size)
    val_dataset   = DOTADataset(dota_root, split="validation", transform=transform, target_size=img_size)
    test_dataset  = DOTADataset(dota_root, split="test",       transform=transform, target_size=img_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, collate_fn=collate_fn)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

  
    backbone = VGGBackbone().to(device)
    fpn = FPN().to(device)
    head = OBBHead().to(device)


    params = list(backbone.parameters()) + list(fpn.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam([p for p in params if p.requires_grad], lr=lr)

    
    for epoch in range(1, num_epochs + 1):
        print(f"\n=== Epoch {epoch}/{num_epochs} ===")
        run_epoch(train_loader, backbone, fpn, head, img_size, optimizer=optimizer, split="train")
        run_epoch(val_loader,   backbone, fpn, head, img_size, optimizer=None,      split="validation")

    print("\n=== Final Test ===")
    run_epoch(test_loader, backbone, fpn, head, img_size, optimizer=None, split="test")


if __name__ == "__main__":
    main()