import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader

# Import everything from your other files
from pipeline import DOTADataset, DOTA_CLASSES, collate_fn, transforms
from backbone import VGGBackbone, FPN, device
from head import OBBHead
from visualize import apply_nms_and_filter, obb_to_corners

@torch.no_grad() # CRITICAL: Turns off training mode for massive memory savings
def run_inference():
    print("1. Loading Models...")
    backbone = VGGBackbone().to(device).eval()
    fpn = FPN().to(device).eval()
    head = OBBHead().to(device).eval()
    
    weight_file = "checkpoints/dota_weights_epoch_60.pth"
    if os.path.exists(weight_file):
        checkpoint = torch.load(weight_file, map_location=device)
        backbone.load_state_dict(checkpoint['backbone'])
        fpn.load_state_dict(checkpoint['fpn'])
        head.load_state_dict(checkpoint['head'])
    
    print("2. Loading Mini Dataset...")
    dota_root = 'DOTA_mini' 
    img_size = 1024
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    dataset = DOTADataset(dota_root, split="validation", transform=transform, target_size=img_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    print("3. Fetching Image...")
    images, gt_boxes, gt_labels = next(iter(loader))
    image = images[0].unsqueeze(0).to(device) # Add batch dimension
    
    print("4. Running Forward Pass...")
    c3, c4, c5 = backbone(image)
    p3, p4, p5, p6 = fpn(c3, c4, c5)
    features = [p3, p4, p5, p6]
    
    out_cls, out_ctr, out_reg = head(features)
    
    print("5. Decoding Predictions & Applying NMS...")
    all_boxes, all_cls, all_ctr = [], [], []
    
    # Decode the predictions from all 4 FPN levels
    for i in range(4):
        cls_lvl = out_cls[i][0] # (C, H, W)
        ctr_lvl = out_ctr[i][0] # (1, H, W)
        reg_lvl = out_reg[i][0] # (5, H, W) -> l, t, r, b, theta
        
        _, H, W = reg_lvl.shape
        stride = head.STRIDES[i]
        
        # Create grid centers for this feature map
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        grid_cx = (grid_x.float() + 0.5) * stride
        grid_cy = (grid_y.float() + 0.5) * stride
        
        # Flatten everything
        grid_cx = grid_cx.reshape(-1)
        grid_cy = grid_cy.reshape(-1)
        
        l = reg_lvl[0].reshape(-1)
        t = reg_lvl[1].reshape(-1)
        r = reg_lvl[2].reshape(-1)
        b = reg_lvl[3].reshape(-1)
        theta = reg_lvl[4].reshape(-1)
        
        # Convert l,t,r,b to cx, cy, w, h
        cx_pred = grid_cx + (r - l) / 2.0
        cy_pred = grid_cy + (b - t) / 2.0
        w_pred = l + r
        h_pred = t + b
        
        boxes_lvl = torch.stack([cx_pred, cy_pred, w_pred, h_pred, theta], dim=-1)
        
        all_boxes.append(boxes_lvl)
        all_cls.append(cls_lvl.permute(1, 2, 0).reshape(-1, 15))
        all_ctr.append(ctr_lvl.permute(1, 2, 0).reshape(-1, 1))

    # Combine all levels
    pred_boxes = torch.cat(all_boxes, dim=0)
    pred_cls = torch.cat(all_cls, dim=0)
    pred_ctr = torch.cat(all_ctr, dim=0)
    
    # Run your custom NMS!
    final_boxes, final_labels, final_scores = apply_nms_and_filter(
        pred_boxes, pred_cls, pred_ctr, conf_thresh=0.05, iou_thresh=0.1
    )
    print("Scores:", final_scores)
    print("Labels:", final_labels)
    
    print(f"6. Network predicted {len(final_boxes)} total objects after NMS.")
    
    print("7. Saving Output Image...")
    # Plotting
    img_pil = to_pil_image(images[0])
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_pil)
    
    # Draw Ground Truth boxes in GREEN (What it SHOULD see)
    for box in gt_boxes[0]:
        cx, cy, w, h, theta = box.tolist()
        corners = obb_to_corners(cx, cy, w, h, theta)
        poly = Polygon(corners, closed=True, edgecolor='lime', facecolor='none', linewidth=1, linestyle='--')
        ax.add_patch(poly)

    # Draw Model Predictions in RED (What it ACTUALLY sees)
    for box, label, score in zip(final_boxes, final_labels, final_scores):
        cx, cy, w, h, theta = box.tolist()
        corners = obb_to_corners(cx, cy, w, h, theta)
        poly = Polygon(corners, closed=True, edgecolor='red', facecolor='none', linewidth=2)
        ax.add_patch(poly)
        
        class_name = DOTA_CLASSES[label.item()]
        ax.text(corners[0][0], corners[0][1], f"{class_name} {score:.2f}", color='white', 
                bbox=dict(facecolor='red', alpha=0.5, pad=1, edgecolor='none'), fontsize=8)

    ax.axis('off')
    ax.set_title("Green = Ground Truth | Red = Model Predictions", fontsize=16)
    plt.savefig('inference_test60.png', bbox_inches='tight', dpi=150)
    print("Done! Saved as 'inference_test60.png'")

if __name__ == "__main__":
    run_inference()