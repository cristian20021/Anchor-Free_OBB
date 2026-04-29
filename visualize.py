import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader
from pipeline import DOTADataset, DOTA_CLASSES, collate_fn, transforms
from backbone import image_loader
from loss import rotated_iou

def obb_to_corners(cx, cy, w, h, theta):
    """
    Converts (cx, cy, w, h, theta) back to 4 corner coordinates.
    Assumes theta is in radians.
    """
    c, s = math.cos(theta), math.sin(theta)
    
    # Calculate corner offsets relative to the center
    dx = w / 2.0
    dy = h / 2.0
    
    # Define the 4 corners relative to the center (unrotated)
    corners = np.array([
        [-dx, -dy],
        [ dx, -dy],
        [ dx,  dy],
        [-dx,  dy]
    ])
    
    # Rotation matrix
    R = np.array([
        [ c, -s],
        [ s,  c]
    ])
    
    rotated_corners = np.dot(corners, R.T)
    return rotated_corners + np.array([cx, cy])

def apply_nms_and_filter(pred_boxes, pred_cls, pred_ctr, conf_thresh=0.05, iou_thresh=0.1, top_k=1000):

# 1. Convert logits to probabilities using Sigmoid
    cls_probs = torch.sigmoid(pred_cls)
    ctr_probs = torch.sigmoid(pred_ctr)
    
    # 2. Get the max class probability and its corresponding class index
    max_cls_probs, class_ids = torch.max(cls_probs, dim=1)
    
    # 3. Centerness Weighting: The secret to high mAP in anchor-free models
    # Multiply class confidence by centerness to get the final score
    final_scores = torch.sqrt(max_cls_probs * ctr_probs.squeeze(-1))
    
    # 4. Confidence Thresholding: Drop garbage predictions to speed up NMS
    keep_mask = final_scores > conf_thresh
    boxes = pred_boxes[keep_mask]
    scores = final_scores[keep_mask]
    labels = class_ids[keep_mask]
    
    if len(boxes) == 0:
        return boxes, labels, scores

    if len(boxes) > top_k:
        top_k_indices = torch.topk(scores, top_k)[1]
        boxes = boxes[top_k_indices]
        scores = scores[top_k_indices]
        labels = labels[top_k_indices]
    
    # 5. Greedy Rotated NMS (Using your custom rotated_iou)
    keep_indices = []
    
    # Sort boxes by score in descending order
    sorted_indices = torch.argsort(scores, descending=True)
    
    while len(sorted_indices) > 0:
        current_idx = sorted_indices[0]
        keep_indices.append(current_idx.item())
        
        if len(sorted_indices) == 1:
            break
            
        # Calculate IoU between the current highest-scoring box and the rest
        current_box = boxes[current_idx]
        rest_indices = sorted_indices[1:]
        
        ious = torch.tensor([
            rotated_iou(current_box, boxes[idx]) for idx in rest_indices
        ], device=boxes.device)
        
        # Keep only the boxes that have an IoU LOWER than the threshold
        # (meaning they don't heavily overlap with the current box)
        overlapping_mask = ious <= iou_thresh
        sorted_indices = rest_indices[overlapping_mask]

    return boxes[keep_indices], labels[keep_indices], scores[keep_indices]

def visualize_batch(images, boxes_list, labels_list, class_names=DOTA_CLASSES):
    """
    Visualizes a batch of images with their oriented bounding boxes.
    """
    batch_size = len(images)
    fig, axes = plt.subplots(1, batch_size, figsize=(6 * batch_size, 6))
    
    # Handle single-item batches gracefully
    if batch_size == 1:
        axes = [axes]
        
    for i in range(batch_size):
        ax = axes[i]
        
        # Convert image tensor back to PIL for plotting
        img = to_pil_image(images[i])
        ax.imshow(img)
        
        boxes = boxes_list[i]
        labels = labels_list[i]
        
        for box, label in zip(boxes, labels):
            cx, cy, w, h, theta = box.tolist()
            
            # Get corners for the polygon
            corners = obb_to_corners(cx, cy, w, h, theta)
            
            # Draw the bounding box
            poly = Polygon(corners, closed=True, edgecolor='lime', facecolor='none', linewidth=1)
            ax.add_patch(poly)
            
            # Draw the label background and text
            class_name = class_names[label.item()]
            
            # Place the text near the first corner
            ax.text(corners[0][0], corners[0][1], class_name, color='black', 
                    bbox=dict(facecolor='lime', alpha=0.3, pad=0.25, edgecolor='none'), 
                    fontsize=8)
        
        ax.axis('off')
        ax.set_title(f"Image {i+1} - Objects: {len(boxes)}")
        
    plt.tight_layout()
    
    # NEW: Save the file instead of trying to open a popup!
    plt.savefig('visualized_boxes.png', bbox_inches='tight', dpi=150)
    print("Saved visualization to 'visualized_boxes.png'")

def test_visualization():
    """
    Loads a single batch from the dataset and plots it.
    """
    dota_root = 'DOTA'  # Adjust to your dataset path
    img_size = 4000

    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    
    # Initialize the dataset using your existing class
    dataset = DOTADataset(dota_root, split="train", transform=transform, target_size=img_size)
    
    # Create a dataloader to grab a single batch of 2 images
    loader = DataLoader(dataset, batch_size=6, shuffle=True, collate_fn=collate_fn)

    # sample_image = image_loader("DOTA/test/images/P0006.png")
    print("Fetching batch...")
    images, boxes, labels = next(iter(loader))

    print(f"Visualizing {len(images)} images...")
    visualize_batch(images, boxes, labels)

if __name__ == "__main__":
    test_visualization()