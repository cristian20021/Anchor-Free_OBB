"""
Usage
-----
# Basic (random/untrained weights, validation split):
    python evaluate_map.py

# With a saved checkpoint:
    python evaluate_map.py --weights checkpoints/dota_weights_epoch_30.pth

# Change IoU threshold or split:
    python evaluate_map.py --weights checkpoints/dota_weights_epoch_30.pth --iou 0.75 --split validation

# COCO-style mAP (averages IoU 0.50 → 0.95 in 0.05 steps):
    python evaluate_map.py --weights checkpoints/dota_weights_epoch_30.pth --coco

Options
-------
--weights PATH   Path to the unified .pth checkpoint file (default: random weights)
--split   STR    dataset split to evaluate on  (default: validation)
--iou     FLOAT  IoU threshold for a True Positive  (default: 0.5)
--coco          Compute COCO-style mAP@[.50:.05:.95]
--conf    FLOAT  Confidence threshold before NMS  (default: 0.05)
--nms-iou FLOAT  IoU threshold inside NMS  (default: 0.1)
--batch   INT    Batch size  (default: 1)
--img-size INT   Resize target  (default: 1024)
--workers INT    DataLoader workers  (default: 4)
--save    PATH   Save per-class results to a .csv file
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
import argparse
import csv
import time
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from pipeline  import DOTADataset, DOTA_CLASSES, collate_fn
from backbone  import VGGBackbone, FPN, device
from head      import OBBHead
from loss      import rotated_iou
from visualize import apply_nms_and_filter

def decode_predictions(out_cls, out_ctr, out_reg, head_strides, target_device):
    """
    Decode raw head outputs from all four FPN levels into flat tensors.

    Returns
    -------
    pred_boxes  : (N, 5)  – cx, cy, w, h, theta
    pred_cls    : (N, 15) – raw class logits
    pred_ctr    : (N, 1)  – raw centerness logits
    """
    all_boxes, all_cls, all_ctr = [], [], []

    for i in range(4):
        reg_lvl = out_reg[i][0]          # (5, H, W)
        cls_lvl = out_cls[i][0]          # (15, H, W)
        ctr_lvl = out_ctr[i][0]          # (1,  H, W)
        _, H, W = reg_lvl.shape
        stride  = head_strides[i]

        # Build the anchor-free grid centres for this level
        gy, gx = torch.meshgrid(
            torch.arange(H, device=target_device),
            torch.arange(W, device=target_device),
            indexing='ij',
        )
        grid_cx = (gx.float() + 0.5) * stride   # (H, W)
        grid_cy = (gy.float() + 0.5) * stride

        # Unpack l, t, r, b, theta and convert to cx/cy/w/h
        l, t, r, b, theta = reg_lvl.unbind(dim=0)   # each (H, W)
        cx    = grid_cx + (r - l) / 2.0
        cy    = grid_cy + (b - t) / 2.0
        w     = (l + r).reshape(-1)
        h     = (t + b).reshape(-1)
        cx    = cx.reshape(-1)
        cy    = cy.reshape(-1)
        theta = theta.reshape(-1)

        boxes = torch.stack([cx, cy, w, h, theta], dim=-1)

        all_boxes.append(boxes)
        all_cls  .append(cls_lvl.permute(1, 2, 0).reshape(-1, len(DOTA_CLASSES)))
        all_ctr  .append(ctr_lvl.permute(1, 2, 0).reshape(-1, 1))

    return (
        torch.cat(all_boxes, dim=0),
        torch.cat(all_cls,   dim=0),
        torch.cat(all_ctr,   dim=0),
    )

def match_predictions_to_gts(pred_boxes, pred_labels, pred_scores,
                              gt_boxes,   gt_labels,
                              iou_threshold):
    """
    For a single image, decide which predictions are TPs and which are FPs.

    Returns a list of (score, is_tp, class_id) tuples — one per prediction —
    plus the per-class GT counts for this image.
    """
    results   = []                     # (score, is_tp, class_id)
    n_gt_cls  = defaultdict(int)       # class_id → # GTs in this image

    for lbl in gt_labels:
        n_gt_cls[lbl.item()] += 1

    if len(pred_boxes) == 0:
        return results, n_gt_cls

    # Track which GT boxes have already been claimed by a higher-scored pred
    gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

    # Walk predictions in descending score order
    order = pred_scores.argsort(descending=True)
    for idx in order:
        p_box   = pred_boxes  [idx]
        p_label = pred_labels [idx].item()
        p_score = pred_scores [idx].item()

        # Only compare against GT boxes of the same class
        gt_mask = (gt_labels == p_label).nonzero(as_tuple=True)[0]

        best_iou  = 0.0
        best_gi   = -1
        for gi in gt_mask:
            iou = rotated_iou(p_box, gt_boxes[gi])
            if iou > best_iou:
                best_iou, best_gi = iou, gi.item()

        if best_iou >= iou_threshold and best_gi >= 0 and not gt_matched[best_gi]:
            gt_matched[best_gi] = True
            results.append((p_score, 1, p_label))   # True Positive
        else:
            results.append((p_score, 0, p_label))   # False Positive

    return results, n_gt_cls


def compute_ap_from_records(records, n_gt):
    """
    records : list of (score, is_tp) sorted by score descending
    n_gt    : total number of GT instances for this class
    Returns  : AP (float), recall at last point, precision at last point
    """
    if n_gt == 0 or len(records) == 0:
        return 0.0, 0.0, 0.0

    records = sorted(records, key=lambda x: -x[0])
    tp_cum  = torch.cumsum(torch.tensor([r[1] for r in records], dtype=torch.float32), dim=0)
    fp_cum  = torch.cumsum(torch.tensor([1 - r[1] for r in records], dtype=torch.float32), dim=0)

    recalls    = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

    # 11-point interpolation (PASCAL VOC)
    ap = 0.0
    for thr in torch.linspace(0.0, 1.0, 11):
        mask = recalls >= thr
        p    = precisions[mask].max().item() if mask.any() else 0.0
        ap  += p / 11.0

    return ap, recalls[-1].item(), precisions[-1].item()


@torch.no_grad()
def evaluate(backbone, fpn, head,
             dataloader,
             iou_thresholds,
             conf_thresh=0.05,
             nms_iou=0.1):
    """
    Run inference over the full dataloader and return per-class AP dicts.

    Returns
    -------
    results : dict  iou_thr → dict  class_id → AP
    """
    backbone.eval(); fpn.eval(); head.eval()

    # Accumulate across the whole dataset
    # class_records[iou_thr][class_id] = [(score, is_tp), ...]
    class_records = {t: defaultdict(list) for t in iou_thresholds}
    class_n_gt    = defaultdict(int)    # total GT per class, threshold-independent

    n_images  = 0
    t_infer   = 0.0

    for images, gt_boxes_list, gt_labels_list in tqdm(dataloader, desc="Evaluating"):
        batch_size = images.shape[0]

        for b in range(batch_size):
            img   = images[b].unsqueeze(0).to(device)
            gt_b  = gt_boxes_list [b].to(device)
            lbl_b = gt_labels_list[b].to(device)
            t0 = time.perf_counter()
            c3, c4, c5 = backbone(img)
            p3, p4, p5, p6 = fpn(c3, c4, c5)
            out_cls, out_ctr, out_reg = head([p3, p4, p5, p6])
            t_infer += time.perf_counter() - t0

            pred_boxes, pred_cls, pred_ctr = decode_predictions(
                out_cls, out_ctr, out_reg, head.STRIDES, device
            )

            final_boxes, final_labels, final_scores = apply_nms_and_filter(
                pred_boxes, pred_cls, pred_ctr,
                conf_thresh=conf_thresh,
                iou_thresh=nms_iou,
            )

            for lbl in lbl_b:
                class_n_gt[lbl.item()] += 1

            for iou_thr in iou_thresholds:
                det_results, _ = match_predictions_to_gts(
                    final_boxes, final_labels, final_scores,
                    gt_b, lbl_b,
                    iou_threshold=iou_thr,
                )
                for score, is_tp, cls_id in det_results:
                    class_records[iou_thr][cls_id].append((score, is_tp))

            n_images += 1

    results = {}
    for iou_thr in iou_thresholds:
        aps = {}
        for cls_id in range(len(DOTA_CLASSES)):
            n_gt = class_n_gt[cls_id]
            ap, _, _ = compute_ap_from_records(
                class_records[iou_thr][cls_id], n_gt
            )
            aps[cls_id] = ap
        results[iou_thr] = aps

    print(f"\nEvaluated {n_images} images  |  avg inference {1000*t_infer/max(n_images,1):.1f} ms/img")
    return results, class_n_gt


def print_results_table(results, class_n_gt, iou_thresholds, save_path=None):
    """Print a formatted per-class AP table and overall mAP."""

    thr_labels = [f"AP@{t:.2f}" for t in iou_thresholds]
    col_w      = max(len(t) for t in thr_labels) + 2
    name_w     = max(len(c) for c in DOTA_CLASSES) + 2

    sep    = "─" * (name_w + 8 + col_w * len(iou_thresholds))
    header = f"{'Class':<{name_w}} {'# GT':>6}  " + "  ".join(f"{t:>{col_w}}" for t in thr_labels)
    print(f"\n{sep}")
    print(header)
    print(sep)

    rows = []    # for CSV export
    per_thr_aps = defaultdict(list)

    for cls_id, cls_name in enumerate(DOTA_CLASSES):
        n_gt  = class_n_gt[cls_id]
        aps   = [results[t][cls_id] for t in iou_thresholds]
        row   = f"{cls_name:<{name_w}} {n_gt:>6}  " + "  ".join(f"{ap*100:>{col_w}.1f}" for ap in aps)
        print(row)
        rows.append([cls_name, n_gt] + [f"{ap*100:.2f}" for ap in aps])
        for t, ap in zip(iou_thresholds, aps):
            per_thr_aps[t].append(ap)

    print(sep)

    map_row = f"{'mAP':<{name_w}} {'':>6}  " + "  ".join(
        f"{np.mean(per_thr_aps[t])*100:>{col_w}.2f}" for t in iou_thresholds
    )
    print(map_row)
    print(sep)

    # Highlight the primary metric
    primary_thr = 0.5
    if primary_thr in results:
        print(f"\n  ★  mAP@0.50 = {np.mean(per_thr_aps[primary_thr])*100:.2f}%")

    if len(iou_thresholds) > 1:
        coco_map = np.mean([np.mean(per_thr_aps[t]) for t in iou_thresholds]) * 100
        print(f"mAP@[.50:.05:.95] = {coco_map:.2f}%")

    if save_path:
        with open(save_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "n_gt"] + thr_labels)
            w.writerows(rows)
            w.writerow(["mAP", ""] + [f"{np.mean(per_thr_aps[t])*100:.2f}" for t in iou_thresholds])
        print(f"\n  Results saved to: {save_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate DOTA OBB detector — mAP")
    p.add_argument("--weights",  default=None,         help="Path to the unified .pth checkpoint file")
    p.add_argument("--split",    default="validation", choices=["train", "validation"])
    p.add_argument("--iou",      default=0.5,  type=float, help="IoU threshold for TP")
    p.add_argument("--coco",     action="store_true",  help="Also compute mAP@[.50:.05:.95]")
    p.add_argument("--conf",     default=0.05, type=float, help="Confidence threshold")
    p.add_argument("--nms-iou",  default=0.1,  type=float, help="IoU threshold inside NMS")
    p.add_argument("--batch",    default=1,    type=int)
    p.add_argument("--img-size", default=1024, type=int)
    p.add_argument("--workers",  default=4,    type=int)
    p.add_argument("--save",     default=None,         help="Save results CSV to this path")
    p.add_argument("--dota-root", default="DOTA",      help="Path to DOTA dataset root")
    return p.parse_args()


def main():
    args = parse_args()

    if args.coco:
        iou_thresholds = [round(t, 2) for t in np.arange(0.50, 1.00, 0.05).tolist()]
    else:
        iou_thresholds = [args.iou]

    print("=" * 60)
    print("  DOTA OBB Detector — mAP Evaluation")
    print("=" * 60)
    print(f"  Split       : {args.split}")
    print(f"  IoU thr(s)  : {iou_thresholds}")
    print(f"  Conf thr    : {args.conf}")
    print(f"  NMS IoU     : {args.nms_iou}")
    print(f"  Device      : {device}")
    print("=" * 60)

    print("\nLoading models...")
    backbone = VGGBackbone().to(device)
    fpn      = FPN().to(device)
    head     = OBBHead().to(device)

    if args.weights:
        checkpoint = torch.load(args.weights, map_location=device)
        backbone.load_state_dict(checkpoint['backbone'])
        fpn.load_state_dict(checkpoint['fpn'])
        head.load_state_dict(checkpoint['head'])
        print(f"  Models loaded successfully from ← {args.weights}")
    else:
        print("No checkpoint provided. Evaluating with random weights.")
        print("(Expected mAP =~ 0%; use --weights to load trained weights.)")

    transform = transforms.Compose([transforms.ToTensor()])
    dataset   = DOTADataset(
        args.dota_root, split=args.split,
        transform=transform, target_size=args.img_size,
    )
    loader = DataLoader(
        dataset,
        batch_size  = args.batch,
        shuffle     = False,
        num_workers = args.workers,
        collate_fn  = collate_fn,
        pin_memory  = device.type == "cuda",
    )
    print(f"\nDataset: {len(dataset)} images  ({args.split})")

    results, class_n_gt = evaluate(
        backbone, fpn, head,
        loader,
        iou_thresholds,
        conf_thresh = args.conf,
        nms_iou     = args.nms_iou,
    )

    print_results_table(results, class_n_gt, iou_thresholds, save_path=args.save)


if __name__ == "__main__":
    main()