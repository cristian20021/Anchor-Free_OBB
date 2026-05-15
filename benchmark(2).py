import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import time
import numpy as np
import argparse
from backbone import VGGBackbone, FPN, device
from head import OBBHead
from visualize import apply_nms_and_filter 

def run_benchmark(backbone, fpn, head, use_nms=False, img_size=1024, warmup_runs=50, test_runs=20):
    mode_name = "End-to-End (With NMS)" if use_nms else "Forward Pass Only (No NMS)"
    print(f"--- Starting Benchmark: {mode_name} ---")
    
    dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float).to(device)
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    with torch.no_grad():
        for _ in range(warmup_runs):
            c3, c4, c5 = backbone(dummy_input)
            p3, p4, p5, p6 = fpn(c3, c4, c5)
            _ = head([p3, p4, p5, p6])
    torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(test_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            
            # 1. Always do the Forward Pass
            c3, c4, c5 = backbone(dummy_input)
            p3, p4, p5, p6 = fpn(c3, c4, c5)
            out_cls, out_ctr, out_reg = head([p3, p4, p5, p6])
            
            # 2. Conditionally run NMS
            if use_nms:
                all_boxes, all_cls, all_ctr = [], [], []
                for i in range(4):
                    stride = head.STRIDES[i]
                    _, _, H, W = out_reg[i].shape
                    
                    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
                    grid_cx = (grid_x.float() + 0.5) * stride
                    grid_cy = (grid_y.float() + 0.5) * stride
                    
                    reg_flat = out_reg[i][0].permute(1, 2, 0).reshape(-1, 5)
                    cls_flat = out_cls[i][0].permute(1, 2, 0).reshape(-1, 15)
                    ctr_flat = out_ctr[i][0].permute(1, 2, 0).reshape(-1, 1)
                    
                    l, t, r, b_var, theta = reg_flat.unbind(dim=-1)
                    decoded_boxes = torch.stack([grid_cx.flatten() + (r - l) / 2.0, grid_cy.flatten() + (b_var - t) / 2.0, l + r, t + b_var, theta], dim=-1)
                    
                    all_boxes.append(decoded_boxes)
                    all_cls.append(cls_flat)
                    all_ctr.append(ctr_flat)
                    
                all_boxes = torch.cat(all_boxes, dim=0)
                all_cls = torch.cat(all_cls, dim=0)
                all_ctr = torch.cat(all_ctr, dim=0)
                
                _ = apply_nms_and_filter(all_boxes, all_cls, all_ctr, conf_thresh=0.01, iou_thresh=0.1)
            
            end_event.record()
            torch.cuda.synchronize() 
            timings.append(start_event.elapsed_time(end_event))

    # RESULTS
    avg_time = np.mean(timings)
    print(f"Results for {mode_name}:")
    print(f"Latency: {avg_time:.2f} ms | Throughput: {1000.0 / avg_time:.2f} FPS | VRAM: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MB\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the Anchor-Free Object Detector")
    parser.add_argument('--nms', action='store_true', help="Include NMS filtering in the benchmark")
    args = parser.parse_args()

    # Initialize models
    backbone = VGGBackbone().to(device).eval()
    fpn = FPN().to(device).eval()
    head = OBBHead().to(device).eval()

    run_benchmark(backbone, fpn, head, use_nms=args.nms, img_size=1024)