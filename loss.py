
import torch
import math
def obb_to_gaussian(boxes):
    """
    boxes: (N, 5) tensor of (cx, cy, w, h, θ) in radians
    """
    cx, cy, w, h, theta = boxes.unbind(dim=-1)


    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    # TODO — build R as a (N,2,2) tensor.
 
    R = torch.stack([
        torch.stack([cos_t, -sin_t], dim=-1),
        torch.stack([sin_t, cos_t], dim=-1),
    ], dim=-2)                        

    # TODO — build D as (N,2,2).
    D = torch.diag_embed(torch.stack([w**2 / 4, h**2 / 4], dim=-1))


    # TODO — use torch.bmm for batched matrix multiply.
    R_T = R.transpose(-1,-2)
    sigma = torch.bmm(torch.bmm(R,D), R_T)

    mu = torch.stack([cx, cy], dim=-1)  # (N, 2)
    return mu, sigma

def gwd_loss(pred_boxes, gt_boxes, tau=1.0, eps=1e-6): # FIX: Change tau to 1.0
    mu1, sigma1 = obb_to_gaussian(pred_boxes)
    mu2, sigma2 = obb_to_gaussian(gt_boxes)

    mean_dist = torch.sum((mu1 - mu2)**2, dim=-1)

    sigma1 = sigma1 + eps * torch.eye(2, device=sigma1.device)
    sigma2 = sigma2 + eps * torch.eye(2, device=sigma2.device)

    vals, vecs = torch.linalg.eigh(sigma1)
    vals = vals.clamp(min=0).sqrt()
    sigma1_sqrt = torch.bmm(torch.bmm(vecs, torch.diag_embed(vals)), vecs.transpose(-1, -2))

    M = torch.bmm(torch.bmm(sigma1_sqrt, sigma2), sigma1_sqrt)
    m_vals = torch.linalg.eigvalsh(M).clamp(min=0).sqrt()
    trace_sqrt_M = m_vals.sum(dim=-1)
    
    tr1 = sigma1[:, 0, 0] + sigma1[:, 1, 1]
    tr2 = sigma2[:, 0, 0] + sigma2[:, 1, 1]
    bures = tr1 + tr2 - 2 * trace_sqrt_M  

    d_w = torch.sqrt((mean_dist + bures).clamp(min=1e-6))
    
    # FIX: Apply log transform to prevent vanishing gradients
    f_d = torch.log1p(d_w) 
    loss = 1 - 1 / (tau + f_d)
    
    return loss.mean()

def clip_polygon_by_edge(poly, edge_start, edge_end):
    """
    Clip a convex polygon against a single infinite directed line.
    poly: list of (x,y) tuples
    Returns: clipped polygon as list of (x,y)
    """
    def inside(p):
        return (edge_end[0] - edge_start[0]) * (p[1] - edge_start[1]) - \
                    (edge_end[1] - edge_start[1]) * (p[0] - edge_start[0]) >= 0
    def intersect(p1, p2):
      
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        ex, ey = edge_end[0] - edge_start[0], edge_end[1] - edge_start[1]
        
        denom = (dy * ex - dx * ey)
        if abs(denom) < 1e-9:
            return p1 # Parallel
        
        t = (ex * (edge_start[1] - p1[1]) - ey * (edge_start[0] - p1[0])) / denom
        return (p1[0] + t * dx, p1[1] + t * dy)
        

    output = []
    if not poly:
        return output

    for i in range(len(poly)):
        current = poly[i]
        previous = poly[i - 1]
        if inside(current):
            if not inside(previous):
                output.append(intersect(previous, current))
            output.append(current)
        elif inside(previous):
            output.append(intersect(previous, current))
    return output


def rotated_iou(box1, box2):
    """
    box1, box2: (5,) tensors (cx, cy, w, h, θ)
    """
    def box_to_corners(b):
        # TODO — convert (cx,cy,w,h,θ) to 4 corners as list of (x,y).
        cx, cy, w, h, theta = b.tolist()
        corners = [(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]
  
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        rotated = []
        for x, y in corners:
          
            rx = x * cos_t - y * sin_t
            ry = x * sin_t + y * cos_t
            # translate back to center (cx, cy)
            rotated.append((cx + rx, cy + ry))
        return rotated
    poly1 = box_to_corners(box1)
    poly2 = box_to_corners(box2)

    # Clip poly1 against each of the 4 edges of poly2
    clipped = poly1
    for i in range(len(poly2)):
        clipped = clip_polygon_by_edge(clipped, poly2[i], poly2[(i+1) % len(poly2)])
        if not clipped:
            return 0.0

    def shoelace(poly):
        # TODO — compute polygon area via shoelace formula.
        n = len(poly)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += poly[i][0]* poly[j][1]
            area -= poly[j][0]* poly[i][1]
        return abs(area) / 2.0

    inter = shoelace(clipped)
    area1 = (box1[2] * box1[3]).item()
    area2 = (box2[2] * box2[3]).item()
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0