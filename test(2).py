import matplotlib.pyplot as plt
import numpy as np

def visualize_feature_map(feature_map, title="Feature Map"):
    # Remove batch dim and take the mean across channels
    # feature_map shape: [1, 256, H, W] -> [H, W]
    am = torch.mean(feature_map.detach(), dim=1).squeeze(0).cpu().numpy()
    
    # Normalize to 0-1 for better display
    am = np.maximum(am, 0)
    am /= np.max(am) if np.max(am) > 0 else 1

    plt.imshow(am, cmap='magma')
    plt.title(title)
    plt.axis('off')
def show_all_levels(image_path):
    # 1. Load and prep image
    img_tensor = image_loader(image_path) # Uses your existing loader
    
    # 2. Forward pass
    c3, c4, c5 = backbone(img_tensor)
    p3, p4, p5, p6 = fpn(c3, c4, c5)
    
    # 3. Plotting
    levels = [p3, p4, p5, p6]
    names = ["P3 (Fine Details)", "P4", "P5", "P6 (Deep Semantics)"]
    
    plt.figure(figsize=(20, 5))
    for i, (lvl, name) in enumerate(zip(levels, names)):
        plt.subplot(1, 4, i + 1)
        visualize_feature_map(lvl, name)
    plt.show()

# Usage
show_all_levels("your_image.jpg")