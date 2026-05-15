import torchvision.models as models
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights

imsize = 1024 if torch.cuda.is_available() else 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

loader = transforms.Compose([
    transforms.Resize(( imsize, imsize )),
    transforms.ToTensor()
])
unloader = transforms.ToPILImage()

def image_loader( image_path ):
    image = Image.open( image_path ).convert( "RGB" )
    image = loader( image ).unsqueeze( 0 )
    return image.to( device, torch.float )

class VGGBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=VGG16_Weights.DEFAULT)
        feats = list(vgg.features.children())
        for i in feats:
            print(f"{i}\n")   
        # TODO 1 — split feats into three stages using the layer table.
        # Each stage is an nn.Sequential of the relevant slices.

        self.stage3 = nn.Sequential(*feats[ : 17 ])
        self.stage4 = nn.Sequential(*feats[ 17 : 24 ])
        self.stage5 = nn.Sequential(*feats[ 24 : 31 ])

        # TODO 2 — freeze stage3 (early features, like ResNet stem).
        # Hint: iterate over self.stage3.parameters() and set requires_grad.
        for param in self.stage3.parameters():
            param.requires_grad = False


      

    def forward(self, x):
        # TODO 3 — forward through each stage sequentially.
        # c3 feeds into stage4, c4 feeds into stage5.
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return c3, c4, c5
    
class FPN(nn.Module):
    # VGG16 channel sizes at each extraction point
    C3_CH, C4_CH, C5_CH, OUT_CH = 256, 512, 512, 256

    def __init__(self):
        super().__init__()

        # TODO 4 — lateral 1×1 projections: each Ci → 256ch.
        # Hint: nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.lat3 = nn.Conv2d(256, 256, kernel_size=1)   # 256 → 256
        self.lat4 = nn.Conv2d(512, 256, kernel_size=1)   # 512 → 256
        self.lat5 = nn.Conv2d(512, 256, kernel_size=1)   # 512 → 256

        # TODO 5 — 3×3 smoothing convs after each top-down merge.
        # All are 256 → 256, padding=1 to preserve spatial size.
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # TODO 6 — P6 via strided conv on raw C5 (not on P5!).
        # Hint: kernel=3, stride=2, padding=1, C5_CH → OUT_CH
        self.p6_conv = nn.Conv2d(512,256, kernel_size = 3, stride=2, padding=1)

    def forward(self, c3, c4, c5):
        # lateral projections
        l3, l4, l5 = self.lat3(c3), self.lat4(c4), self.lat5(c5)

        # TODO 7 — top-down merge path.
        # Order: build P5 first, then upsample to build P4, then P3.
        # Use F.interpolate with size=l_i.shape[-2:] and mode='nearest'
        # to avoid off-by-one errors at odd spatial sizes.
        p5 = self.smooth5(l5)
        p4 = self.smooth4(l4 + F.interpolate(p5, size=l4.shape[-2:], mode='nearest'))
        p3 = self.smooth3(l3 + F.interpolate(p4, size=l3.shape[-2:], mode='nearest'))

        # TODO 8 — P6 from raw C5.
        p6 = self.p6_conv(c5)

        return p3, p4, p5, p6  # shapes: 128², 64², 32², 16²
    

def test_fpn_shapes():
    backbone = VGGBackbone().to(device)
    fpn      = FPN().to(device)
    x        = torch.zeros(1, 3, 1024, 1024).to(device)

    c3, c4, c5   = backbone(x)
    p3, p4, p5, p6 = fpn(c3, c4, c5)

    # TODO 9 — fill in expected shapes and run this.
    assert p3.shape == (1, 256, 128, 128)
    assert p4.shape == (1, 256, 64, 64)
    assert p5.shape == (1, 256, 32, 32)
    assert p6.shape == (1, 256, 16, 16)
    
if __name__ == "__main__":
    # 1. Initialize models and move them to the device (MPS/CPU)
    # Note: Added .to(device) here!
    backbone = VGGBackbone().to(device).eval()
    fpn = FPN().to(device).eval()

    import matplotlib.pyplot as plt
    import numpy as np

    def visualize_feature_map(feature_map, title="Feature Map"):
        # Mean across channels and move to CPU for numpy/matplotlib
        am = torch.mean(feature_map.detach(), dim=1).squeeze(0).cpu().numpy()
        
        am = np.maximum(am, 0)
        if np.max(am) > 0:
            am /= np.max(am)

        plt.imshow(am, cmap='magma')
        plt.title(title)
        plt.axis('off')

    def show_all_levels(image_path):
        # 1. Load and prep image (loader already sends to device)
        img_tensor = image_loader(image_path) 
        
        # 2. Forward pass (Wrap in no_grad to save memory)
        with torch.no_grad():
            c3, c4, c5 = backbone(img_tensor)
            # Use the 'fpn' INSTANCE, not the 'FPN' CLASS
            p3, p4, p5, p6 = fpn(c3, c4, c5)
        
        # 3. Plotting
        levels = [p3, p4, p5, p6]
        names = ["P3 (Fine Details)", "P4", "P5", "P6 (Deep Semantics)"]
        
        plt.figure(figsize=(20, 5))
        for i, (lvl, name) in enumerate(zip(levels, names)):
            plt.subplot(1, 4, i + 1)
            visualize_feature_map(lvl, name)
        plt.tight_layout()
        plt.show()

    # 4. Run it
    show_all_levels("original.jpg")