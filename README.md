# Anchor-Free Oriented Bounding Box Detection

An anchor-free object detector for aerial images that predicts oriented bounding boxes with rotation angles on the DOTA dataset.

## Architecture

- **Backbone**: VGG16 (pretrained, early layers frozen)
- **Neck**: Feature Pyramid Network (P3-P6)
- **Head**: Anchor-free detection head with 4 FPN levels
- **Output**: Classification (15 classes), Centerness, Regression (l,t,r,b,θ)

## Dataset

15 DOTA classes: plane, baseball-diamond, bridge, ground-track-field, small-vehicle, large-vehicle, ship, tennis-court, basketball-court, storage-tank, soccer-ball-field, roundabout, harbor, swimming-pool, helicopter

## Installation

1. **Clone/Setup**
   ```bash
   git clone https://github.com/cristian20021/Anchor-Free_OBB.git
   cd Anchor-Free_OBB-main
   ```

2. **Install Dependencies**
   ```bash
   pip install torch torchvision numpy pillow matplotlib tqdm
   ```

3. **Prepare Dataset**
   
   Download DOTA dataset from: https://captain-whu.github.io/DOTA/dataset.html
   
   Then organize it:
   ```bash
   # Download and extract DOTA dataset
   # Expected structure:
   # DOTA/
   # ├── train/images/ + labelTxt/
   # ├── validation/images/ + labelTxt/labelTxt/
   # └── test/images/
   
   ```

## Quick Start

### Train Model
```bash
python pipeline.py
```
- Trains for 60 epochs
- Saves checkpoints every 15 epochs to `./checkpoints/`
- Logs training progress to `training_log.csv`
- Takes ~1-2 hours on GPU (depends on hardware)

### Run Inference
```bash
python inference.py
```
- Loads trained weights from `checkpoints/dota_weights_epoch_60.pth`
- Processes validation images
- Generates visualizations with detected boxes
- Output images saved with predictions

### Evaluate Performance
```bash
python evaluate_map.py
```
- Computes mAP (mean Average Precision)
- Per-class evaluation metrics
- Comparison with ground truth

### Benchmark Speed
```bash
python benchmark.py
```
- Measures inference speed
- Reports memory usage
- Performance profiling

### Run Tests
```bash
python test.py
```
- Unit tests for model components
- Validates data loading
- Loss computation checks

## Training Details

- **Epochs**: 60
- **Optimizer**: Adam (lr=1e-4)
- **Loss**: GWD Loss (oriented bboxes) + Focal Loss (classification)
- **Batch Size**: 4 (GPU) | 1 (CPU)
- **Image Size**: 1024×1024
- **Final Loss**: Train 0.6080 | Val 0.7427
- **Checkpoints**: Saved at epochs 15, 30, 45, 60

## Project Structure

```
.
├── pipeline.py           # Training script (main)
├── backbone.py           # VGG16 backbone + FPN
├── head.py              # Detection head
├── loss.py              # GWD + Focal loss functions
├── inference.py         # Inference pipeline
├── visualize.py         # Visualization & NMS
├── evaluate_map.py      # mAP evaluation
├── benchmark.py         # Performance profiling
├── test.py              # Unit tests
├── preprocess.py        # Data preprocessing
├── download_dota.sh     # Dataset download
├── training_log.csv     # Training history
├── checkpoints/         # Model weights directory
└── README.md            # This file
```

## Configuration

Edit these values in `pipeline.py` to customize:
```python
num_epochs = 60          # Number of training epochs
lr = 1e-4               # Learning rate
img_size = 1024         # Image size (1024 or 256)
batch_size = 4          # Batch size
num_workers = 4         # Data loading workers
save_interval = 15      # Checkpoint save interval
```

## Output

After training:
- Model checkpoints in `./checkpoints/dota_weights_epoch_*.pth`
- Training logs in `training_log.csv` (epoch, train loss, val loss)
- Inference visualizations with oriented bounding boxes

## Requirements

```
torch>=1.9
torchvision>=0.10
numpy
pillow
matplotlib
tqdm
```

Optional (for better performance):
```
opencv-python  # For advanced image processing
```

## Hardware Requirements

- **GPU Recommended**: NVIDIA GPU with CUDA support
- **Minimum RAM**: 8GB (16GB+ recommended)
- **VRAM**: 4GB+ (tested on RTX 3090)
- **CPU-only**: Supported but slow (~10-50x slower)

## Key Features

- No anchor boxes (anchor-free approach)
- Oriented bounding boxes with rotation angles
- Multi-scale detection (strides: 8, 16, 32, 64)
- Real-time inference on GPU
- Proper handling of rotated objects

## Troubleshooting

**CUDA Out of Memory**: Reduce `batch_size` or `img_size` in `pipeline.py`

**Missing Data**: Ensure DOTA dataset is in correct structure (see Installation)

**Slow Training**: Use GPU, increase `num_workers`, or reduce `img_size` for testing




