#!/bin/bash
cd ~/Test
export PATH="$HOME/.local/bin:$PATH"
pip install --user gdown
mkdir -p DOTA/train/images DOTA/train/labelTxt
mkdir -p DOTA/validation/images DOTA/validation/labelTxt/labelTxt
mkdir -p DOTA/test/images
echo "=== Downloading DOTA v1.0 ==="
echo "Downloading train images..."
gdown "https://drive.google.com/uc?id=1PFbserE_JnMGSHg1dBIOf5_fEaRMBNaS" -O DOTA/train/train_images_part1.zip
gdown "https://drive.google.com/uc?id=1cf4sk_GB4JVnDE4gHVCiGEjq9ENxCrkI" -O DOTA/train/train_images_part2.zip
echo "Downloading train labels..."
gdown "https://drive.google.com/uc?id=1uVoE-fMJn5j5mhYkGaqFMnCIjLMGbRGl" -O DOTA/train/train_labels.zip
echo "Downloading validation images..."
gdown "https://drive.google.com/uc?id=1mtBd4PCYAX4d4t7fGeRDiUI7r0F2PMbM" -O DOTA/validation/val_images.zip
echo "Downloading validation labels..."
gdown "https://drive.google.com/uc?id=1I_e2XSuB9sCOIWmCBFK8JzFgWwQk6MUb" -O DOTA/validation/val_labels.zip
echo "Downloading test images..."
gdown "https://drive.google.com/uc?id=1KKEArY1HX6_dKH_KBpIMKDBb3cOzwBdE" -O DOTA/test/test_images_part1.zip
gdown "https://drive.google.com/uc?id=1EEcBR8jwGAR8OLpfWgLU38eGy0oIzfvA" -O DOTA/test/test_images_part2.zip
echo "=== Extracting ==="
for z in DOTA/train/*.zip; do [ -f "$z" ] && unzip -o "$z" -d DOTA/train/ && rm "$z"; done
for z in DOTA/validation/*.zip; do [ -f "$z" ] && unzip -o "$z" -d DOTA/validation/ && rm "$z"; done
for z in DOTA/test/*.zip; do [ -f "$z" ] && unzip -o "$z" -d DOTA/test/ && rm "$z"; done
echo "=== Done ==="
echo "Train images: $(ls DOTA/train/images/ 2>/dev/null | wc -l)"
echo "Train labels: $(ls DOTA/train/labelTxt/ 2>/dev/null | wc -l)"
echo "Val images:   $(ls DOTA/validation/images/ 2>/dev/null | wc -l)"
echo "Test images:  $(ls DOTA/test/images/ 2>/dev/null | wc -l)"
