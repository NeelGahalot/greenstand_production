import os
import numpy as np
from PIL import Image
from tqdm import tqdm  # add tqdm

# Change working directory
os.chdir('/home/neel/mnt_data/greenstand/old_deeplab/segmentation_model_build/Deeplab/crf_sam_annotations_large')

# Directories 
prob_dir = 'probs'
mask_dir = 'binary_masks'
os.makedirs(mask_dir, exist_ok=True)

# List all relevant prob files
prob_files = [f for f in os.listdir(prob_dir) if f.endswith('_prob.npy')]

# Loop with tqdm
for fname in tqdm(prob_files, desc="Converting probs to masks"):
    # Extract base filename
    basename = fname.replace('_prob.npy', '')

    # Load probability map
    prob_path = os.path.join(prob_dir, fname)
    prob_map = np.load(prob_path)

    # Threshold to binary mask
    binary_mask = (prob_map > 0.5).astype(np.uint8) * 255

    # Save binary mask as image
    mask_img = Image.fromarray(binary_mask, mode='L')
    mask_path = os.path.join(mask_dir, f'{basename}_binarymask.jpg')
    mask_img.save(mask_path)
