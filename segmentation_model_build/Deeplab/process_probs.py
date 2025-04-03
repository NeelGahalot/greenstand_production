import warnings
import sys
from torchvision.ops import box_convert
from torchvision.io import read_image, write_jpeg
from torchvision.ops import masks_to_boxes
import os
from dotenv import load_dotenv
import time
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import boto3
import pandas as pd
import argparse
from PIL import Image
import urllib.request
import io
from torchvision import transforms
from io import BytesIO
import time
import torch
import torchvision
from scipy.special import expit

current_home =os.getcwd()
# import Deeplab dependencies
os.chdir('/home/neel/mnt_data/greenstand/old_deeplab/segmentation_model_build/Deeplab')
print(os.getcwd())
sys.path.append(os.getcwd())
from inference.infer import get_s3_bucket, flip, is_s3_object_key, is_url, load_deeplab_model, pil_to_grayscale_tensor
from post_processing.control_random_field import crf_with_prob
import utils
device = "cuda"
ckpt_path = '/home/neel/mnt_data/greenstand/old_deeplab/segmentation_model_build/Deeplab/saved_models/best_deeplabv3plus_mobilenet_custom_os16_0.7854892764326529.pth'


model = load_deeplab_model(ckpt_path, device).eval()

# Import SAM Dependecies
os.chdir('/home/neel/mnt_data/greenstand/segment-anything')
sys.path.append("/home/neel/mnt_data/greenstand/segment-anything")
print(os.getcwd())
from segment_anything import sam_model_registry, SamPredictor
sam_checkpoint = "/home/neel/mnt_data/greenstand/segment-anything/sam_vit_l_0b3195.pth"
model_type = "vit_l"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

os.chdir(f'{current_home}/crf_sam_annotations_large')

input_dir = 'samples'
prob_dir = 'probs'
mask_dir = 'binary_masks'
os.makedirs(prob_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)

img_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === Loop over all images ===
for filename in tqdm(os.listdir(input_dir)):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_dir, filename)
    image = Image.open(img_path).convert("RGB")

    # === Prepare image for DeepLab ===
    img_tensor = img_transform(image).unsqueeze(0).to(device, dtype=torch.float32)

    with torch.no_grad():
        output = model(img_tensor)
        output = torch.squeeze(output, dim=1)
        prob = torch.sigmoid(output).detach()
        pred = (prob > 0.5).long().cpu().numpy()[0]

    # === Save prob tensor ===
    prob_np = prob[0].cpu().numpy()
    #prob_save_path = os.path.join(prob_dir, filename.replace(".jpg", ".npy").replace(".png", ".npy"))
    #np.save(prob_save_path, prob_np)

    # === Denormalize image for visualization/SAM ===
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_np = img_tensor[0].detach().cpu().numpy()
    img_np = (denorm(img_np) * 255).transpose(1, 2, 0).astype(np.uint8)

    # === Compute confidence score ===
    count_pixel = np.sum(prob_np > 0.5)
    confidence = np.sum(prob_np[prob_np > 0.5]) / count_pixel if count_pixel != 0 else 0

    # === Clean with CRF ===
    cleaned_mask = crf_with_prob(img_np, (pred * 255).astype(np.uint8), prob_np)

    # === Save binary mask ===
    #binary_mask_path = os.path.join(mask_dir, filename.replace(".jpg", ".png").replace(".jpeg", ".png"))
    #cv2.imwrite(binary_mask_path, (cleaned_mask * 255).astype(np.uint8))

    # === Prompt SAM if mask is valid ===
    if np.sum(cleaned_mask == 1) == 0:
        print(f"Skipping {filename}: mask is empty")
        continue

    # === Get bounding box from mask ===
    cleaned_mask_img = Image.fromarray((cleaned_mask * 255).astype(np.uint8))
    boxes = masks_to_boxes(pil_to_grayscale_tensor(cleaned_mask_img))
    box = np.array(boxes.tolist()[0])

    # === Predict with SAM ===
    predictor.set_image(img_np)
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=box[None, :],
        multimask_output=False,
        return_logits = True
    )
    sam_logits_np = expit(masks[0])
    
    #binary_array = np.where(masks[0], 255, 0).astype(np.uint8)
    binary_array = (masks[0].astype(np.uint8)) * 255

    binary_image = Image.fromarray(binary_array, 'L')
    basename = os.path.splitext(os.path.basename(filename))[0]
    mask_name = f"{basename}_binarymask.jpg"
    binary_image.save(f'{mask_dir}/{mask_name}')
    prob_name = f"{basename}_prob.npy"
    np.save(f'{prob_dir}/{prob_name}',sam_logits_np)
    # mnt_data/greenstand/old_deeplab/segmentation_model_build/Deeplab/crf_sam_annotations_large/samples/eastafrica_acactort_2020.06.02.20.07.57_ae1bffbc-d431-4723-bdaa-836959e029c2_img_20100103_110400_130004792.jpg