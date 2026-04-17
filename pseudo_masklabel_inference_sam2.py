# -----------------------------------------------------------------------------
# Script Name: sam2_batch_generate.py
# 
# Description:
# This script performs batch segmentation inference using Segment Anything 2 (SAM2).
# It takes object detection bounding boxes (COCO format) as input and generates 
# high-quality segmentation masks (polygons).
#
# Key Features:
# 1. Box-prompted segmentation using SAM2.
# 2. Area filtering to remove small discrete noise/artifacts.
# 3. Polygon simplification (RDP algorithm) for efficient storage and smoothing.
# -----------------------------------------------------------------------------

import json
import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

# --- SAM2 Library Import ---
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Error: segment-anything-2 not found. Please install it from the official repository.")
    exit()

# ==================== Configuration Area ====================
# Input/Output Paths (Replace with your own directory structure)
INPUT_JSON = "path/to/your/input_detection_labels.json"
OUTPUT_JSON = "path/to/your/output_segmentation_pseudolabel.json"
IMAGE_ROOT = "path/to/your/image"

# Model Checkpoints and Configs
SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ============================================================

def mask_to_polygons(mask):
    """
    Converts a binary mask to simplified polygons.
    
    Args:
        mask (np.ndarray): Binary mask from SAM2.
        
    Returns:
        list: A list of flattened polygon coordinates [x1, y1, x2, y2, ...].
    """
    # 1. Extract contours from the binary mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    polygons = []
    
    # 2. Sort contours by area to identify the primary object
    contours_with_area = []
    for contour in contours:
        area = cv2.contourArea(contour)
        contours_with_area.append((area, contour))
    
    contours_with_area.sort(key=lambda x: x[0], reverse=True)
    
    if not contours_with_area:
        return []
    
    max_area = contours_with_area[0][0]  # Reference area for noise filtering

    for area, contour in contours_with_area:
        # --- A. Area Filtering ---
        # Discard fragments smaller than 1% of the largest contour to remove noise
        if area < max_area * 0.01:
            continue
        
        # --- B. Polygon Simplification ---
        # Uses the Douglas-Peucker algorithm to reduce the number of points
        perimeter = cv2.arcLength(contour, True)
        # epsilon=0.001 is a balance between precision and data size
        epsilon = 0.001 * perimeter 
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Valid polygons must have at least 3 points (6 coordinates)
        if approx_contour.size >= 6:
            poly = approx_contour.flatten().tolist()
            polygons.append(poly)
            
    return polygons

def main():
    # --- 1. Model Initialization ---
    print(f"Initializing SAM2 on {DEVICE}...")
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    predictor = SAM2ImagePredictor(sam2_model)

    # --- 2. Data Loading ---
    print(f"Loading input JSON: {INPUT_JSON}...")
    with open(INPUT_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Map image IDs to their info for fast lookup
    img_map = {img['id']: img for img in data['images']}
    
    # Group annotations by image_id to optimize SAM2 'set_image' calls
    img_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        img_to_anns.setdefault(img_id, []).append(ann)

    print(f"Start processing {len(img_to_anns)} images...")
    
    processed_count = 0
    new_annotations = []
    
    # --- 3. Batch Inference Loop ---
    for img_id, anns in tqdm(img_to_anns.items(), desc="Generating Masks"):
        img_info = img_map.get(img_id)
        if not img_info:
            continue

        file_name = img_info['file_name']
        full_path = os.path.join(IMAGE_ROOT, file_name)
        
        if not os.path.exists(full_path):
            continue

        # Read image and convert to RGB (SAM2 requirement)
        image_bgr = cv2.imread(full_path)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Pre-compute image embeddings (expensive step done once per image)
        predictor.set_image(image_rgb)

        # Iterate through all bounding boxes for this specific image
        for ann in anns:
            bbox = ann['bbox']  # Format: [x, y, width, height]
            x, y, w, h = bbox
            box_input = np.array([x, y, x + w, y + h])  # Convert to [x1, y1, x2, y2]

            # SAM2 Inference: Use the box as a prompt
            # multimask_output=False since we assume one object per box
            masks, scores, _ = predictor.predict(box=box_input, multimask_output=False)
            best_mask = masks[0].astype(np.uint8)
            
            # Post-processing: Mask to simplified Polygon
            segmentation = mask_to_polygons(best_mask)
            
            # Update annotation if a valid mask was found
            if len(segmentation) > 0:
                ann['segmentation'] = segmentation
                processed_count += 1
            
            new_annotations.append(ann)

    # --- 4. Result Serialization ---
    data['annotations'] = new_annotations
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    
    print(f"Saving results to: {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        # Use indent=4 for human-readable output, remove for smaller file size
        json.dump(data, f, indent=4)

    print("\n" + "="*50)
    print(f"Inference Complete!")
    print(f"Successfully generated segmentation for {processed_count} instances.")
    print("="*50)

if __name__ == "__main__":
    main()
