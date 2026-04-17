# -----------------------------------------------------------------------------
# Script Name: step1_inference.py
#
# Description:
# 1. Read the raw dataset JSON and extract valid image paths.
# 2. Perform batch inference using YOLO11x.
# 3. Serialize and save the inference results to a local JSON file (cache).
#    - Cache Structure: Key=relative_path(rel_path), Value=[[x1, y1, x2, y2, conf, cls], ...]
# -----------------------------------------------------------------------------

import json
import os
import math
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from ultralytics import YOLO

# ================= Configuration Area =================
# Input Paths (Replace these with your actual paths)
FISHNET_LIST_JSON = "path/to/your/annotations.json"  # Raw list format JSON
IMAGE_ROOT = "path/to/your/image"            # Root directory of images
YOLO_MODEL_PATH = "path/to/your/fishlabel_printer_yolo11x.pt"             # Path to your trained YOLO model

# Output File Path
OUTPUT_CACHE_JSON = "annotations/yolo_inference_cache.json"

# Inference Parameters
NUM_GPUS = 2         # Number of GPUs to use
BATCH_SIZE = 16      # Batch size per GPU
CONF_MIN = 0.1       # Minimum confidence threshold
CHUNK_SIZE = 144     # <--- CRITICAL: Number of images fed to YOLO at once
WORKERS = 0          # <--- CRITICAL: Must be 0 to prevent multiprocessing deadlock
# ======================================================

def run_worker(gpu_id, image_paths, abs_to_rel_map, return_dict):
    """
    Worker process: Responsible for processing the assigned subset of image paths.
    """
    print(f"[GPU {gpu_id}] Initializing... Assigned {len(image_paths)} images.")
    
    # 1. Initialize the model (loaded independently in each process to avoid conflicts)
    try:
        model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"[GPU {gpu_id}] Model Load Error: {e}")
        return

    local_cache = {}
    total_imgs = len(image_paths)
    
    # 2. Inner loop: Slice the assigned tasks into chunks
    # Use the 'position' parameter to stack progress bars neatly across GPUs
    pbar = tqdm(total=total_imgs, desc=f"GPU {gpu_id}", position=gpu_id)
    
    for i in range(0, total_imgs, CHUNK_SIZE):
        # Extract a chunk of images
        chunk = image_paths[i : i + CHUNK_SIZE]
        
        try:
            # 3. Run inference
            # workers=0 ensures data is read in the main thread of the current subprocess, ensuring stability
            results = model.predict(
                chunk,
                batch=BATCH_SIZE,
                device=gpu_id,  # Force the current process to use only this GPU
                stream=True,
                verbose=False,
                conf=CONF_MIN,
                workers=WORKERS 
            )
            
            # 4. Process results
            for res in results:
                # Get absolute path -> Look up relative path in dictionary
                # Note: Ultralytics might change path separators, use abspath for consistency
                abs_path = os.path.abspath(res.path)
                rel_path = abs_to_rel_map.get(abs_path)
                
                if rel_path:
                    boxes = res.boxes.data.cpu().numpy().tolist()
                    if len(boxes) > 0:
                        local_cache[rel_path] = boxes
                
                pbar.update(1)
                
        except Exception as e:
            print(f"\n[GPU {gpu_id}] Error in chunk {i}: {e}")
            # Do not interrupt on error, continue to the next chunk
            continue

    pbar.close()
    
    # 5. Return results
    return_dict[gpu_id] = local_cache
    print(f"[GPU {gpu_id}] Finished. Found targets in {len(local_cache)} images.")


def main():
    # Set multiprocessing start method to 'spawn' (required for CUDA)
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # --- 1. Organize paths (Main Process) ---
    print(f"Loading Raw Data: {FISHNET_LIST_JSON}...")
    with open(FISHNET_LIST_JSON, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    valid_paths = []    # List of absolute paths
    abs_to_rel = {}     # Mapping dictionary

    print("Checking files...")
    for entry in tqdm(raw_data, desc="Scanning"):
        family = entry.get('family')
        fname = entry.get('image')
        if not family or not fname: continue

        rel_path = os.path.join(family, fname)
        full_path = os.path.abspath(os.path.join(IMAGE_ROOT, rel_path))
        
        if os.path.exists(full_path):
            valid_paths.append(full_path)
            abs_to_rel[full_path] = rel_path

    total_images = len(valid_paths)
    print(f"\nTotal valid images: {total_images}")
    if total_images == 0: return

    # --- 2. Task Distribution (Split Tasks) ---
    # Split valid_paths evenly across GPUs
    chunk_size_per_gpu = math.ceil(total_images / NUM_GPUS)
    gpu_tasks = []
    
    for i in range(NUM_GPUS):
        start_idx = i * chunk_size_per_gpu
        end_idx = min((i + 1) * chunk_size_per_gpu, total_images)
        task_subset = valid_paths[start_idx : end_idx]
        gpu_tasks.append(task_subset)
        print(f"Task for GPU {i}: {len(task_subset)} images")

    # --- 3. Start multiprocessing ---
    print("\nStarting Multi-GPU Inference Process...")
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []

    for gpu_id in range(NUM_GPUS):
        if len(gpu_tasks[gpu_id]) > 0:
            p = mp.Process(
                target=run_worker,
                args=(gpu_id, gpu_tasks[gpu_id], abs_to_rel, return_dict)
            )
            processes.append(p)
            p.start()

    # --- 4. Wait for completion ---
    for p in processes:
        p.join()

    # --- 5. Merge results ---
    print("\nMerging results from GPUs...")
    final_cache = {}
    for gpu_id in range(NUM_GPUS):
        if gpu_id in return_dict:
            data = return_dict[gpu_id]
            final_cache.update(data)
            print(f"Merged {len(data)} results from GPU {gpu_id}")

    # --- 6. Save ---
    print(f"Saving final cache to {OUTPUT_CACHE_JSON}...")
    os.makedirs(os.path.dirname(OUTPUT_CACHE_JSON), exist_ok=True)
    with open(OUTPUT_CACHE_JSON, 'w', encoding='utf-8') as f:
        json.dump(final_cache, f)
    
    print("Step 1 (Multi-GPU Chunked Inference) Complete!")

if __name__ == "__main__":
    main()
