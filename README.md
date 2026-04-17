# OpenAqua: A Large-Scale Fine-Grained Dataset and Benchmark for Open Underwater Visual Perception

<div align="center">
  <img src="./radar.png" alt="Model Performance Across Taxonomic Levels" width="500">
  <p><em>Cross-level zero-shot performance evaluation</em></p>
</div>

## 1. Introduction

Monitoring aquatic biodiversity is vital for maintaining global ecological balance. While advancements in computer vision have revolutionized underwater perception, existing datasets are predominantly limited to coarse-grained categories or lack spatial localization annotations, severely constraining the applicability of models for fine-grained biological identification in real-world scenarios. 

To address this gap, we introduce **OpenAqua**, the first large-scale fine-grained dataset dedicated to open underwater visual tasks. OpenAqua is structured around a five-level biological taxonomic hierarchy, comprising 77,970 high-quality images covering 16,540 aquatic species, and providing 132,885 fine-grained bounding boxes and corresponding instance segmentation masks. 

Based on this dataset, we establish a comprehensive benchmark suite that encompasses not only standard object detection and instance segmentation tasks but also pioneers an underwater Open-Vocabulary Object Detection benchmark. Extensive experimental evaluations reveal a significant performance degradation in current models as they progress from coarse-grained to fine-grained recognition. These results highlight the substantial challenges associated with fine-grained semantic perception and domain adaptation in degraded underwater environments. We believe OpenAqua holds the potential to advance fine-grained underwater vision research, facilitate learning from long-tailed distributions, and enable more effective aquatic ecosystem monitoring.

## 2. Dataset Download

All data for OpenAqua (including images, annotation files, and reference physical prior files) can be accessed here: 

Google Drive: https://drive.google.com/drive/folders/1vo-b8BYGD2Kb2J__HTQK28yn5RzytMnz?usp=drive_link

Baidu Drive: https://pan.baidu.com/s/1PJoL3u4vlRG03Y83nmsw0Q?pwd=7ist

## 3. Data Annotation Pipeline

To efficiently construct a large-scale and high-quality dataset, we adopted a robust two-step annotation pipeline that combines automated pseudo-labeling with meticulous human refinement.

#### Step 1: Model-Assisted Coarse Pre-annotation
To establish a versatile fish detector with strong generalization capabilities across complex underwater conditions (e.g., high turbidity, uneven illumination), we aggregated multiple existing underwater object detection datasets. This collection includes Brackish, DeepFish, RUOD, UIIS10K, USIS10K, and Aquarium UOD. 

Using this merged dataset, we trained a **YOLO11x** model as our baseline detector to generate pseudo-labels. During inference, the confidence threshold was intentionally set to `0.2` to maximize recall and capture as many potential targets as possible. To facilitate large-scale processing, we provide a [batch inference script](./pseudo_boxlabel_inference_yolo11x.py) for efficient pseudo-boxlabel generation.

Below, we show the evaluation metrics of this baseline model alongside its complete training logs and pre-trained weights:

| Annotator | $AP_{50}$ | $AP_{50}^{fish}$ | Code | Weights |
| :--- | :---: | :---: | :---: | :---: |
| **YOLO11x** | 76.9 | 86.6 | [link](https://github.com/ultralytics/ultralytics) | [model](https://drive.google.com/file/d/1D7m5j2qZjq-9i-Id9K8nbSzO1vgalJU8/view?usp=sharing) \| [log](https://drive.google.com/file/d/1Ve7DrpN-GjfI2eqSLxRgvx_V9CIedI-S/view?usp=sharing) |

#### Step 2: Fine-grained Human Refinement
To ensure the absolute precision of the final dataset, we utilized the [T-Rex Labeling Platform](https://www.trexlabel.com/) for the manual correction and verification of the initial bounding boxes generated in Step 1.

#### Step 3: Instance Mask Annotation
In the final stage, we leverage Segment Anything Model 2 (SAM 2) to convert the refined bounding boxes into high-quality instance segmentation masks. The human-corrected boxes serve as precise visual prompts for SAM 2. To facilitate large-scale processing, we provide a [batch inference script](./pseudo_masklabel_inference_sam2.py) for efficient pseudo-masklabel generation. To ensure absolute accuracy, these generated masks undergo a secondary manual review and refinement process, resulting in the final fine-grained labels.

We provide SAM 2 pre-train weight resource for the mask generation stage:

| Annotator | Code | Weights |
| :--- | :---: | :---: |
| SAM 2 | [link](https://github.com/facebookresearch/sam2) | [sam2.1_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) |

## 4. Benchmarks

### 🐟 Object Detection Benchmark

Our object detection benchmark includes a comprehensive range of general-purpose detectors, covering two-stage, single-stage, and Transformer-based architectures. Furthermore, we evaluate specialized Underwater Object Detection (UOD) methods, encompassing approaches driven by data augmentation, architecture optimization, and physical prior guidance.

You can click the `link` in the table to quickly navigate to the open-source codebase of each method:

| Method | Code |
| :--- | :--- |
| **YOLO11m** | [link](https://github.com/ultralytics/ultralytics) |
| **GCC-Net** | [link](https://github.com/Ixiaohuihuihui/GCC-Net/tree/main) |
| **UnitModule** | [link](https://github.com/LEFTeyex/UnitModule) |
| **AMSP-UOD** | [link](https://github.com/zhoujingchun03/AMSP-UOD) |
| **Hy-UOD** | [link](https://github.com/White-cat-ed/HyUOD) |

### 🌊 Instance Segmentation Benchmark

In addition to Object Detection tasks, we have established Instance Segmentation Benchmark. All experiments uniformly adopt **ResNet-101** as the feature extraction backbone to ensure fair comparisons.

You can click the `link` in the table to quickly navigate to the open-source codebase of each method:

| Method | Backbone | Code |
| :--- | :---: | :--- |
| **Mask R-CNN** | ResNet-101 | [link](https://github.com/open-mmlab/mmdetection/blob/main/configs/mask_rcnn/mask-rcnn_r101_fpn_2x_coco.py) |
| **YOLACT** | ResNet-101 | [link](https://github.com/dbolya/yolact) |
| **CondInst** | ResNet-101 | [link](https://github.com/YuqingWang1029/CondInst) |
| **BoxInst** | ResNet-101 | [link](https://github.com/aim-uofa/AdelaiDet/blob/master/configs/BoxInst/README.md) |
| **WaterMask** | ResNet-101 | [link](https://github.com/LiamLian0727/WaterMask) |

### 🌐 Open-Vocabulary Object Detection Benchmark

To facilitate reproduction and further research, we provide the specific configuration files and pre-trained weights for each model variant. The **Method** column links to the official source repository, while the **Config** and **Weights** columns point to the exact files used in our benchmark. Additionally, we provide a detailed multi-level taxonomic vocabulary ([Class](./classes_class.json), [Order](./classes_order.json), [Family](./classes_family.json), [Genus](./classes_genus.json), [Species](./classes_species.json)).

| Method (Repo) | Backbone | Params | Pre-trained Data | Config | Weights |
| :--- | :--- | :---: | :--- | :---: | :---: |
| [**Grounding DINO**](https://github.com/open-mmlab/mmdetection/tree/main/configs/mm_grounding_dino) | Swin-T | 172M | O365, GoldG, V3Det | [config](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_v3det.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_20231218_095741-e316e297.pth)      \| [log](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_v3det_20231218_095741.log.json) |
| | Swin-B | 233M | O365, GoldG, V3Det | [config](https://github.com/open-mmlab/mmdetection/blob/main/configs/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_obj365_goldg_v3det/grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth) |
| | Swin-L | 341M | O365V2, OpenImageV6, GoldG | [config](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg.py) | [model](https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth) |
| [**YOLO-World**](https://github.com/AILab-CVC/YOLO-World) | YOLOv8-S | 77M | O365+GoldG | [config](https://github.com/AILab-CVC/YOLO-World/blob/master/configs/pretrain_v1/yolo_world_s_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) | [model](https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_s_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-18bea4d2.pth) |
| | YOLOv8-M | 92M | O365+GoldG | [config](https://github.com/AILab-CVC/YOLO-World/blob/master/configs/pretrain_v1/yolo_world_m_dual_l2norm_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) | [model](https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_m_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-2b7bd1be.pth) |
| | YOLOv8-L | 110M | O365+GoldG | [config](https://github.com/AILab-CVC/YOLO-World/blob/master/configs/pretrain_v1/yolo_world_l_dual_vlpan_l2norm_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py) | [model](https://huggingface.co/wondervictor/YOLO-World/resolve/main/yolo_world_l_clip_base_dual_vlpan_2e-3adamw_32xb16_100e_o365_goldg_train_pretrained-0e566235.pth) |
| [**LLMDet**](https://github.com/iSEE-Laboratory/LLMDet) | Swin-T | 172M | GroundingCap-1M | [config](https://github.com/iSEE-Laboratory/LLMDet/blob/main/configs/grounding_dino_swin_t.py) | [model](https://huggingface.co/fushh7/LLMDet/blob/main/tiny.pth)\| [log](https://huggingface.co/fushh7/LLMDet/blob/main/tiny.log) |
| | Swin-B | 233M | GroundingCap-1M | [config](https://github.com/iSEE-Laboratory/LLMDet/blob/main/configs/grounding_dino_swin_b.py) | [model](https://huggingface.co/fushh7/LLMDet/blob/main/base.pth)\| [log](https://huggingface.co/fushh7/LLMDet/blob/main/base.log) |
| | Swin-L | 341M | GroundingCap-1M | [config](https://github.com/iSEE-Laboratory/LLMDet/blob/main/configs/grounding_dino_swin_l.py) | [model](https://huggingface.co/fushh7/LLMDet/blob/main/large.pth)\| [log](https://huggingface.co/fushh7/LLMDet/blob/main/large.log) |

> **💡 Note:** > - **Config:** Links to the specific `.py` or `.yaml` files in the official repositories.
> - **Weights:** Links to the pre-trained checkpoints (e.g., Hugging Face or Google Drive).

## 🙏 Acknowledgments

We thank [FishNet](https://github.com/faixan-khan/FishNet/) and [iNaturalist](https://www.inaturalist.org/) for providing the image data, [MMDetection](https://github.com/open-mmlab/mmdetection) and [Ultralytics](https://github.com/ultralytics/ultralytics) for their open-source toolboxes, and the [T-Rex Labeling Platform](https://www.trexlabel.com/) for annotation support.

## ⚖️ License and Disclaimer

The OpenAqua dataset (including its annotations, splits, and evaluation protocols) is released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/) (CC BY-NC-SA 4.0).

**Data Sources Disclaimer:**
The raw images within the OpenAqua dataset are aggregated from [FishNet](https://github.com/faixan-khan/FishNet/) and [iNaturalist](https://www.inaturalist.org/). 
- We do not claim ownership of the original images. The copyrights of the original images remain with their respective creators and owners.
- The dataset is curated and provided **strictly for academic, non-commercial research purposes**. 
- If you are a copyright holder and wish to have your image removed from this dataset, please contact us, and we will remove it promptly.

By downloading and using this dataset, you agree to abide by these terms.
