# OpenAqua: A Large-Scale Fine-Grained Dataset and Benchmark for Open Underwater Visual Perception
<div align="center">
  <img src="./radar" alt="Model Performance Across Taxonomic Levels" width="400">
  <p><em>Cross-level zero-shot performance evaluation</em></p>
</div>
## 1. Introduction

Monitoring aquatic biodiversity is vital for maintaining global ecological balance. While advancements in computer vision have revolutionized underwater perception, existing datasets are predominantly limited to coarse-grained categories or lack spatial localization annotations, severely constraining the applicability of models for fine-grained biological identification in real-world scenarios. 

To address this gap, we introduce **OpenAqua**, the first large-scale fine-grained dataset dedicated to open underwater visual tasks. OpenAqua is structured around a five-level biological taxonomic hierarchy, comprising 77,970 high-quality images covering 16,540 aquatic species, and providing 132,885 fine-grained bounding boxes and corresponding instance segmentation masks. 

Based on this dataset, we establish a comprehensive benchmark suite that encompasses not only standard object detection and instance segmentation tasks but also pioneers an underwater Open-Vocabulary Object Detection benchmark. Extensive experimental evaluations reveal a significant performance degradation in current models as they progress from coarse-grained to fine-grained recognition. These results highlight the substantial challenges associated with fine-grained semantic perception and domain adaptation in degraded underwater environments. We believe OpenAqua holds the potential to advance fine-grained underwater vision research, facilitate learning from long-tailed distributions, and enable more effective aquatic ecosystem monitoring.

## 2. Dataset Download

All data for OpenAqua (including images, annotation files, and reference physical prior files) can be accessed here: [Insert Link Here]
