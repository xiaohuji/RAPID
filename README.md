# RADIP

### **Radiology-Supervised Multimodal Artificial Intelligence for Histologic Classification of Renal Cell Carcinoma on Needle Biopsy**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C.svg)](https://pytorch.org/)[![Research](https://img.shields.io/badge/Use-Research%20Only-lightgrey.svg)](#license)

RADIP is a multimodal learning framework that transfers complementary radiological information from CT and MRI into a pathology-only model. During training, radiology-aware teacher networks use paired CT/MRI and whole-slide image (WSI) features. Their predictions and representations are then distilled into a WSI student, enabling pathology-only inference when radiology is unavailable.

<p align="center">
  <img src="img/Graphical Abstract.jpg" width="100%" alt="Overview of the RADIP framework">
</p>

## Repository structure

```text
RADIP/
├── _1_feature_extracting/
│   ├── _1_feature_extraction_CT/       # Merlin-based CT features
│   ├── _2_feature_extraction_MRI/      # 3DINO-based MRI features
│   ├── _3_feature_extraction_WSI_UNI/  # UNI WSI patch features
│   ├── _4_feature_extraction_WSI_CONCH/# CONCH WSI patch features
│   ├── datasets/                       # CT/MRI preprocessing and datasets
│   └── DINO_main/                      # Included 3DINO implementation
├── _2_tumor_patch_identification/      # Zero-shot tumor filtering and QC
├── _3_multimodal_teacher_training/     # CT–WSI and MRI–WSI teacher training
├── _4_student_training/                # Pathology-only knowledge distillation
├── _5_teacher_eval/                    # Teacher evaluation
└── _6_student_eval/                    # Student evaluation
```

## Requirements

The included 3DINO code targets **Python 3.12**, **PyTorch 2.0.0**, and **torchvision 0.15.0**. A CUDA-capable GPU is strongly recommended for feature extraction and training.

### Core environment

```bash
conda create -n radip python=3.9 -y
conda activate radip

pip install -r _1_feature_extracting/DINO_main/requirements.txt
pip install numpy pandas h5py anndata SimpleITK \
  matplotlib seaborn scikit-learn scikit-image opencv-python \
  timm einops tqdm wandb huggingface-hub lazyslide wsidata \
  openslide-python
```

OpenSlide also requires a platform-specific system library. Follow the installation instructions for your operating system before processing WSIs.

## Configuration

Before running a stage, edit the path block near the top or inside the `if __name__ == "__main__"` section of the corresponding script. At minimum, configure:

- metadata CSV paths;
- raw CT/MRI and WSI roots;
- output feature directories;
- Merlin, 3DINO, UNI, and CONCH checkpoints;
- CT–WSI and MRI–WSI teacher checkpoints for distillation; and
- checkpoint/result output directories.

The training scripts expose optimization and model settings through `argparse`, but data paths in the current release remain script-level placeholders. Several experiment settings are also assigned in the main blocks; verify them before launching a new experiment.

## Running RADIP

All commands below are executed from the repository root unless stated otherwise. `PYTHONPATH` is set explicitly because each stage retains its original research-code module layout.

### 1. Install

On an NVIDIA RTX4090 Tensor Core GPU machine, with CUDA toolkit enabled.

1. Download and open our repository

```buash
git clone https://github.com/xiaohuji/RAPID
cd RAPID
```

1. Install dependencies

```bash
conda create -n youre_env_name
conda activate your_env_name
pip install requirement.txt
```

### 2. Extract radiology and pathology features

```bash
# CT features (Merlin)
python _1_feature_extracting/python _1_feature_extraction_CT/feature_extraction_CT.py

# MRI features (3DINO)
python _1_feature_extracting/_2_feature_extraction_MRI/feature_extraction_MRI.py

# WSI features (UNI and CONCH)
python _1_feature_extracting/_3_feature_extraction_WSI_UNI/uni_feature_extracting.py
python _1_feature_extracting/_4_feature_extraction_WSI_CONCH/conch_feature_extracting.py
```

The default WSI tessellation uses 256 × 256 pixel patches at 0.5 µm/pixel. Adjust `TILE_SIZE`, `TILE_MPP`, batch size, and device settings to match the scanner resolution and available hardware.

### 3. Identify tumor-associated WSI patches

```bash
# CONCH zero-shot tumor scoring
python _2_tumor_patch_identification/_1_tumor_patch_identification.py

# Transfer tumor labels to the corresponding UNI feature files
python _2_tumor_patch_identification/_2_tumor_transfer_to_uni.py

# Optional heatmap visualization for quality control
python _2_tumor_patch_identification/_3_tumor_visualization.py
```

The default tumor probability threshold is `0.5`. Review the generated distributions and spatial heatmaps before training, particularly when adapting RADIP to a new organ, stain, or scanner domain.

### 4. Train multimodal teachers

```bash
# CT–WSI teacher
PYTHONPATH=.:_3_multimodal_teacher_training \
python _3_multimodal_teacher_training/train_wsi_ct_teacher_merlin_MJ_rl_control_tumor_nd.py \
  --device 0 --epochs 50 --batch_size 1

# MRI–WSI teacher
PYTHONPATH=.:_3_multimodal_teacher_training \
python _3_multimodal_teacher_training/train_wsi_mri_teacher_DINO_rl_control_tumor_nd.py \
  --device 0 --epochs 50 --batch_size 1
```

### 5. Distill the pathology-only student

```bash
PYTHONPATH=.:_4_student_training \
python _4_student_training/train_student.py \
  --device 0 \
  --teacher_ct_ckpt /path/to/ct_teacher.pth \
  --teacher_mri_ckpt /path/to/mri_teacher.pth \
  --batch_size 1 \
  --use_teacher_conf
```

### 6. Evaluate

```bash
# Teacher evaluation
PYTHONPATH=.:_5_teacher_eval python _5_teacher_eval/eval_wsi_ct_teacher.py --device 0
PYTHONPATH=.:_5_teacher_eval python _5_teacher_eval/eval_wsi_mri_teacher.py --device 0

# Pathology-only student evaluation
PYTHONPATH=.:_6_student_eval python _6_student_eval/eval_student.py --device 0
```


