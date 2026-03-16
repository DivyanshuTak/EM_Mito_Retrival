# EM Mitochondria Feature Extraction & Retrieval 

## Project Overview

This repo contains the code to extract features and perform zeroshot retrieval for mitochondria from EM data using [DINOv3](https://github.com/facebookresearch/dinov3) and [UNI](https://huggingface.co/MahmoodLab/UNI) pretrained models. The workflow is data download, patch extraction, feature extraction, retrieval evaluation and visualiztion. Feature extaction can done with both patch-level features ( CLS, masked mean pooled) and dense token features. Two types of retrieval are performed - instance retrieval ( retriving mitocondria patch from same instance in the dataset) and semantic retrieval ( retriving mitocondria path from mitocondria and backgroun patches). The writeup of the project can be found [here](https://docs.google.com/document/d/1Tl2-BRAkeHL12qqEcJfXIVlCqYdkf4I2ZmuCkRbm0To/edit?usp=sharing)

---

## Project Structure

```text
EM_Mito_retrieval/
├── README.md
├── requirements.txt
├── config/                     # YAML configs for DINOv3 and UNI for model name, datapath, output path
├── scripts/                    # all pipeline scripts
├── notebooks/                  # analysis notebook
├── data/                       # downloaded slices, extracted patches and labels 
├── results/                    # extracted feature files and retrieval metrics
└── saliency/                   # saliency maps 
```



---

## Environment Setup

This project utilizes the UV environment manager. All required dependencies are listed in `requirements.txt`. To set up and activate the environment with UV, please run the following commands:

```bash
cd EM_Mito_retrieval/

uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## Data Acquisition & Preprocessing

### 1. Download EM Slices and Labels

The mitochondria data is downladed from the openorganelle [portal](https://openorganelle.janelia.org/organelles/mito). For this project 2D image slices with segmentation labels are downloaded for jrc-hela_2 and jrc-macrophage_2 datasets.  To download the data slices from the portal run the following commands.

```bash
# change the dataset name, number of slices accordingly 
python scripts/download_data.py \
  --datasets jrc_hela-2 jrc_macrophage-2 \
  --num-slices 200 \
  -o data/em_slices

python scripts/download_labels.py \
  --datasets jrc_hela-2 jrc_macrophage-2 \
  -o data/em_slices
```

### 2. Extract Centered Mitochondria Patches

To extract mitochondria patches from the 2D slices and build the dataset for model inference, first modify the configuration block at the top of `scripts/extract_mito_patches.py` (`DATA_DIR`, `DATASETS`, `OUTPUT_PATH`, `PATCH_SIZE`) accordingly, then run:

```bash
python scripts/extract_mito_patches.py
```



### 3. Extract Background Patches for Semantic Retrieval

To extract background patches from the 2D slices for semantic retrieval (mitochondria vs non-mito background) run:

```bash
python scripts/extract_background_patches.py \
  -n 642 \
  -o data/patches/jrc_macrophage2_background_256.npz \
  --data-dir data/em_slices \
  --datasets jrc_macrophage-2 \
  --patch-size 256
```
### 4. Data Structure

Download the 2D EM image slices, corresponding label files, and pre-extracted mito/background patches used for this project [here](https://www.dropbox.com/scl/fo/dfsy2rplq2q72kiuy2jc0/AHgAtiEZOR3OVDHg23W6hrM?rlkey=bru3qtki4s8ajz899glqepolt&st=41tc6cmk&dl=0).
To run retrieval, and reproduce the writeup results, organize your data and features into the following directory structure:

```text
data/
├── em_slices/                # 2D slices form the 3D EM volume 
│   ├── jrc_hela-2/
│   │   ├── slice_000.npy
│   │   ├── slice_001.npy
│   │   └── labels/
│   │       ├── slice_000.npy
│   │       └── slice_001.npy
│   └── jrc_macrophage-2/
│       ├── slice_000.npy
│       └── labels/
|            ├── slice_000.npy
└── patches/                    # mitochondria and background patches with patch/tile size 256 and metadata
    ├── jrc_hela2_mito_dataset_256.npz
    ├── jrc_hela2_background_256.npz
    ├── jrc_macrophage2_mito_dataset_256.npz
    └── jrc_macrophage2_background_256.npz
```


---

## Feature Extraction

The embeddings from DINOv3 (ViTs, ViTl) and UNI (ViTl) used in this project are provided in `/results`. 

To extract the CLS embeddings run: 

```bash
# FOR DINOV3 Update config file with current patch size, dataset name, and input path
python scripts/update_config.py config/feature_extraction.yaml \
  --patch-size 256 \
  --dataset-name jrc_macrophage2 \
  --input-npz data/patches/jrc_macrophage2_mito_dataset_256.npz

# Extract features 
python scripts/extract_dino_features.py -c config/feature_extraction.yaml

# For UNI, update the corresponding config and run extraction
python scripts/update_config.py config/uni.yaml \
  --patch-size 256 \
  --dataset-name jrc_macrophage2 \
  --input-npz data/patches/jrc_macrophage2_mito_dataset_256.npz

python scripts/uni_extract_cls.py -c config/uni.yaml
```



Mitochondria-only embeddings can be extracted by doing mean pooling over the 16×16 spatial tokens corresponding to each mitochondrion region using the segmentation mask:

```bash
# DINOv3
python scripts/extract_dino_features_masked.py -c config/feature_extraction.yaml

# UNI
python scripts/uni_extract_masked.py -c config/uni.yaml
```

#### Dense Embedding Extraction

Checkout the writeup behind the logic of dense embedding extraction. Dense embeddings can be extracted by running:

```bash
# change the config file for DINOv3 or UNI model 
python scripts/extract_dense_shifted.py -c config/feature_extraction.yaml
```

NOTE: Dense embeddings are extremely large so sharding is done and sharded files are stored in`.npy` output. Dense embeddings for UNI model on jrc_machophage-2 can be downloaded from [here](https://www.dropbox.com/scl/fo/dfsy2rplq2q72kiuy2jc0/AHgAtiEZOR3OVDHg23W6hrM?rlkey=bru3qtki4s8ajz899glqepolt&st=41tc6cmk&dl=0).

---

## Retrieval 

### Visualization

The retrieval is done using the extracted embeddings. Within-dataset and cross-dataset retrieval can be performed by changing the query and database npz files accordingly.
Modify the configuration block at the top of `scripts/visualize_retrieval.py` (`QUERY_FEATURE_NPZ`, `QUERY_IMAGE_NPZ`, `DB_DATASETS`) accordingly, then run:

```bash
python scripts/visualize_retrieval.py
```

### Evaluation 

Retrieval evaluation is on the mAP and Precision@5 metrics, for instance retrieval the labels are same mitocondria instance per patch, and for semantic retrieval the labels are mitochondria vs backround. 
NOTE : Instance retrieval can be done using both the CLS and mitochondria-embedding. However, Semantic retrieval can only be done using CLS embedding. Similar to visualizaiton, the within-dataset and cross-dataset retrieval can be performed by changing the query and database npz files accordingly.


```bash
# saves the output metrics in a json file 
python scripts/evaluate_retrieval.py \
  -q results/features_maskedmeanpooled_dinov3_vits16_pretrain_lvd1689m_masked_jrc_macrophage2_patch256.npz \
  -d results/features_maskedmeanpooled_dinov3_vits16_pretrain_lvd1689m_masked_jrc_macrophage2_patch256.npz \
     results/features_background_dinov3_vits16_pretrain_lvd1689m_patch256_jrc_macrophage2_bg.npz \
  -m semantic \    # change to 'instance' for instance retieval evaluation 
  -k 10 \
  -o results/retrieval/dino_retrieval_macrophage_semantic.json
```


---

## Analysis 

`notebooks/analysis.ipynb` contains analysis blocks documentation with runnable cells:
mito width distribution, patch sanity checks, saliency maps (CLS), PCA-RGB semantic maps.

