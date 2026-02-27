# Kaggle Healthy Leaves Dataset

Source: [Grapes Leafs Disease 7 Classes — PlantCity 2025](https://www.kaggle.com/datasets/codewithsk/grapes-leafs-disease-7-classes-plantcity-2025)

## Usage in this project

Only the **healthy** class is used from this dataset. The 6 disease classes come
from INRAE (higher quality labels). The Kaggle healthy images are merged from
both `train/` and `test/` splits and balanced to ~350 images.

## Why not use Kaggle for all classes?

The Kaggle dataset was initially used for all classes but was discarded due to
labelling inconsistencies. INRAE data provided higher quality ground truth for
disease classes.

> **Note:** This folder is in `.gitignore` — raw images are not committed.
> The dataset is downloaded automatically by `main.py` via the Kaggle API URL.