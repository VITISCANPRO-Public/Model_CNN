# INRAE Disease Dataset

Grape leaf disease images from INRAE-labelled scientific sources.

## Classes (6 diseases)

| Folder name | Common name |
|---|---|
| `colomerus_vitis` | Erinose |
| `elsinoe_ampelina` | Anthracnose |
| `erysiphe_necator` | Powdery mildew |
| `guignardia_bidwellii` | Black rot |
| `phaeomoniella_chlamydospora` | Esca |
| `plasmopara_viticola` | Downy mildew |

## Structure

Each class is a folder containing `.jpg` images at the root of `data-inrae/`.

> **Note:** This folder is in `.gitignore` — raw images are not committed.
> The dataset is prepared locally before training via `main.py`.