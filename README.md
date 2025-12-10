# Projet final FS37 - VitiScan model

__Auteurs__ : Mounia, Inti, Samuel, Guillaume

## Préparation de l'environnement

- activer conda
`source ~/app/miniconda3/etc/profile.d/conda.sh && conda activate`

- vous devez avoir un prompt du type
`(base)$`

- créer l'environnement python avec conda

`(base)$ conda env create -n vitiscan_cnn --file env_vitiscan_cnn.yml`

ou

`(base)$ conda env create -f env_vitiscan_cnn.yml`

- si vous avec un GPU Nvidia, décommenter la ligne `- nvidia` dans les channels et `- pytorch-cuda=12.4` dans les dependencies

- activer l'environnement

`(base)$ conda activate vitiscan_cnn`

- déclarer le noyau ipykernel pour Jupyter

`(jedha_rag)$ python -m ipykernel install --user -n vitiscan_cnn --display-name "Vitiscan CNN"`


## Préparation des données

[Dataset Kaggle Grapes Leafs Disease PlantCity 2025](https://www.kaggle.com/datasets/codewithsk/grapes-leafs-disease-7-classes-plantcity-2025)


## Lancement de l'entrainement des données



