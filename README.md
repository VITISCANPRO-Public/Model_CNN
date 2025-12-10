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

- déclarer le noyau ipykernel pour Jupyter (doit être fait avec l'environnement correspondant activé)

`(vitiscan_cnn)$ python -m ipykernel install --user --name vitiscan_cnn --display-name "Vitiscan CNN"`

- pour ajouter une lib, ajoutez là aux dépendances et faite
`conda env update -f env_vitiscan_cnn.yml`

## Préparation des données

[Dataset Kaggle Grapes Leafs Disease PlantCity 2025](https://www.kaggle.com/datasets/codewithsk/grapes-leafs-disease-7-classes-plantcity-2025)

A l'extraction du ZIP on s'aperçoit que :

- les images sont déjà augmentés par des flips
- doublonnage du nom de répertoire train/train ou test/test
- des noms de répertoire pour les classes pas bien normalisés


## Lancement de l'entrainement des données



