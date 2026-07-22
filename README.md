# Physics-Informed ViT for Global Upper-Ocean Forecasting

Official code repository for the physics-informed Vision Transformer developed
for global upper-ocean forecasting.

The model explicitly embeds the COARE bulk-flux formulation and predicts
upper-ocean temperature, salinity, and horizontal currents.

This repository provides the model architecture, COARE-based flux calculation,
data-loading modules, training code, inference code, and a Google Colab notebook
for reproducing the high-resolution training and inference workflows.

## Repository files

```text
.
├── OM_revision_highres.ipynb
├── architectures.py
├── coare35vn.py
├── flux_util.py
├── forecast_real_from_GLO12v4_ERA5_nc.py
├── loader.py
├── loss_functions.py
├── meteo.py
├── model.py
├── test.py
├── train_module.py
├── trainer.py
└── vit_new.py

The recommended entry point is:

OM_revision_highres.ipynb

This notebook contains examples for:
training and validating the model;
loading a pretrained checkpoint;
running the inference workflow.

The notebook is designed to run in Google Colab.

Data and model weights
Pretrained model checkpoints

The pretrained model checkpoints are available at:

https://drive.google.com/drive/folders/18ION0rGY8yrRyQRoGll2uW10qW11Qv0B?usp=sharing

Download the required checkpoint and update the checkpoint path in
OM_revision_highres.ipynb.

Training and validation data

The example training data, validation data, and normalization statistics are
available at:

https://drive.google.com/drive/folders/1b76UsJbITAejPVgZvs4z3cF5Hq6-nZvG?usp=drive_link

This folder contains:

training data;
validation data;
mean values calculated from the training dataset;
standard-deviation values calculated from the training dataset.

The same mean and standard-deviation files must be used during training and
inference.
