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
