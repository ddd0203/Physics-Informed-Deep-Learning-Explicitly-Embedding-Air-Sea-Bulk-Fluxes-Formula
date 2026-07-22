# Physics-Informed Vision Transformer for Global Upper-Ocean Forecasting

Official implementation of the physics-informed Vision Transformer developed
for global upper-ocean forecasting.

The framework explicitly incorporates air–sea bulk fluxes calculated using the
COARE formulation and predicts the evolution of upper-ocean temperature,
salinity, and horizontal currents.

This repository provides the model implementation, COARE-based flux
calculation, data-loading utilities, training modules, inference scripts, and a
Jupyter/Google Colab notebook demonstrating the high-resolution training and
inference workflows described in the manuscript.

## Repository status

The code required to run the released high-resolution training and inference
examples is publicly available.

Large model checkpoints and example datasets are distributed separately
through Google Drive because they are too large to be stored directly in this
GitHub repository.

The complete original reanalysis, operational forecast, and observational
datasets used in the manuscript are not redistributed. These datasets should
be obtained from their original data providers according to the data
availability information given in the manuscript.

## Main features

- Physics-informed Vision Transformer for upper-ocean forecasting.
- Explicit embedding of the COARE 3.5 air–sea bulk flux formulation.
- Prediction of temperature, salinity, and horizontal-current tendencies.
- Training and validation workflow for the high-resolution experiment.
- Autoregressive inference initialized from GLO12v4 forecast fields.
- Atmospheric forcing based on ERA5 variables.
- Pretrained model checkpoints for reproducing the released inference example.
- Training-set mean and standard-deviation files for normalization and
  denormalization.

## Repository structure

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
