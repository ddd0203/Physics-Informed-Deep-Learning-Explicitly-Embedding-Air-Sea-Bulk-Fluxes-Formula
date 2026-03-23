# Physics-Informed-Deep-Learning-Explicitly-Embedding-Air-Sea-Bulk-Fluxes-Formula
Official repository for the physics-informed Vision Transformer (ViT) framework for global upper-ocean forecasting.

# Physics-Informed ViT for Global Upper-Ocean Forecasting

[![Status](https://img.shields.io/badge/Status-Under_Review-blue.svg)]()
[![Code](https://img.shields.io/badge/Code-Work_in_Progress-orange.svg)]()

This is the official code repository for our submitted manuscript on global upper-ocean forecasting (temperature, salinity, and surface currents) using a **Physics-Informed Vision Transformer (ViT)**.

## ⚠️ Repository Status (Work in Progress)
**Please note: The manuscript associated with this repository is currently under review.** 

At present, we have uploaded the preliminary version of our partial code (e.g., the core model architecture and the integration of the COARE bulk flux algorithm). 

To ensure code clarity and usability for the community, **the complete codebase — including the full data-processing pipelines, model training/testing scripts, and visualization programs — is currently being thoroughly cleaned and organized. The full source code will be made entirely public upon the formal acceptance of our paper.** 

## 🌟 Highlights
- ViT model embeds COARE formula for explicit air-sea flux parameterization.
- Tendency forecasting mitigates over-smoothing and preserves kinetic energy.
- Framework outperforms XiHe and GLO12v4 over a 10-day global ocean forecast.

## 📁 Repository Structure (Upcoming)
Once finalized, this repository will contain:
- `model/`: The core physics-informed ViT framework.
- `physics/`: Scripts for the COARE bulk formulae calculations.
- `train/` & `evaluate/`: Complete training pipelines and evaluation scripts.
- `visualization/`: Plotting scripts for metrics and spatial maps (e.g., capturing the cold wake of Typhoon Lekima).

## ✉️ Contact
If you have any questions regarding the partial code or the upcoming release, please feel free to open an issue or contact the corresponding author.
