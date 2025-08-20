# 📘 Notebooks Overview: Interpretable ML for Free-Fall Motion

This folder contains a series of modular notebooks exploring free-fall motion through interpretable machine learning techniques. Each notebook builds on the previous one, documenting a scientific journey from clean data modeling to robust experimentation.

## 📚 Contents

### `01_free_fall_motion_clean.ipynb`
- Models free-fall motion using linear regression in PyTorch
- Compares MAE and MSE loss functions with and without normalization
- Recovers physically meaningful parameters and visualizes predictions
- Implements early stopping and best model checkpointing

### `02_free_fall_with_noise.ipynb` *(planned)*
- Adds controlled noise to simulate real-world measurement uncertainty
- Repeats experiments to assess robustness of loss functions and parameter recovery

### `03_loss_function_comparison.ipynb` *(planned)*
- Explores alternative loss functions such as Huber and Log-Cosh
- Evaluates trade-offs between sensitivity, stability, and interpretability

### `04_regularization_and_constraints.ipynb` *(planned)*
- Introduces physical constraints and regularization techniques
- Investigates their impact on parameter realism and generalization

## 🛠️ Utilities

### `utils/denormalizing.py`
- Contains `denormalize_parameters()` function to reverse normalization and recover interpretable model weights

### `utils/plot_predictions.py` *(optional)*
- Utility for visualizing predicted vs. true trajectories

## 🧠 Philosophy

This project is driven by a desire to blend machine learning with physics in a way that preserves meaning, interpretability, and scientific rigor. Each notebook is a step toward building a portfolio that reflects thoughtful experimentation and reproducible insight.

> These notebooks are designed to be modular, extensible, and narratively rich — ideal for showcasing scientific modeling in a modern ML framework.
