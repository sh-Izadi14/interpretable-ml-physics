# 📘 Notebooks Overview: Interpretable ML for Free-Fall Motion

This folder contains a series of modular notebooks exploring free-fall motion through interpretable machine learning techniques. Each notebook builds on the previous one, documenting a scientific journey from clean data modeling to robust experimentation.

## 📚 Contents

### `01_free_fall_motion_clean.ipynb`
- Models free-fall motion using linear regression in PyTorch
- Compares MAE and MSE loss functions with and without normalization
- Recovers physically meaningful parameters and visualizes predictions
- Implements early stopping and best model checkpointing

### `02_free_fall_with_noise.ipynb`
- Adds controlled noise to simulate real-world measurement uncertainty
- Repeats experiments to assess robustness of loss functions and parameter recovery

### `03_loss_function_comparison.ipynb`
- Explores alternative loss function (Huber)
- Evaluates trade-offs between sensitivity, stability, and interpretability

## ⚙️ Utilities Overview

This project adopts a modular design for clarity, scalability, and ease of experimentation. Below is a breakdown of the utility modules located in the `utils/` directory:

### 1. `denormalizing.py`
- **`denormalizing()`**  
  Reverts normalized predictions back to their original scale for interpretability and evaluation.

---

### 2. `experiment_preparation.py`
Essential preprocessing functions for setting up experiments:
- **`adding_noise()`** – Adds controlled Gaussian noise to simulate real-world measurement errors.  
- **`normalization()`** – Scales input features to improve model convergence and stability.  
- **`splitting_data()`** – Splits the dataset into training and testing sets for evaluation.

---

### 3. `model_architecture.py`
Defines the model used for learning physical relationships:
- **`LinearRegressionModel(nn.Module)`**  
  A PyTorch-based linear regression model tailored for interpretable physics tasks.

---

### 4. `ploting_utils.py`
Visualization tools to monitor training and evaluate predictions:
- **`plot_loss_curves()`** – Plots training and test loss over epochs.  
- **`plot_predictions()`** – Visualizes model predictions against ground truth for qualitative assessment.

---

### 5. `recovered_parameters.py`
- **`recovered_parameters()`**  
  Extracts and displays learned model parameters, enabling interpretation in the context of physical laws.

---

### 6. `train_model.py`
Handles model training with early stopping and checkpointing:
- **`TrainingConfig`** – A dataclass encapsulating training hyperparameters and settings.  
- **`train_model()`** – Trains the model using the provided configuration, tracks losses, and saves the best-performing checkpoint.

---

## 🧠 Philosophy

This project is driven by a desire to blend machine learning with physics in a way that preserves meaning, interpretability, and scientific rigor. Each notebook is a step toward building a portfolio that reflects thoughtful experimentation and reproducible insight.

> These notebooks are designed to be modular, extensible, and narratively rich — ideal for showcasing scientific modeling in a modern ML framework.
