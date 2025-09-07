import torch
from collections import namedtuple

def adding_noise(noise_level, y):
    torch.manual_seed(42)
    
    noise_level = noise_level
    std_relative = noise_level * y.abs().mean()  # or y.max() for stricter bound
    gaussian_noise = torch.randn_like(y) * std_relative
    y_noisy = y + gaussian_noise
    
    return y_noisy


NormalizationResult = namedtuple(
    "NormalizationResult",
    ["X_normalized", "y_normalized", "X_mean", "X_std", "y_mean", "y_std"]
)

def normalization(X, y):
    X_mean = X.mean(dim=0)
    X_std  = X.std(dim=0)
    X_normalized = (X - X_mean)/(X_std + 1e-8)
    
    y_mean = y.mean()
    y_std = y.std()
    y_normalized = (y - y_mean)/(y_std + 1e-8)

    return NormalizationResult(X_normalized, y_normalized, X_mean, X_std, y_mean, y_std)

def splitting_data(X, y):    
    train_size = int(0.8 * len(y))
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    return X_train, y_train, X_test, y_test