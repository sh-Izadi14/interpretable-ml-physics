from dataclasses import dataclass
import torch
import numpy as np


torch.manual_seed(42)

@dataclass
class TrainingConfig:
    epochs: int
    window_size: int
    patience: int
    optimizer: torch.optim.Optimizer
    loss_fn: torch.nn.Module
    verbose: bool
    model_name: str


from collections import namedtuple
TrainResult = namedtuple("TrainResult", ["train_losses", "test_losses", "epochs_run"])

def train_model(model, X_train, y_train, X_test, y_test, config: TrainingConfig):
    """
    Train a PyTorch model with early stopping and save the best checkpoint.

    Parameters
    ----------
    X_train, y_train : torch.Tensor
        Training data and labels.
    X_test, y_test : torch.Tensor
        Validation data and labels.
    config : TrainingConfig
        All training hyperparameters and settings.
    """
    
    best_loss = np.inf
    patience_counter = 0
    prev_loss = float("inf")
    
    # Create empty loss lists to track values
    train_losses = []
    test_losses = []
    epochs_run = []
    recent_losses = []
    
    for epoch in range(config.epochs):
        # ---- Training step ----
        model.train()
        config.optimizer.zero_grad()
        y_pred = model(X_train)
        loss = config.loss_fn(y_pred, y_train)
        loss.backward()
        config.optimizer.step()

        # ---- Validation step ----
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test)
            test_loss = config.loss_fn(test_pred, y_test)

        test_loss_value = test_loss.item()
        train_losses.append(loss.item())
        test_losses.append(test_loss_value)
        
        # ---- Verbose logging ----
        if config.verbose and (epoch % config.window_size == 0 or epoch == config.epochs - 1):
            print(f"Epoch {epoch:5d} | Train Loss: {loss.item():.6f} | Test Loss: {test_loss.item():.6f}")
            
        # Save best model if this is the best so far
        if test_loss_value < best_loss:
            best_loss = test_loss_value
            torch.save(model.state_dict(), f"../models/{config.model_name}.pth")

        # Early stopping logic:
        if epoch % config.window_size == 0:
            if test_loss_value >= prev_loss:  # Only increment patience if loss is NOT decreasing compared to previous epoch
                patience_counter += 1
            else:
                patience_counter = 0  # reset if we are still improving compared to last epoch
    
        prev_loss = test_loss_value  # update for next iteration
    
        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch}")
            return TrainResult(train_losses, test_losses, epoch + 1)
            break

    print(f"Best model saved as {config.model_name}.pth with Test Loss: {best_loss:.6f}\n")
    return TrainResult(train_losses, test_losses, config.epochs)