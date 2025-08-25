import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from model_architecture import LinearRegressionModel
from collections import deque

class FreeFallExperiment:
    def __init__(self, X, y, noise_level, loss_fn, epochs, patience, model_name):
        torch.manual_seed(42)
        self.X = X
        self.y = y
        self.noise_level = noise_level
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.patience = patience
        self.model_name = model_name
        

    def generate_data(self):
        # Add noise to y based on self.noise_level
        torch.manual_seed(42)
        noise_level = self.noise_level  # 5% noise
        std_relative = noise_level * self.y.abs().mean()  # or y.max() for stricter bound
        gaussian_noise = torch.randn_like(self.y) * std_relative
        self.y_noisy = self.y + gaussian_noise

        self.X_mean = self.X.mean(dim=0)
        self.X_std = self.X.std(dim=0)
        X_normalized = (self.X - self.X_mean)/(self.X_std + 1e-8)
        
        self.y_mean = self.y_noisy.mean()
        self.y_std = self.y_noisy.std()
        y_normalized = (self.y_noisy - self.y_mean)/(self.y_std + 1e-8)
        
        train_size = int(0.8 * len(self.y))
        self.X_train = X_normalized[:train_size]
        self.y_train = y_normalized[:train_size]
        self.X_test = X_normalized[train_size:]
        self.y_test = y_normalized[train_size:]

    def train_model(self):
        

        model = LinearRegressionModel()
        optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.01)
        loss_fn = self.loss_fn

        epochs = self.epochs

        window_size = 100  # Number of recent epochs to track
        patience = 1000    # How long to wait after plateau
        loss_window = deque(maxlen=window_size)
        best_loss = float('inf')
        trigger_times = 0

        # Create empty loss lists to track values
        self.train_loss_values = []
        self.test_loss_values = []
        self.best_loss_values= []
        self.epoch_count= []

        for epoch in range(epochs):
            model.train()
            
            y_preds = model(self.X_train)
            loss = loss_fn(y_preds, self.y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.inference_mode():
                test_preds = model(self.X_test)

            test_loss = loss_fn(test_preds, self.y_test.type(torch.float))
            
            current_loss = test_loss.item()
            loss_window.append(current_loss)
        
            if current_loss < best_loss:
                best_loss = current_loss
                self.best_loss_values.append(best_loss)
                torch.save(model.state_dict(), f'../models/{self.model_name}.pth')
                trigger_times = 0
            else:
                # Check if current loss is consistently worse than the average of recent losses
                if len(loss_window) == window_size and current_loss > sum(loss_window)/window_size:
                    trigger_times += 1
                    if trigger_times >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
        
        
            # Print out what's happening
            if epoch % 200 == 0:
                self.epoch_count.append(epoch)
                self.train_loss_values.append(loss.detach().numpy())
                self.test_loss_values.append(test_loss.detach().numpy())
                print(f"Epoch: {epoch} | Train Loss: {loss} | Test Loss: {test_loss} ")
        print(f"Best Loss Value: {best_loss}")
    
    def plot_loss_curves(self):
        loss_fn = self.loss_fn
        loss_type = loss_fn.__class__.__name__.replace("Loss", "") 
        plt.title(f'{loss_type} Loss Curves')
        plt.plot(self.epoch_count, self.train_loss_values, c ='b', label='Train Loss')
        plt.plot(self.epoch_count, self.test_loss_values, c = 'r', label='Test Loss')
        plt.plot(self.best_loss_values, '.', c='g', label="Best loss")
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend()
        
    
    def recovered_parameters(self):       
        from denormalizing import denormalize_parameters
        
        state_dict = torch.load(f'../models/{self.model_name}.pth', weights_only=True)
        model = LinearRegressionModel()
        model.load_state_dict(state_dict)
        
        
        weights = state_dict['linear_layer.weight']
        bias = state_dict['linear_layer.bias']

        loss_fn = self.loss_fn
        loss_type = loss_fn.__class__.__name__.replace("Loss", "")
        
        a, b, c = denormalize_parameters(weights, bias, self.X_mean, self.X_std, self.y_mean, self.y_std)
        print(f"Recovered Physical Parameters ({loss_type} Loss):\na = {a:.2f}, b = {b:.2f}, c = {c:.2f}")
    
    def plot_results(self):
        from plot_predictions import plot_predictions
        
        X_train_denorm = self.X_train * self.X_std + self.X_mean
        X_test_denorm  = self.X_test  * self.X_std + self.X_mean
        y_test_denorm  = self.y_test  * self.y_std + self.y_mean
        y_train_denorm  = self.y_train  * self.y_std + self.y_mean


        state_dict = torch.load(f'../models/{self.model_name}.pth', weights_only=True)
        model = LinearRegressionModel()
        model.load_state_dict(state_dict)
        
        model.eval()
        with torch.inference_mode():
          y_preds = model(self.X_test)
        
        y_pred_denorm  = y_preds  * self.y_std + self.y_mean
        
        plot_predictions(train_data=X_train_denorm, 
                             train_labels=y_train_denorm, 
                             test_data=X_test_denorm, 
                             test_labels=y_test_denorm, 
                             predictions=y_pred_denorm)
    def run(self):
        self.generate_data()
        self.train_model()