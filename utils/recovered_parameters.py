import torch
from model_architecture import LinearRegressionModel
from denormalizing import denormalize_parameters

def recovered_parameters(model_name, X_mean, X_std, y_mean, y_std):       
                
        state_dict = torch.load(f'../models/{model_name}.pth', weights_only=True)
        model = LinearRegressionModel()
        model.load_state_dict(state_dict)
                
        weights = state_dict['linear_layer.weight']
        bias = state_dict['linear_layer.bias']
        
        a, b, c = denormalize_parameters(weights, bias, X_mean, X_std, y_mean, y_std)
        print(f"Recovered Physical Parameters ({model_name} Loss):\n a = {a:.2f}, b = {b:.2f}, c = {c:.2f}\n")