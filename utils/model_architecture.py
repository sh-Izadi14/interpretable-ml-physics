from torch import nn

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(in_features=2, out_features=1)

    def forward(self, x):
        return self.linear_layer(x)