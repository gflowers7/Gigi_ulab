import torch
import numpy as np

#hidden function
x_data = np.linspace(-3 * np.pi, 3 * np.pi, 1000)
y_data = np.linspace(-3 * np.pi, 3 * np.pi, 1000)
X, Y = np.meshgrid(x_data, y_data)
hidden_pattern = np.sin(X) + np.cos(2*Y)

#defining dataset dimensions
N, D_in, D_out = 1000, 2, 1

#input data
x = torch.randn(N, D_in) * 3.1415
y = (x[:, 0].sin() + (2 * x[:, 1]).cos()).unsqueeze(1)

#noise
noise = torch.randn(N, D_out) * 0.2
y += noise

x_values = x.numpy()[:, 0]
y_values = x.numpy()[:, 1]
color_values = y.numpy().flatten()