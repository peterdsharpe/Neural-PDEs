import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Let's solve:
# -d^2(u)/dx^2 = sin(2 * pi * x)
# On the domain [0, 1], with the BC u_func(0) = 0, u_func(1) = 0.

# Device
run_on_gpu = False
if run_on_gpu:
    assert torch.cuda.is_available(), "No CUDA-enabled GPU found!"
    device = torch.device("cuda")
    print("Running on GPU!")
else:
    device = torch.device("cpu")
    print("Running on CPU!")


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x_orig):
        x = torch.tanh(self.fc1(x_orig))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc4(x)
        x = x
        return x


# Define the function
u_func = Net().to(device)

# Solve ODE
# optimizer = torch.optim.Adam(u_func.parameters())
# optimizer = torch.optim.Rprop(u_func.parameters())
optimizer = torch.optim.LBFGS(u_func.parameters(), line_search_fn="strong_wolfe")
iters = 10
for iter in range(iters):
    loss = 0.
    def closure():
        # Generate a domain
        x = torch.linspace(0, 1, 10001, device=device).unsqueeze(1)
        # x = torch.rand(10001, device=device).unsqueeze(1)
        x.requires_grad=True

        # Get u
        u = u_func(x)

        # Get dudx
        x.grad = None
        u.backward(torch.ones_like(x), create_graph=True)
        dudx = x.grad

        # Get d2udx2
        x.grad = None
        dudx.backward(torch.ones_like(x), create_graph=True)
        d2udx2 = x.grad

        # Generate a boundary
        s = torch.tensor([0.,1.],device=device).unsqueeze(1)
        s.requires_grad=True

        # Get residuals
        domain_residual = -d2udx2 - torch.sin(2*np.pi*x)
        boundary_residual = u_func(s) - 0

        # Get residual derivative
        x.grad=None
        domain_residual.backward(torch.ones_like(x), create_graph=True)
        domain_residual_grad = x.grad

        # Get loss
        global loss
        loss = (
                torch.mean(domain_residual ** 2)
                + torch.mean(domain_residual_grad ** 2)
                + 2 * torch.mean(boundary_residual ** 2)
        )
        optimizer.zero_grad()
        loss.backward()
        return loss

    optimizer.step(closure)
    print("Iter %3.i: Loss = %f" % (iter, loss))

import matplotlib.pyplot as plt
x = torch.linspace(0,1,1001, device=device).unsqueeze(1)
u = u_func(x)
plt.plot(x.cpu().detach().numpy(), u.cpu().detach().numpy(), "-")
plt.show()
