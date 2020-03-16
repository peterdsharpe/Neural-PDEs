import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Let's solve:
# du/dx + u = x
# On the domain [0, 1], with the BC u_func(0) = 1.

# Device
run_on_gpu = True
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
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)

        return x


# Define the function
u_func = Net().to(device)

# Solve ODE
optimizer = optim.Adam(u_func.parameters(), lr=0.003)
iters = 100
for iter in range(iters):
    # Define the collocation points
    x = torch.linspace(0, 1, 11).unsqueeze(1).to(device)
    x.requires_grad = True

    # Get u
    u = u_func(x)

    # Get dudx
    u.backward(torch.ones_like(x), create_graph=True)
    dudx = x.grad

    # Get residuals and loss
    domain_residual = dudx + u - x
    boundary_residual = u[0] - 1

    loss = (
            torch.mean(domain_residual ** 2) +
            torch.mean(boundary_residual ** 2)
    )
    optimizer.zero_grad()
    loss.backward()

    print("Iter %3.i: Loss = %f" % (iter, loss.detach()))
    optimizer.step()

import matplotlib.pyplot as plt
plt.plot(x.cpu().detach().numpy(), u.cpu().detach().numpy(),".-")
plt.show()