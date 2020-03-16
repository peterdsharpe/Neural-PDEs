import numpy as np
import torch
import torch.nn as nn

# Let's solve:
# du/dy + u * du/dx = 0.
# On the domain x=[0, 1], y=[0, 1], with Dirichlet zero boundary conditions on x=1 x=-1, Dirichlet u(x,0)=sin(2*pi*x)

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
        self.fc1 = nn.Linear(2, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 1)

    def forward(self, u_orig):
        u = torch.tanh(self.fc1(u_orig))
        u = torch.tanh(self.fc2(u))
        u = torch.tanh(self.fc3(u))
        u = self.fc4(u)
        u = u
        return u


# Define the function
u_func = Net().to(device)

# Solve ODE
optimizer = torch.optim.Adam(u_func.parameters())
# optimizer = torch.optim.Rprop(u_func.parameters())
# optimizer = torch.optim.LBFGS(u_func.parameters(), line_search_fn="strong_wolfe")
iters = 500
for iter in range(iters):


    def closure():


        global loss
        global X
        global Y
        global u
        global s
        global domain_residual
        global boundary_residual

        # Generate a domain
        x = torch.linspace(0, 1, 101, device=device)  # type: torch.Tensor
        y = torch.linspace(0, 1, 101, device=device)  # type: torch.Tensor
        Y, X = torch.meshgrid(y, x)  # type: torch.Tensor
        X = X.reshape(-1)
        Y = Y.reshape(-1)
        X.requires_grad = True
        Y.requires_grad = True
        inputs = torch.stack([X, Y], 1)

        # Get u
        u = u_func(inputs).squeeze()

        # Get dudx
        # try:
        #     X.grad.data.zero_()
        # except AttributeError:
        #     print('pass')
        #     pass
        del X.grad
        # X.grad = torch.zeros_like(X, device=device)
        # u_func.zero_grad()
        u.backward(torch.ones_like(X), create_graph=True)
        dudx = X.grad

        # Get dudx2
        # X.grad.data.zero_()
        del X.grad
        # X.grad = torch.zeros_like(X, device=device)
        # u_func.zero_grad()
        dudx.backward(torch.ones_like(X), create_graph=True)
        dudx2 = X.grad

        # Get dudy
        # try:
        #     Y.grad.data.zero_()
        # except AttributeError:
        #     print('pass')
        #     pass
        del Y.grad
        # Y.grad = torch.zeros_like(Y, device=device)
        # u_func.zero_grad()
        u.backward(torch.ones_like(Y), create_graph=True)
        dudy = Y.grad

        # Get dudy2
        # Y.grad.data.zero_()
        del Y.grad
        # Y.grad = torch.zeros_like(Y, device=device)
        # u_func.zero_grad()
        dudy.backward(torch.ones_like(Y), create_graph=True)
        dudy2 = Y.grad

        # Generate a boundary
        # s = torch.tensor([0.,1.],device=device).unsqueeze(1)
        s = torch.cat([
            torch.stack([
                torch.linspace(0, 1, 101),
                torch.linspace(0, 0, 101),
            ], 1),
            # torch.stack([
            #     torch.linspace(1, 1, 101),
            #     torch.linspace(0, 1, 101),
            # ], 1),
            torch.stack([
                torch.linspace(1, 0, 101),
                torch.linspace(1, 1, 101),
            ], 1),
            torch.stack([
                torch.linspace(0, 0, 101),
                torch.linspace(1, 0, 101),
            ], 1)
        ], 0).to(device)
        s.requires_grad = True

        # Get residuals
        domain_residual = dudy + u * dudx
        boundary_residual = u_func(s) - torch.sin(2*np.pi*s[:,0])

        # # Get residual derivative
        # x.grad = None
        # domain_residual.backward(torch.ones_like(x), create_graph=True)
        # domain_residual_grad = x.grad

        # Get loss
        loss = (
                torch.mean(domain_residual ** 2)
                # + torch.mean(domain_residual_grad ** 2)
                + torch.mean(boundary_residual ** 2)
        )
        # u_func.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        return loss


    optimizer.step(closure)
    print("Iter %3.i: Loss = %f" % (iter, loss))

import plotly.express as px
px.scatter_3d(
    x = X.cpu().detach().numpy(),
    y = Y.cpu().detach().numpy(),
    z = u.cpu().detach().numpy(),
    color = u.cpu().detach().numpy(),
).show()