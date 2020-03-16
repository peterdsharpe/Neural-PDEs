import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import torch.optim as optim

x = torch.tensor([0.,1.,2.],requires_grad=True)
a = torch.tensor([1.], requires_grad=True)
b = torch.tensor([1.], requires_grad=True)
c = torch.tensor([1.], requires_grad=True)

def u(s):
    return a * s ** 2 + b * s + c


def jacobian(output_func, input):
    input.squeeze()
    n = input.size()[0]
    input = input.repeat(n, 1)
    input.requires_grad_(True)
    output = output_func(input)
    output.backward(torch.eye(n), retain_graph = True)
    # return input.grad.data
#
jacobian(u,x)