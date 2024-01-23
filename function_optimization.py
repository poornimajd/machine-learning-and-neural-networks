import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch import optim
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D





def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def beale(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def ackley(x, y):
    return -20 * torch.exp(-0.2 * torch.sqrt(0.5 * (x**2 + y**2))) - torch.exp(0.5 * (torch.cos(2 * torch.pi * x) + torch.cos(2 * torch.pi * y))) + torch.exp(torch.tensor([1.0])) + 20


###############################################################################################
# Plot the objective function

# You will need to use Matplotlib's 3D plotting capabilities to plot the objective functions.
# Alternate plotting libraries are acceptable.
###############################################################################################

def plot_function(func, range_x, range_y, title):
    x = np.linspace(range_x[0], range_x[1], 400)
    y = np.linspace(range_y[0], range_y[1], 400)
    x, y = np.meshgrid(x, y)
    z = func(torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z.numpy(), cmap='viridis')
    ax.set_title(title)
    plt.show()

plot_function(rosenbrock, [-30, 30], [-30, 30], 'Rosenbrock Function')
plot_function(beale, [-30, 30], [-30, 30], 'Beale Function')
plot_function(ackley, [-30, 30], [-30, 30], 'Ackley Function')
plot_function(ackley, [-3, 3], [-3, 3], 'Ackley Function lesser range')


# SGD Optimization
def sgd_optimize_function(func):
    x = torch.tensor([10.0], requires_grad=True)
    y = torch.tensor([10.0], requires_grad=True)
    values = []
    optimizer_sgd = optim.SGD([x, y], lr=0.001)

    for _ in range(10000):
        optimizer_sgd.zero_grad()
        loss = func(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([x,y],max_norm=1.0)
        optimizer_sgd.step()
        values.append(loss.item())

    print(f"x: {x.item()}, y: {y.item()}, function value: {loss.item()}")
    return values

# ADAM Optimization
def adam_optimize_function(func):
    x = torch.tensor([10.0], requires_grad=True)
    y = torch.tensor([10.0], requires_grad=True)
    values = []
    optimizer_adam = optim.Adam([x, y], lr=0.001)
    # print("before",func(x,y).item())
    for _ in range(10000):
        optimizer_adam.zero_grad()
        loss = func(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([x,y],max_norm=1.0)
        optimizer_adam.step()
        values.append(loss.item())

    print(f"x: {x.item()}, y: {y.item()}, function value: {loss.item()}")

    return values


values_sgd_rosen = sgd_optimize_function(rosenbrock)
values_adam_rosen = adam_optimize_function(rosenbrock)
values_sgd_beale = sgd_optimize_function(beale)
values_adam_beale = adam_optimize_function(beale)
values_sgd_ackley = sgd_optimize_function(ackley)
values_adam_ackley = adam_optimize_function(ackley)



def plot_convergence(values_sgd, values_adam, title):
    plt.plot(values_sgd, label='SGD')
    plt.plot(values_adam, label='Adam')
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.title(title)
    plt.legend()
    plt.show()

plot_convergence(values_sgd_rosen, values_adam_rosen, 'Convergence Comparison (rosenbrock Function)')
plot_convergence(values_sgd_beale, values_adam_beale, 'Convergence Comparison (beale Function)')
plot_convergence(values_sgd_ackley, values_adam_ackley, 'Convergence Comparison (ackley Function)')
