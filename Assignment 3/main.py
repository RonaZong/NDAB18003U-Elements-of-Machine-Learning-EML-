from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from os import listdir

def normalize(arr):
    # Function to scale an input array to [-1, 1]
    arr_min = arr.min()
    arr_max = arr.max()
    # Check the original min and max values
    # print("Min: %.3f, Max: %.3f" % (arr_min, arr_max))
    arr_range = arr_max - arr_min
    scaled = np.array((arr-arr_min) / float(arr_range), dtype="f")
    arr_new = -1 + (scaled * 2)
    # Make sure min value is -1 and max value is 1
    # print("Min: %.3f, Max: %.3f" % (arr_new.min(), arr_new.max()))
    return arr_new

def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

# Remember to zero gradients before the backward call with a call to zero_grad()
def training_loop(n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        # In the backward() call on the loss tensor, gradients are accumulated in the params tensor (.grad)
        optimizer.zero_grad()
        loss.backward()
        # The optimizer can then use the gradients to compute new params with a call to step()
        optimizer.step()
        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
    return params

def loaddata(path):
    # Loop through all files in the directory
    for filename in listdir(path):
        # Load image
        image = Image.open(path + filename)
        # Convert to numpy array
        image = np.array(image)
        # Find number of channels
        if image.ndim == 2:
            channels = 1
            print(filename + " has 1 channel")
        else:
            channels = image.shape[-1]
            print(filename + " has", channels, "channels")
        # Scale to [-1,1]
        image = normalize(image)
        # Convert to Tensor
        image = transforms.ToTensor()(image)
    return image

# Paths to folder containing images
path_test = "./EML_exercise3_data/Images_test/"
path_train = "./EML_exercise3_data/Images_train/"
path_label = "./EML_exercise3_data/Labels_train/"

image_test = loaddata(path_test)
image_train = loaddata(path_train)
label_train = loaddata(path_label)

# Track gradients across the computation graph involving this tensor
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2

# Use Stochastic Gradient Descent (SGD)
# Disregard the term stochastic for now, the optimizer itself is just regular gradient descent (with the default setting of momentum = 0)
optimizer = optim.SGD([params], lr=learning_rate)
training_loop(
    n_epochs=5000,
    optimizer=optimizer,
    params=params,
    t_u=t_u,
    t_c=t_c)

learning_rate = 1e-1
optimizer = optim.Adam([params], lr=learning_rate) # <1>
training_loop2(
    n_epochs = 2000,
    optimizer = optimizer,
    params = params,
    t_u = t_u,
    t_c = t_c)

t_range = torch.arange(20., 90.).unsqueeze(1)

fig = plt.figure(dpi=600)
plt.xlabel("")
plt.ylabel("")
plt.plot(t_u.numpy(), t_c.numpy(), 'o')
plt.plot(t_range.numpy(), seq_model(0.1 * t_range).detach().numpy(), 'c-')
plt.plot(t_u.numpy(), seq_model(0.1 * t_u).detach().numpy(), 'kx')





