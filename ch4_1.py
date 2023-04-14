#!/usr/bin/env python
# coding: utf-8

# # Fashion MNIST 데이터셋 알아보기

from torchvision import datasets, transforms, utils
from torch.utils import data

import matplotlib.pyplot as plt
import numpy as np


# ## Fashion MNIST 데이터셋

transform = transforms.Compose([
    transforms.ToTensor()
])


trainset = datasets.FashionMNIST(
    root      = './.data/',
    train     = True,
    download  = True,
    transform = transform
)
testset = datasets.FashionMNIST(
    root      = './.data/',
    train     = False,
    download  = True,
    transform = transform
)


batch_size = 16

train_loader = data.DataLoader(
    dataset     = trainset,
    batch_size  = batch_size
)
test_loader = data.DataLoader(
    dataset     = testset,
    batch_size  = batch_size
)


dataiter       = iter(train_loader)
images, labels = next(dataiter)


# ## 멀리서 살펴보기
img   = utils.make_grid(images, padding=0)
npimg = img.numpy()
plt.figure(figsize=(10, 7))
plt.imshow(np.transpose(npimg, (1,2,0)))
plt.show()