# This file contains utility functions for this project.
# Author: Qidong Yang
# Date: 2022-07-21


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class DataAugmentor(object):
    def __init__(self, n_rotations=4, n_flips=3):
        super(DataAugmentor, self).__init__()

        self.n_rotations = n_rotations
        self.n_flips = n_flips

    def transform(self, x, y, n_transforms):

        rand1 = np.random.uniform(low=0.0, high=1.0)
        #rand2 = np.random.uniform(low=0.0, high=1.0)

        if n_transforms == 0:

            return x, y

        if 0 / n_transforms <= rand1 < 1 / n_transforms:
            k = 1
            x = torch.rot90(x, k, dims=(1, 2))
            y = torch.rot90(y, k, dims=(1, 2))

        if 1 / n_transforms <= rand1 < 2 / n_transforms:
            k = 2
            x = torch.rot90(x, k, dims=(1, 2))
            y = torch.rot90(y, k, dims=(1, 2))

        if 2 / n_transforms <= rand1 < 3 / n_transforms:
            k = 3
            x = torch.rot90(x, k, dims=(1, 2))
            y = torch.rot90(y, k, dims=(1, 2))

        if 3 / n_transforms <= rand1 < 4 / n_transforms:
            k = 4
            x = torch.rot90(x, k, dims=(1, 2))
            y = torch.rot90(y, k, dims=(1, 2))

        if 4 / n_transforms <= rand1 < 5 / n_transforms:
            x, y = torch.flip(x, (1,)), torch.flip(y, (1,))

        if 5 / n_transforms <= rand1 < 6 / n_transforms:
            x, y = torch.flip(x, (2,)), torch.flip(y, (2,))

        #if 0 / self.n_flips <= rand2 < 1 / self.n_flips:
        #    x, y = torch.flip(x, (1,)), torch.flip(y, (1,))

        #if 1 / self.n_flips <= rand2 < 2 / self.n_flips:
        #    x, y = torch.flip(x, (2,)), torch.flip(y, (2,))

        return x.contiguous(), y.contiguous()


class MaxMinNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(MaxMinNormalizer, self).__init__()

        # normalization using overall maximum and minmum

        self.max = torch.max(x)
        self.min = torch.min(x)
        self.eps = eps

    def encode(self, x):

        x = (x - self.min) / (self.max - self.min + self.eps)

        return x

    def decode(self, x):

        x = x * (self.max - self.min + self.eps) + self.min

        return x

    def cuda(self):

        self.max = self.max.cuda()
        self.min = self.min.cuda()

    def cpu(self):

        self.max = self.max.cpu()
        self.min = self.min.cpu()


class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        # Loss function with relative/absolute Lp loss (Used in Fourier Neural Operator paper)
        # d: the dimension of function domain
        # p: Lp-norm order
        # reduction: loss shape reduction or not
        # size_average: reduction by sum or average

        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):

        # compute absolute Lp loss

        num_examples = x.size()[0]
        # function mesh delta assume uniform mesh and uniform mesh delta
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d / self.p)) * torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):

        # compute relative Lp loss

        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x, y):

        # use relative Lp loss like the FNO paper

        # x,y (n_batch, n_dim1, n_dim2, n_channels)

        return self.rel(x, y)
