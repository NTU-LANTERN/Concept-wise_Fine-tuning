from .logger import CompleteLogger
from .meter import *
from .data import ForeverDataIterator
from PIL import Image
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
import torchvision.transforms as T
import vision.datasets as datasets

__all__ = ['metric', 'meter', 'data', 'logger']

def get_train_transform(resizing='default', random_horizontal_flip=True, random_color_jitter=False):

    if resizing == 'default':
        transform = T.RandomResizedCrop(224, scale=(0.2, 1.))
    elif resizing == 'core':
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.5, 1.)),
            # T.RandomRotation(15,),
            T.RandomHorizontalFlip(),
            # AutoAugImageNetPolicy(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    elif resizing == 'core_large':
            return T.Compose([
            T.Resize((512, 512), Image.BILINEAR),
            T.RandomCrop((448, 448)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    elif resizing == 'res.':
        transform = T.Resize((224, 224))
    elif resizing == 'res.crop':
        transform = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224)
        ])
    elif resizing == "res.sma|crop":
        transform = T.Compose([
            T.Resize(256),
            T.RandomCrop(224)
        ])
    elif resizing == 'inc.crop':
        transform = T.RandomResizedCrop(224)
    elif resizing == 'cif.crop':
        transform = T.Compose([
            T.Resize((224, 224)),
            T.Pad(16,padding_mode='symmetric'),
            T.RandomCrop(224),
        ])
    else:
        raise NotImplementedError(resizing)
    transforms = [transform]
    if random_horizontal_flip:
        transforms.append(T.RandomHorizontalFlip())
    if random_color_jitter:
        transforms.append(T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return T.Compose(transforms)

def get_val_transform(resizing='default'):
    if resizing == 'default':
        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
        ])
    elif resizing == 'res':
        transform = T.Resize((224, 224))
    elif resizing == 'res.|crop':
        transform = T.Compose([
            T.Resize((256, 256)),
            T.CenterCrop(224),
        ])
    else:
        raise NotImplementedError(resizing)
    return T.Compose([
        transform,
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_dataset(dataset_name, root, train_transform, val_transform, sample_rate=100, num_samples_per_classes=None):
    """
    When sample_rate < 100,  e.g. sample_rate = 50, use 50% data to train the model.
    Otherwise,
        if num_samples_per_classes is not None, e.g. 5, then sample 5 images for each class, and use them to train the model;
        otherwise, keep all the data.
    """

    dataset = datasets.__dict__[dataset_name]
    if sample_rate < 100:
        train_dataset = dataset(root=root, split='train', sample_rate=sample_rate, download=True, transform=train_transform)
        test_dataset = dataset(root=root, split='test', sample_rate=100, download=True, transform=val_transform)
        num_classes = train_dataset.num_classes
    else:
        train_dataset = dataset(root=root, split='train', download=True, transform=train_transform)
        test_dataset = dataset(root=root, split='test', download=True, transform=val_transform)
        num_classes = train_dataset.num_classes
        if num_samples_per_classes is not None:
            samples = list(range(len(train_dataset)))
            random.shuffle(samples)
            samples_len = min(num_samples_per_classes * num_classes, len(train_dataset))
            print("Origin dataset:", len(train_dataset), "Sampled dataset:", samples_len, "Ratio:", float(samples_len) / len(train_dataset))
            train_dataset = Subset(train_dataset, samples[:samples_len])
    return train_dataset, test_dataset, num_classes

class SinkhornDistance(nn.Module):
    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y, pos=True):
        device = x.device
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]
        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().cuda()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().cuda()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-3

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        # C = C.reshape(C.size(0),-1)
        batch_size = pi.size(0)

        if pos:
            topk = 3
            C = C.reshape(-1)
            pi = pi.reshape(batch_size, -1)
            leng = pi.size(1)
            _,b = torch.topk(pi, topk, dim=1)
            rand_pairs = torch.randint(topk,(batch_size,)).unsqueeze(1).to(device)
            rand_index = torch.gather(b, 1, rand_pairs).reshape(-1).cpu() + torch.arange(batch_size)*leng#.to(device)
            cost = C[rand_index.cuda()]
        else:
            topk = 10
            C = C.reshape(-1)
            pi = pi.reshape(batch_size, -1)
            leng = pi.size(1)
            _,b = torch.topk(pi, topk, dim=1) 
            shift = (torch.arange(batch_size)*leng).unsqueeze(1).repeat(1,topk)
            top_index = (b.cpu() + shift).reshape(-1)
            cost = C[top_index.cuda()]
        return cost

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        if x.size(0) == 1 and x.size(0) != y.size(0):
            x = x.repeat(y.size(0),1,1)
        C = 1.0-torch.bmm(x, y.transpose(1,2))
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1
    

def fix_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.momentum = 0.0
        
def recover_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.momentum = 0.1