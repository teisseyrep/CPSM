#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm


######################################
###########    DRO    ################
######################################


def fit_dro(Str_x, Str_y, Str_g, eta = 0.5, batch_size = 250,
            epochs = 500, lr = 1e-3, gamma = 0.99, 
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            ITER = 0):
    """
    Implementation for Hashimoto et al. "Fairness Without Demographics in Repeated Loss Minimization"
    
    Str_x: source training x
    Str_y: source training y
    Str_g: source training g
    eta: loss threshold
    batch_size: minibatch size
    epochs: epochs
    lr: learning rate
    gamma: exponential decay for learning rate (at each 100th epochs)
    device: cpu or cuda
    ITER: initioalization for the classifier
    """
    torch.manual_seed(ITER)
    np.random.seed(ITER)
    sigmoid = torch.nn.Sigmoid()
    
    class DRO(torch.nn.Module):
        def __init__(self, d):
            super(DRO, self).__init__()
            self.linear = torch.nn.Linear(d, 1)
    
        def forward(self, x):
            x = self.linear(x)
            x = torch.reshape(x, (-1, ))
            x = sigmoid(x)
            return x
    
    dro = DRO(d = Str_x.shape[1])
    dro = dro.to(device)
    
    
    t_Str_x = torch.from_numpy(Str_x).type(torch.float32).to(device)
    t_Str_y = torch.from_numpy(Str_y).type(torch.float32).to(device)
    
    dataset = TensorDataset(t_Str_x, t_Str_y)
    loader = DataLoader(dataset = dataset, batch_size = batch_size)
    optimizer = optim.Adam([{'params': dro.linear.parameters() ,'lr': lr}])
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    relu = torch.nn.ReLU()
    
    for epoch in range(1, epochs+1):
        print(f'epoch: {epoch}/{epochs}')
        with tqdm(loader, unit="batch") as tepoch:
            for batch_data in tepoch:
                # --- recollect source_x, source_y and target_x from batch data
                
                x_source, y_source, = batch_data 
                
                optimizer.zero_grad()
                p = dro(x_source) 
                cross_entropy = - y_source * torch.log(p) - (1 - y_source) * torch.log(1 - p)
                loss = torch.mean(relu(cross_entropy - eta) ** 2)
                loss.backward()
                optimizer.step()
                
                tepoch.set_postfix(loss=loss.item(), )
                
        if epoch % 100 == 0:
            p = dro(t_Str_x).detach().numpy()
            y_hat = p > 0.5
            for g in range(4):
                acc = (Str_y == y_hat)
                acc = (acc[Str_g == g]).mean()
                print(f'group {g} acc {acc}')
                
        if epoch % 100 == 0:
            scheduler.step()
                
    return dro.to(torch.device('cpu'))


######################################
###########    GDRO    ###############
######################################

def groups_(y, g, n_groups = 4):
    idx_g, idx_b = [], []
    all_g = y * n_groups + g

    for g in all_g.unique():
        idx_g.append(g)
        idx_b.append(all_g == g)

    return zip(idx_g, idx_b)

def compute_loss_value_(x, y, g, losses, q, eta = 0.1):
    
    
    for idx_g, idx_b in groups_(y, g):
        idx_g = idx_g.type(torch.int32)
        q[idx_g] *= (
            eta * losses[idx_b].mean()).exp().item()

    q /= q.sum()

    loss_value = 0
    for idx_g, idx_b in groups_(y, g):
        idx_g = idx_g.type(torch.int32)
        loss_value += q[idx_g] * losses[idx_b].mean()

    return loss_value, q


def fit_gdro(Str_x, Str_y, Str_g, eta = 0.5, batch_size = 250,
            epochs = 500, lr = 1e-3, gamma = 0.99, 
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            ITER = 0):
    
    """
    Implementation for Hashimoto et al. "Distributionally Robust Neural Networks
    for Group Shifts: On the Importance of Regularization for Worst-Case Generalization."
    
    Str_x: source training x
    Str_y: source training y
    Str_g: source training g
    eta: loss threshold
    batch_size: minibatch size
    epochs: epochs
    lr: learning rate
    gamma: exponential decay for learning rate (at each 100th epochs)
    device: cpu or cuda
    ITER: initioalization for the classifier
    """
    
    torch.manual_seed(ITER)
    np.random.seed(ITER)
    sigmoid = torch.nn.Sigmoid()
    
    class DRO(torch.nn.Module):
        def __init__(self, d):
            super(DRO, self).__init__()
            self.linear = torch.nn.Linear(d, 1)
    
        def forward(self, x):
            x = self.linear(x)
            x = torch.reshape(x, (-1, ))
            x = sigmoid(x)
            return x
    
    dro = DRO(d = Str_x.shape[1])
    dro = dro.to(device)
    
    n_classes = np.unique(Str_y).shape[0]
    n_groups = np.unique(Str_g).shape[0]
    
    t_Str_x = torch.from_numpy(Str_x).type(torch.float32).to(device)
    t_Str_y = torch.from_numpy(Str_y).type(torch.float32).to(device)
    t_Str_g = torch.from_numpy(Str_g).type(torch.int32).to(device)
    q = torch.ones(n_classes * n_groups).to(device)
    
    dataset = TensorDataset(t_Str_x, t_Str_y, t_Str_g)
    loader = DataLoader(dataset = dataset, batch_size = batch_size)
    optimizer = optim.Adam([{'params': dro.linear.parameters() ,'lr': lr}])
    scheduler = ExponentialLR(optimizer, gamma=gamma)
    
    for epoch in range(1, epochs+1):
        print(f'epoch: {epoch}/{epochs}')
        with tqdm(loader, unit="batch") as tepoch:
            for batch_data in tepoch:
                
                # --- recollect source_x, source_y and target_x from batch data
                
                x_source, y_source, g_source = batch_data 
                
                optimizer.zero_grad()
                p = dro(x_source) 
                cross_entropy = - y_source * torch.log(p) - (1 - y_source) * torch.log(1 - p)
                gdro_loss, _ = compute_loss_value_(x_source, y_source, g_source, cross_entropy, q, eta = eta)
                gdro_loss.backward()
                optimizer.step()
                
                tepoch.set_postfix(loss=gdro_loss.item(), )
                
        if epoch % 100 == 0:
            p = dro(t_Str_x).detach().numpy()
            y_hat = p > 0.5
            for g in range(4):
                acc = (Str_y == y_hat)
                acc = (acc[Str_g == g]).mean()
                print(f'group {g} acc {acc}')
                
        if epoch % 100 == 0:
            scheduler.step()
                
    return dro.to(torch.device('cpu')), q.to(torch.device('cpu')).detach().numpy()
