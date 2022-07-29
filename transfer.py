# -*- coding: utf-8 -*-
# Python version: 3.6
# Author: penguink

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

from utils.options import args_parser
from models.Resnet34 import ResNet34
from dataset import Chest_XRay

import numpy as np


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(args.seed)

    # load train_set and split users
    train_db = Chest_XRay('COVID-19_Radiography_Dataset', 'eval')
    test_db = Chest_XRay('COVID-19_Radiography_Dataset', 'eval_test')

    # build model
    net_glob = ResNet34(args.num_classes).to(args.device)

    # load params
    net_glob.load_state_dict(torch.load('./save/fedour_net_params.pth'))

    # TODO: fix the params
    net_glob.fc.weight.requires_grad = False
    net_glob.fc.bias.requires_grad = False
    net_glob.layer4[1].requires_grad_(False)
    
    # training
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net_glob.parameters()), lr=args.lr,
                                momentum=0.2)
    train_loader = DataLoader(train_db, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_db, batch_size=64, shuffle=True)

    list_train_loss = []
    list_test_loss = []
    list_test_acc = []
    for epoch in range(10):
        # train
        net_glob.train()
        batch_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = net_glob(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 5 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
            batch_loss.append(loss.item())
        loss_avg = sum(batch_loss)/len(batch_loss)
        print('Train loss: {:.3f} '.format(loss_avg))
        list_train_loss.append(loss_avg)

        # eval
        test_loss = 0
        correct = 0
        net_glob.eval()
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(args.device), target.to(args.device)
            log_probs = net_glob(data)
            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        test_loss /= len(test_loader.dataset)
        correct = 100.00 * correct / len(test_loader.dataset)
        list_test_loss.append(test_loss)
        list_test_acc.append(correct)
        print('Test loss: {:.3f} '.format(test_loss))
        print('Test acc: {:.2f} % '.format(correct))

    np.savetxt("./save/fedour_transfer_loss", list_test_loss)
    np.savetxt("./save/fedour_transfer_acc", list_test_acc)