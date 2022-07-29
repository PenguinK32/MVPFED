# -*- coding: utf-8 -*-
# Python version: 3.6
# Author: penguink


import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, train_dataset=None, train_idxs=None, test_dataset=None, test_idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(train_dataset, train_idxs), batch_size=self.args.local_bs,
                                    shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(test_dataset, test_idxs), batch_size=self.args.local_bs,
                                    shuffle=True)

    def train_base_layers(self, net):
        """
        train the net
        :param net:
        :return: net params, base train loss
        """
        net.train()
        # train and update
        # optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr)

        # train base layers
        epoch_loss_base = []
        for p in net.parameters():
            p.requires_grad = True
        net.fc.weight.requires_grad = False
        net.fc.bias.requires_grad = False
        # net.layer4[1].requires_grad_(False)

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=self.args.lr,
                                    momentum=self.args.momentum)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device).squeeze().long()
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss_base.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss_base) / len(epoch_loss_base)

    def train_person_layers(self, net):
        # train personalized layers
        epoch_loss_person = []

        for p in net.parameters():
            p.requires_grad = False
        net.fc.weight.requires_grad = True
        net.fc.bias.requires_grad = True
        net.layer4[1].requires_grad_(True)

        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=self.args.lr,
                                    momentum=self.args.momentum)

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device).squeeze().long()
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss_person.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss_person) / len(epoch_loss_person)

    def eval(self, net):
        net.eval()
        correct = 0
        for batch_idx, (images, labels) in enumerate(self.ldr_test):
            images, labels = images.to(self.args.device), labels.to(self.args.device)

            log_probs = net(images)
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(labels.data.view_as(y_pred)).long().cpu().sum()

        accuracy = 100.00 * correct / len(self.ldr_test.dataset)

        return accuracy