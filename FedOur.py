# -*- coding: utf-8 -*-
# Python version: 3.6
# Author: penguink


import numpy as np
import copy
from torchvision import datasets, transforms
import torch

from utils.sampling import non_iid_chest
from utils.options import args_parser
from Run.FedOur_LocalUpdate import LocalUpdate
from models.Resnet34 import ResNet34
from models.Resnet18 import ResNet18
from Run.FedOur_Aggr import FedAggr
from utils.test import client_test

from dataset import Chest_XRay

from tqdm import tqdm


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load train_set and split users
    if args.train_set == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('./cifar-10', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('./cifar-10', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_train_users, dict_test_user = cifar_iid(dataset_train, dataset_test, args.num_users)
        else:
            print("dataset is non-i.i.d cifar-10")
            dict_train_users, dict_test_user = cifar_noniid(dataset_train, dataset_test, args.num_users)
    elif 'mnist' in str(args.train_set):  # load medmnist data
        data_flag = str(args.train_set)
        info = INFO[data_flag]
        args.num_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])
        # preprocessing
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])
        # load the data
        dataset_train = DataClass(split='train', transform=data_transform, download=True)
        dataset_test = DataClass(split='test', transform=data_transform, download=True)
        # non-iid
        dict_train_users, dict_test_user = non_iid(dataset_train, dataset_test, args.num_users)
    elif args.train_set == 'COVID':     # load COVID data
        dataset_train = Chest_XRay('COVID-19_Radiography_Dataset', 'train')
        dataset_test = Chest_XRay('COVID-19_Radiography_Dataset', 'test')
        # non-iid
        dict_train_users, dict_test_user = non_iid(dataset_train, dataset_test, args.num_users)
    else:
        exit('Error: unrecognized train_set')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'resnet34':
        net_glob = ResNet34(num_classes=args.num_classes).to(args.device)
        if args.model == 'resnet18':
        net_glob = ResNet18(num_classes=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()
    print("Aggregation over {} clients".format(int(args.frac * args.num_users)))
    w_locals = [w_glob for i in range(args.num_users)]

    # record params
    loss_trains = []  # average client test loss
    U_Accs = []  # average client test acc
    U_loss = []  # average client test loss

    # train
    # for each communication rounds
    for iter in range(args.epochs):
        print("communication rounds {}: ".format(iter))

        # chose user
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # idxs_users = range(args.num_users)

        # local record params
        loss_locals = []
        u_acc = []
        u_loss = []

        # for each clients
        for idx in tqdm(idxs_users):
            # LocalUpdate
            local = LocalUpdate(args=args, train_dataset=dataset_train, train_idxs=dict_train_users[idx],
                                test_dataset=dataset_test, test_idxs=dict_test_user[idx])
            # train and compute loss and acc
            # load model
            net_glob.load_state_dict(w_locals[idx])
            # train person layers, fix base layers
            person_w, person_loss = local.train_person_layers(net=copy.deepcopy(net_glob).to(args.device))
            # load base model
            net_glob.load_state_dict(person_w)
            # train base laysers, fix person layers
            base_w, base_loss = local.train_base_layers(net=copy.deepcopy(net_glob).to(args.device))

            # print('客户端{}, Acc: {:.2f}'.format(idx, u_acc.pop()))
            w_locals[idx] = copy.deepcopy(base_w)
            loss_locals.append(copy.deepcopy((base_loss + person_loss) / 2))

        # compute all UAcc
        for idx in range(args.num_users):
            # LocalUpdate
            net_glob.load_state_dict(w_locals[idx])
            uacc, loss = client_test(net=copy.deepcopy(net_glob).to(args.device), datatest=dataset_test, test_idxs=dict_test_user[idx], args=args)
            u_acc.append(uacc)
            u_loss.append(loss)
            # print('客户端{}, Acc: {:.2f} Loss: {:.2f}'.format(idx, u_acc[-1], u_loss[-1]))
        # U_ACCs
        U_Accs.append(sum(u_acc)/len(u_acc))
        U_loss.append(sum(u_loss)/len(u_loss))
        print('Clients average Testing accuracy: {:.2f}'.format(U_Accs[-1]))
        print('Clients average Testing loss: {:.2f}'.format(U_loss[-1]))

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Clients average Training Loss {:.3f}'.format(loss_avg))
        loss_trains.append(loss_avg)

        # Server
        # send base_w and update global weights
        w_glob, w_locals = FedAggr(w_locals, idxs_users)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # clean cuda cache
        torch.cuda.empty_cache()

    # save model
    torch.save(net_glob.state_dict(), './save/fedour_net_params.pth')

    # save record params
    np.savetxt("./save/fedour_{}round_{}train_avg_loss.txt".format(args.epochs, args.num_users), loss_trains)
    np.savetxt("./save/fedour_{}round_{}clients_avg_acc.txt".format(args.epochs, args.num_users), U_Accs)
    np.savetxt("./save/fedour_{}round_{}clients_avg_loss.txt".format(args.epochs, args.num_users), U_loss)
