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


def test(net_g, data_loader):
    # testing
    net_g.eval()
    test_loss = 0
    correct = 0
    l = len(data_loader)
    for idx, (data, target) in enumerate(data_loader):
        data, target = data.to(args.device), target.to(args.device)
        log_probs = net_g(data)
        test_loss += F.cross_entropy(log_probs, target).item()
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    return correct, test_loss


if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    torch.manual_seed(args.seed)

    # load train_set and split users
    train_db = Chest_XRay('COVID-19_Radiography_Dataset', 'train')
    test_db = Chest_XRay('COVID-19_Radiography_Dataset', 'test')

    # build model
    net_glob = ResNet34(args.num_classes).to(args.device)

    # training
    optimizer = optim.SGD(net_glob.parameters(), lr=0.001, momentum=0.5)
    train_loader = DataLoader(train_db, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_db, batch_size=128, shuffle=True)

    list_loss = []
    for epoch in range(10):
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
        print('\nTrain loss:', loss_avg)
        list_loss.append(loss_avg)

        # eval
        test_acc, test_loss = test(net_glob, test_loader)

    # save model
    torch.save(net_glob.state_dict(), './save/glob_net_params.pth')
