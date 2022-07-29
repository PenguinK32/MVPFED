import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


class ResidualBlock(nn.Module):
    # 显式的继承自nn.Module
    # resnet是卷积的一种
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        # shortcut是直连，resnet和densenet的精髓所在
        # 层的定义都在初始化里
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outchannel)
        self.conv2 = nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outchannel)

        self.right = shortcut

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)

        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


class ResNet34(nn.Module):
    # 包括34，50，101等多种结构，可以按需实现，这里是Resnet34
    def __init__(self, num_classes=10):
        super(ResNet34, self).__init__()
        self.pre = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False),
                                 nn.BatchNorm2d(64),  # 这个64是指feature_num
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d(3, 2, 1))
        self.layer1 = self._make_layer(64, 128, 3)
        self.layer2 = self._make_layer(128, 256, 4, stride=2)
        self.layer3 = self._make_layer(256, 512, 6, stride=2)
        self.layer4 = self._make_layer(512, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        short_cut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, short_cut))
        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))  # 输入和输出要一致
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.pre(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)  # 注意F和原生的区别
        out = out.reshape(x.shape[0], -1)
        return self.fc(out)


from utils.options import args_parser
import copy

if __name__ == '__main__':
    model = ResNet34(num_classes=args_parser().num_classes)
    # print(model)
    w = copy.deepcopy(model.state_dict())
    for k in w.keys():
        # if "fc" not in str(k) and "layer4.1" not in str(k):
            print(k)
    # print(w['fc.weight'])

    for p in model.parameters():
        p.requires_grad = False
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True

    opt = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()
    images = torch.rand([4, 3, 32, 32])
    # one turn update test
    model.zero_grad()
    log_probs = model(images)
    loss = loss_func(log_probs, torch.tensor([0, 1, 2, 3], dtype=torch.long))
    loss.backward()
    opt.step()

    # view if fc changed
    # w = copy.deepcopy(model.state_dict())
    # print(w['fc.weight'])