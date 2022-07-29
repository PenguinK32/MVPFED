import torch
import os, glob
import random, csv
# 数据集基类 Dataset
from torch.utils.data import Dataset, DataLoader
# 图片变换工具
from torchvision import transforms
# 图片工具
from PIL import Image
# 展示图片
from matplotlib import pyplot as plt


class Chest_XRay(Dataset):

    def __init__(self, root="./", mode="train"):
        """
        :param root: 数据集根目录
        :param resize: 重塑图片大小
        :param mode: train/test ?
        """
        super(Chest_XRay, self).__init__()
        self.root = root
        self.resize = 64
        self.mode = mode

        # 定义一个Transform变换以后用
        self.transform = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),  # string path => 3 channel image data
            transforms.Resize((int(self.resize), int(self.resize))),
            transforms.RandomRotation(15),  # 随机旋转
            transforms.CenterCrop(self.resize),  # 中心裁剪
            transforms.ToTensor(),  # to tensor对象
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])  # 数值(mean.std)化
        ])

        if self.mode == "train":
            self.root = os.path.join(self.root, "train")
        elif self.mode == "test":
            self.root = os.path.join(self.root, "test")
        elif self.mode == "eval":
            self.root = os.path.join(self.root, "eval")
        elif self.mode == "eval_test":
            self.root = os.path.join(self.root, "eval_test")
        else:
            exit("not existing mode")

        self.name2label = {}
        # load the dirs as classes
        for name in sorted(os.listdir(os.path.join(self.root))):
            if not os.path.isdir(os.path.join(self.root, name)):  # if it is not a dir
                continue
            self.name2label[name] = len(self.name2label.keys())
        # print(self.name2label)
        self.class_num = len(self.name2label)

        # load images and labels from csv
        self.images, self.labels = self.load_csv('images.csv')

    def load_csv(self, filename):
        """
        将图片和它的label对应并存到csv表
        :param filename: 保存csv表格文件名
        :return: 返回(image, label)对
        """
        # 1. if do not have a csv, build one
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            # read pics
            for name in self.name2label.keys():
                # e.g. dataset\\class\\00001.png
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                # images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                # images += glob.glob(os.path.join(self.root, name, '*.jpeg'))

            # write pics info into the csv file
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images:
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])
                print('writen into csv file:', filename)

        # 2. read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:  # read rows
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)   # 保证两个长度一致

        return images, labels

    def __len__(self):
        """
        len
        :return:
        """
        return len(self.images)

    def __getitem__(self, idx):
        """
        根据idx得到(images,labels)对
        :param idx:
        :return: (images,labels)
        """
        img, label = self.images[idx], self.labels[idx]

        img = self.transform(img)
        label = torch.tensor(label)

        return img, label


if __name__ == '__main__':
    db = Chest_XRay('COVID-19_Radiography_Dataset', 'train')

    x, y = next(iter(db))  # 使用迭代器获取数据
    print(x.shape)
    plt.imshow(x.numpy().transpose(1, 2, 0))
    plt.show()
