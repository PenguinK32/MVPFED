# -*- coding: utf-8 -*-
# Python version: 3.6
# Author: penguink


import numpy as np
from torchvision import datasets, transforms

from dataset import Chest_XRay


def non_iid(train_dataset, test_dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST train_set
    :param dataset:
    :param num_users:
    :return: dict of non-iid image index
    """
    num_shards, num_train_imgs, num_test_imgs = int(num_users * 2), \
                                                int(len(train_dataset) / (num_users * 2)), \
                                                int(len(test_dataset) / (num_users * 2))

    idx_shard = [i for i in range(num_shards)]
    user_train_dict = {i: np.array([], dtype='int64') for i in range(num_users)}
    train_idxs = np.arange(len(list(train_dataset.labels)))
    train_labels = train_dataset.labels
    user_test_dict = {i: np.array([], dtype='int64') for i in range(num_users)}
    test_idxs = np.arange(len(list(test_dataset.labels)))
    test_labels = test_dataset.labels

    # sort labels, build idx tuple
    # train
    train_idxs_tuple = np.vstack((train_idxs, train_labels))  # ([数据idx],[标签])
    train_idxs_tuple = train_idxs_tuple[:, train_idxs_tuple[1, :].argsort()]  # 按标签排序
    # print(train_idxs_tuple[1][9999:10001])
    train_idxs = train_idxs_tuple[0, :]
    # test
    test_idxs_tuple = np.vstack((test_idxs, test_labels))  # ([数据idx],[标签])
    test_idxs_tuple = test_idxs_tuple[:, test_idxs_tuple[1, :].argsort()]  # 按标签排序
    # print(test_idxs_tuple[1][1999:2001])
    test_idxs = test_idxs_tuple[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        # print("客户端{} 选中了组：{}".format(i, rand_set))
        for rand in rand_set:
            user_train_dict[i] = np.concatenate(
                (user_train_dict[i], train_idxs[rand * num_train_imgs:(rand + 1) * num_train_imgs]), axis=0)
            user_test_dict[i] = np.concatenate(
                (user_test_dict[i], test_idxs[rand * num_test_imgs:(rand + 1) * num_test_imgs]), axis=0)

    return user_train_dict, user_test_dict


if __name__ == '__main__':
    dataset_train = Chest_XRay('../COVID-19_Radiography_Dataset', 'train')
    dataset_test = Chest_XRay('../COVID-19_Radiography_Dataset', 'test')

    d_train, d_test = non_iid(dataset_train, dataset_test, 10)
    print(d_train[0], d_test[0])
    print(d_train[4], d_test[4])
    print(d_train[5], d_test[5])
    print(d_train[9], d_test[9])
    print(len(d_train[0]), len(d_test[0]))