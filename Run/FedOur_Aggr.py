# -*- coding: utf-8 -*-
# Python version: 3.6
# Author: penguink


import copy
import torch
from torch import nn


def FedAggr(w_per, users):
    # Server FedAvg the global base layers
    w_avg = copy.deepcopy(w_per[users[0]])
    for k in w_avg.keys():
        if "fc" not in str(k) and "layer4.1" not in str(k):
            for i in users[1:]:
                w_avg[k] += w_per[i][k]
            w_avg[k] = torch.true_divide(w_avg[k], len(users))

    # server send base_w, copy trained glob_w to person_w
    for i in users:
        for k in w_per[i].keys():
            if "fc" not in str(k) and "layer4.1" not in str(k):  # copy the base laysers
                # print(k)
                w_per[i][k] = copy.deepcopy(w_avg[k])

    return w_avg, w_per