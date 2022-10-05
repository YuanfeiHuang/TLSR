import random

import tifffile
import torch

from data import common
import imageio
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import matplotlib.pyplot as plt

class DF2K(data.Dataset):
    def __init__(self, args):
        self.args = args
        self._set_filesystem(args.dir_data)

        def _scan():
            list_gt = []
            for i in range(0, self.args.n_train):
                list_gt.append(os.path.join(self.dir_gt, self.filename_gt[i]))
            return list_gt

        self.names_gt = _scan()
        if self.args.store_in_ram:
            self.images_gt = []
            for i in range(0, self.args.n_train):
                image_gt = imageio.imread(self.names_gt[i])
                self.images_gt.append(image_gt)
            print("load {}-images in memory".format(self.args.n_train))

    def _set_filesystem(self, dir_data):
        self.dir_gt = dir_data + 'Train/DF2K/DF2K_train_HR'

        self.filename_gt = os.listdir(self.dir_gt)
        self.filename_gt.sort()
        self.filename_gt = np.array(self.filename_gt)

        data_length = len(self.filename_gt)
        if self.args.n_train < data_length:
            if self.args.shuffle:
                idx = np.arange(0, data_length)
                idx = np.random.choice(idx, size=self.args.n_train)
            else:
                idx = np.arange(0, self.args.n_train)
            self.filename_gt = self.filename_gt[idx]

    def __getitem__(self, idx):

        if self.args.store_in_ram:
            idx = idx % len(self.images_gt)
            img_gt = self.images_gt[idx]
        else:
            idx = idx % len(self.names_gt)
            img_gt = imageio.imread(self.names_gt[idx])
            # if self.colors == 1:
            #     image_in, image_gt = sc.rgb2ycbcr(image_in)[:, :, 0:1], sc.rgb2ycbcr(image_gt)[:, :, 0:1]
        img_gt = common.set_channel(img_gt, self.args.n_colors)
        img_gt = self._get_patch(img_gt)
        img_gt = common.np2Tensor(img_gt, self.args.value_range)
        return img_gt

    def __len__(self):
        return self.args.iter_epoch * self.args.batch_size

    def _get_patch(self, img_gt):
        img_gt = common.get_patch(img_gt, self.args.patch_size, self.args.scale)
        img_gt = common.augment(img_gt)
        return img_gt