
from data import common

import imageio
import numpy as np

import torchvision.transforms as transforms
import torch.utils.data as data

class SRData(data.Dataset):
    def __init__(self, args, train=True):
        self.args = args
        self.train = train
        self.split = 'train' if train else 'test'
        self.scale = args.scale
        self.idx_scale = 0
        self.repeat = args.iter_epoch // (args.n_train // args.batch_size)
        self._set_filesystem(args.dir_data)
        def _scan():
            list_hr = []
            idx_begin = 0 if train else args.n_train
            idx_end = args.n_train if train else args.offset_val + args.n_val
            for i in range(idx_begin + 1, idx_end + 1):
                filename = self._make_filename(i)
                list_hr.append(self._name_hrfile(filename))

            return list_hr

        def _load():
            self.images_hr = np.load(self._name_hrbin())

        if args.ext == 'img':
            self.images_hr = _scan()
        elif args.ext.find('bin') >= 0:
            try:
                if args.ext.find('reset') >= 0:
                    raise IOError
                print('Loading a binary file')
                _load()
            except:
                print('Preparing a binary file')
                list_hr = _scan()
                hr = [imageio.imread(f) for f in list_hr]
                np.save(self._name_hrbin(), hr)
                del hr
                _load()
        else:
            print('Please define data type')

    def _set_filesystem(self, dir_data):
        raise NotImplementedError

    def _make_filename(self, idx):
        raise NotImplementedError

    def _name_hrfile(self, filename):
        raise NotImplementedError

    def _name_hrbin(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        img_hr = self._load_file(idx)
        img_hr = self._get_patch(img_hr)
        img_hr = common.set_channel(img_hr, self.args.n_colors)
        img_hr = common.np2Tensor(img_hr, self.args.rgb_range)

        return img_hr

    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        img_hr = self.images_hr[idx]
        if self.args.ext == 'img':
            img_hr = imageio.imread(img_hr)

        return img_hr

    def _get_patch(self, img_hr):
        patch_size = self.args.patch_size
        if self.train:
            img_hr = common.get_patch(img_hr, patch_size)
            img_hr = common.augment(img_hr)

        return img_hr
