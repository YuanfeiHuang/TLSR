import os
from data import SRData


class DIV2K(SRData.SRData):
    def __init__(self, args, train=True):
        super(DIV2K, self).__init__(args, train)

    def _set_filesystem(self, dir_data):
        self.apath = dir_data + 'Train/DIV2K'
        self.dir_hr = os.path.join(self.apath, 'DIV2K_HR')
        self.ext = '.png'

    def _make_filename(self, idx):
        return '{:0>4}'.format(idx)

    def _name_hrfile(self, filename):
        return os.path.join(self.dir_hr, filename + self.ext)

    def _name_hrbin(self):
        return os.path.join(self.apath, '{}_bin_HR.npy'.format(self.split))

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

