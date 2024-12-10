import os
import torch
from torch.utils.data import Dataset
import augmentations
import pandas as pd
from pathlib import Path

class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    def register_args(self, **kwargs):
        
        self.device = None  #torch.device('cuda') if kwargs.get('cuda') else torch.device('cpu')
        self.index_l = []
        self.norm = 'global_normalize'
        self.complementary = 'per_voxel_normalize'
        self.random_TR = kwargs.get('random_TR')
        self.set_augmentations(**kwargs)
        self.stride_factor = 1
        self.sequence_stride = 1
        self.sequence_length = kwargs.get('sequence_length')
        self.sample_duration = self.sequence_length * self.sequence_stride
        self.stride = max(round(self.stride_factor * self.sample_duration), 1)
        self.TR_skips = range(0, self.sample_duration, self.sequence_stride)

    def get_input_shape(self):
        shape = torch.load(os.path.join(self.index_l[0][2], self.index_l[0][3] + '.pt')).squeeze().shape
        return shape

    def set_augmentations(self, **kwargs):
        if kwargs.get('augment_prob') > 0:
            self.augment = augmentations.brain_gaussian(**kwargs)
        else:
            self.augment = None

    def TR_string(self, filename_TR, x):
        # all datasets should have the TR mentioned in the format of 'some prefix _ number.pt'
        TR_num = [xx for xx in filename_TR.split('_') if xx.isdigit()][0]
        assert len(filename_TR.split('_')) == 4
        filename = filename_TR.replace(TR_num, str(int(TR_num) + x)) + '.pt'
        return filename

    def determine_TR(self, TRs_path, TR):
        if self.random_TR:
            possible_TRs = len(os.listdir(TRs_path)) - self.sample_duration
            TR = 'rfMRI_r_TR_' + str(torch.randint(0, possible_TRs, (1,)).item())
        return TR

    def load_sequence(self, TRs_path, TR):
        # the logic of this function is that always the first channel corresponds to global norm and if there is a
        # second channel it belongs to per voxel.
        TR = self.determine_TR(TRs_path, TR)
        y = torch.cat(
            [torch.load(os.path.join(TRs_path, self.TR_string(TR, x)), map_location=self.device).unsqueeze(0) for x in
             self.TR_skips], dim=4)
        if self.complementary is not None:
            y1 = torch.cat([torch.load(
                os.path.join(TRs_path, self.TR_string(TR, x)).replace(self.norm, self.complementary),
                map_location=self.device).unsqueeze(0)
                            for x in self.TR_skips], dim=4)
            y1[y1 != y1] = 0
            y = torch.cat([y, y1], dim=0)
            del y1
        return y
