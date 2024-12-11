import os
import numpy as np
import nibabel as nib
import torch
from multiprocessing import Process, Queue


def read_bnu(file_path, global_norm_path, per_voxel_norm_path, hand, count, queue=None):
    img_orig = torch.from_numpy(np.asanyarray(nib.load(file_path).dataobj)[8:-8, 8:-8, :-10, 10:]).to(
        dtype=torch.float32)
    background = img_orig == 0
    img_temp = (img_orig - img_orig[~background].mean()) / (img_orig[~background].std())
    img = torch.empty(img_orig.shape)
    img[background] = img_temp.min()
    img[~background] = img_temp[~background]
    img = torch.split(img, 1, 3)
    for i, TR in enumerate(img):
        torch.save(TR.clone(),
                   os.path.join(global_norm_path, 'rfMRI_' + hand + '_TR_' + str(i) + '.pt'))
    # repeat for per voxel normalization
    img_temp = (img_orig - img_orig.mean(dim=3, keepdims=True)) / (img_orig.std(dim=3, keepdims=True))
    img = torch.empty(img_orig.shape)
    img[background] = img_temp.min()
    img[~background] = img_temp[~background]
    img = torch.split(img, 1, 3)
    for i, TR in enumerate(img):
        torch.save(TR.clone(),
                   os.path.join(per_voxel_norm_path, 'rfMRI_' + hand + '_TR_' + str(i) + '.pt'))
    print('finished another subject. count is now {}'.format(count))

