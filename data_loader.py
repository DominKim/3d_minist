import torch

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
import h5py

from sklearn.model_selection import train_test_split

from PIL import Image
import numpy as np

import pandas as pd
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class CustomDataset(Dataset):
    def __init__(self, id_list, label_list, point_list):
        self.id_list = id_list
        self.label_list = label_list
        self.point_list = point_list

    def __getitem__(self, index):
        image_id = self.id_list[index]

        # h5파일을 바로 접근하여 사용하면 학습 속도가 병목 현상으로 많이 느릴 수 있습니다.
        points = self.point_list[str(image_id)][:]
        image = self.get_vector(points)

        if self.label_list is not None:
            label = self.label_list[index]
            return torch.Tensor(image).unsqueeze(0), label
        else:
            return torch.Tensor(image).unsqueeze(0)

    def get_vector(self, points, x_y_z=[16, 16, 16]):
        # 3D Points -> [16,16,16]
        xyzmin = np.min(points, axis=0) - 0.001
        xyzmax = np.max(points, axis=0) + 0.001

        diff = max(xyzmax - xyzmin) - (xyzmax - xyzmin)
        xyzmin = xyzmin - diff / 2
        xyzmax = xyzmax + diff / 2

        segments = []
        shape = []

        for i in range(3):
            # note the +1 in num
            if type(x_y_z[i]) is not int:
                raise TypeError("x_y_z[{}] must be int".format(i))
            s, step = np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i] + 1), retstep=True)
            segments.append(s)
            shape.append(step)

        n_voxels = x_y_z[0] * x_y_z[1] * x_y_z[2]
        n_x = x_y_z[0]
        n_y = x_y_z[1]
        n_z = x_y_z[2]

        structure = np.zeros((len(points), 4), dtype=int)
        structure[:, 0] = np.searchsorted(segments[0], points[:, 0]) - 1
        structure[:, 1] = np.searchsorted(segments[1], points[:, 1]) - 1
        structure[:, 2] = np.searchsorted(segments[2], points[:, 2]) - 1

        # i = ((y * n_x) + x) + (z * (n_x * n_y))
        structure[:, 3] = ((structure[:, 1] * n_x) + structure[:, 0]) + (structure[:, 2] * (n_x * n_y))

        vector = np.zeros(n_voxels)
        count = np.bincount(structure[:, 3])
        vector[:len(count)] = count

        vector = vector.reshape(n_z, n_y, n_x)
        return vector

    def __len__(self):
        return len(self.id_list)

def collate_fn(batch: torch.Tensor):
    return tuple(zip(*batch))


def get_loader(config):
    all_df = pd.read_csv('./train.csv')
    all_points = h5py.File('./train.h5', 'r')
    test_df = pd.read_csv('./sample_submission.csv')
    test_points = h5py.File('./test.h5', 'r')

    if config.train_ratio == 1.0:
        train_loader = DataLoader(dataset=CustomDataset(all_df, all_points), batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=8, collate_fn=collate_fn)
        return train_loader

    if config.train_ratio < 1.0:
        train_df = all_df.iloc[:int(len(all_df) * config.train_ratio)]
        val_df = all_df.iloc[int(len(all_df) * config.train_ratio):]

        train_loader = DataLoader(dataset=CustomDataset(train_df['ID'].values, train_df['label'].values, all_points), batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=0, collate_fn=collate_fn)
        valid_loader = DataLoader(dataset=CustomDataset(val_df['ID'].values, val_df['label'].values, all_points), batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=0, collate_fn=collate_fn)
        test_loader = DataLoader(dataset=CustomDataset(test_df['ID'].values, None, test_points),
                                 batch_size = config.batch_size, shuffle = False, num_workers=4, collate_fn=collate_fn)

        return train_loader, valid_loader, test_loader