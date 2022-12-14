from data_loader import get_loader, CustomDataset
import argparse
from model_loader import BaseModel, PointNetCls
import torch
from tqdm.auto import tqdm
import h5py
from data_loader import PointCloudDataset

from torch.utils.data import Dataset, DataLoader, random_split

import pandas as pd
import os
import random
import numpy as np

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)

    p.add_argument('--model', type=str, default='3d')

    config = p.parse_args()

    return config

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_model(config):
    if config.model == '3d':
        model = BaseModel()
    elif config.model == "pointnet":
        model = PointNetCls()
    else:
        print("haha")

    return model


def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)
    print(device)
    test_df = pd.read_csv('./sample_submission.csv')
    test_points = h5py.File('./test.h5', 'r')
    test_points = [np.array(test_points[str(i)]) for i in tqdm(test_df["ID"])]
    test_dataset = PointCloudDataset(test_df['ID'].values, None, test_points, 8600, 'test')
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    print("Test:", len(test_loader.dataset))
    seed_everything(41)

    model = get_model(config).to(device)
    d = torch.load(config.model_fn)
    model.load_state_dict(d["model"])
    model.eval()


    model_preds = []
    with torch.no_grad():
        for data in tqdm(iter(test_loader)):
            data = data.float().to(device)

            batch_pred, trans_feat = model(data)
            
            model_preds += batch_pred.argmax(1).detach().cpu().numpy().tolist()

    test_df = pd.read_csv("./sample_submission.csv")
    test_df["label"] = model_preds
    test_df.to_csv("./submit.csv", index = False)


if __name__ == '__main__':
    config = define_argparser()
    main(config)