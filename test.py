from data_loader import get_loader, CustomDataset
import argparse
from model_loader import BaseModel
import torch
import tqdm
import h5py

from torch.utils.data import Dataset, DataLoader, random_split

import pandas as pd

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--lr', type=float, default=0.001)

    p.add_argument('--model', type=str, default='3d')

    config = p.parse_args()

    return config


def get_model(config):
    if config.model == '3d':
        model = BaseModel()
    else:
        print("haha")

    return model


def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)
    print(device)
    test_df = pd.read_csv("./sample_submission.csv")
    test_points = h5py.File("./test.h5", "r")
    test_dataset = CustomDataset(test_df['ID'].values, None, test_points)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    print("Test:", len(test_loader.dataset))

    model = get_model(config).to(device)
    d = torch.load("./base_model.pth", map_location = device)
    model.state_dict(d["model"])
    model.eval()

    model_preds = []
    with torch.no_grad():
        for data in tqdm.tqdm(test_loader):
            data = data.float().to(device)

            batch_pred = model(data)
            
            model_preds += batch_pred.argmax(1).detach().cpu().numpy().tolist()

    test_df["label"] = model_preds
    test_df.to_csv("./submit.csv", index = False)


if __name__ == '__main__':
    config = define_argparser()
    main(config)