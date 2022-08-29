import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from data_loader import get_loader
from trainer import Trainer

from model_loader import BaseModel
import random
import os
import numpy as np

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--lr', type=float, default=0.0001)

    p.add_argument('--model', type=str, default='3d')

    config = p.parse_args()

    return config


def get_model(config):
    if config.model == '3d':
        model = BaseModel()
    else:
        print("haha")

    return model

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main(config):
    # Set device based on user defined configuration.
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    train_loader, valid_loader, test_loader = get_loader(config)

    print("Test:", len(test_loader.dataset))

    seed_everything(41)
    model = get_model(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr = config.lr)
    crit = nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           verbose=True,
                                                           patience=7,
                                                           factor=0.5)

    if config.verbose >= 2:
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(config)
    trainer.train(model, crit, optimizer,  train_loader, valid_loader)

if __name__ == '__main__':
    config = define_argparser()
    main(config)