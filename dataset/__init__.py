#!/usr/bin/env python
# coding: utf-8

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler


def get_dataloader(dataroot):
    
    from .dataset import dataset

    dataset = dataset(dataroot)
    samplerC = SequentialSampler(dataset)
    loader = DataLoader(dataset, 
                        num_workers=1,
                        pin_memory=True,
                        shuffle=False,
                        sampler=samplerC)

    return loader