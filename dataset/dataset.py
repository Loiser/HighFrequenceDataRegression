import pandas as pd
import torch
import torch.utils.data as data

import os
import random 

class dataset(data.Dataset):
    def __init__(self, dataroot):
        self.dataroot = dataroot

        # SP
        datas_sp = os.listdir(os.path.join(self.dataroot, 'SP'))
        self.SP = [os.path.join(self.dataroot,'SP', x) for x in datas_sp]

        # OEF
        datas_oef = os.listdir(os.path.join(self.dataroot, 'OEF'))
        self.OEF = [os.path.join(self.dataroot,'OEF', x) for x in datas_oef]
        
        self.data_size = len(self.SP)
        print('date_num:',self.data_size)


    def __getitem__(self, index):
        data_SP=pd.read_csv(self.SP[index]).values
        data_OEF=pd.read_csv(self.OEF[index])['V3'].values
        return data_SP, data_OEF

    def __len__(self):
        return self.data_size