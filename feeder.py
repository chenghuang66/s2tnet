# sys
import os
import glob
import numpy as np
import pickle

# torch
import torch

from data_processor import GenerateData

class Feeder(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 data_cache,
                 train_percent=0.8,
                 train_val_test='train'):

        self.data_path = data_path
        self.data_cache = data_cache
        self.train_val_test = train_val_test

        self.load_data()

        total_num = len(self.all_data)
        # equally choose validation set
        train_id_list = list(np.linspace(0, total_num-1, int(total_num*train_percent)).astype(int))
        val_id_list = list(set(list(range(total_num))) - set(train_id_list))

        # # last 20% data as validation set
        if train_val_test.lower() == 'train':
            self.all_data = self.all_data[train_id_list]
        elif train_val_test.lower() == 'val':
            self.all_data = self.all_data[val_id_list]

    def load_data(self):

        if not (os.path.exists(self.data_cache)):
            
            train_file_path_list = sorted(
            glob.glob(os.path.join(self.data_path, 'prediction_train/*.txt')))
            print('Generating Training Data.')
            GenerateData(train_file_path_list,self.data_path, is_train=True)

            test_file_path_list = sorted(
                glob.glob(os.path.join(self.data_path, 'prediction_test/*.txt')))
            print('Generating Testing Data.')
            GenerateData(test_file_path_list,self.data_path, is_train=False)

            with open(self.data_cache, 'rb') as reader:
                # [self.all_data,self.mean,self.std] = pickle.load(reader)
                [self.all_data] = pickle.load(reader)

        else:
            with open(self.data_cache, 'rb') as reader:
                [self.all_data] = pickle.load(reader)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):

        data = self.all_data[idx].copy() 

        if self.train_val_test.lower() == 'train':
            th = np.random.random() * np.pi * 2
            data['features'][:, :, -2] = data['features'][:, :, -2] * np.cos(th) - data['features'][:, :, -1] * np.sin(th)
            data['features'][:, :, -1] = data['features'][:, :, -2] * np.sin(th) + data['features'][:, :, -1] * np.cos(th)
            data['mean'][0] = data['mean'][0] * np.cos(th) - data['mean'][1] * np.sin(th)
            data['mean'][1] = data['mean'][0] * np.sin(th) + data['mean'][1] * np.cos(th)
        
        if self.train_val_test.lower() == 'test':
            # return data['features'],data['masks'],data['mean'],data['origin'],self.mean,self.std
            return data['features'],data['masks'],data['mean'],data['origin'],data['neighbors']
        else:
            return data['features'],data['masks'],data['mean'],data['neighbors']