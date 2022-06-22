#!/usr/bin/env python
import os
import time
import adamod
import yaml
import numpy as np
import zipfile

from tqdm import tqdm
from arg_parser import get_parser
from datetime import datetime
from arg_types import arg_boolean, arg_dict
from tensorboardX import SummaryWriter
from collections import OrderedDict
# torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import s2tnet
from feeder import Feeder
from utilies import *

localtime = time.asctime(time.localtime(time.time()))
x_writer = SummaryWriter('writer/'+ localtime)
test_result_file = 'prediction_result.txt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Processor():
    """
        Processor for s2tnet
    """

    def __init__(self, arg):
        self.arg = arg
        self.save_arg()
        self.load_data()
        self.load_model()
        self.load_optimizer()
        self.best_fde=5
        self.best_ade=1.25

        with open('{}/log.txt'.format(self.arg.work_dir), 'w') as f:
            print('start', file=f)

    def start(self):
        if self.arg.phase == 'train':
            if self.arg.load_checkpt:
                self.load_checkpoint(self.arg.test_model,self.arg.ade)

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                eval_model_flag = ((epoch + 1) % self.arg.eval_interval == 0) or (
                    epoch + 1 == self.arg.num_epoch)
                ######################Training##################
                if self.arg.val_test:
                    self.val_epoch(epoch)
                else:
                    self.train_epoch(epoch)
                    ######################valuing##################
                    if eval_model_flag or epoch > self.arg.eval_interval:                    
                        wsade,wsfde=self.val_epoch(epoch)
                        if wsade < self.best_ade:
                            self.best_ade = wsade 
                            self.best_epoch = epoch 
                            self.best_fde = wsfde
                            self.save_checkpoint(self.best_epoch,self.best_ade)

        if self.arg.phase == 'test':
            self.load_checkpoint(self.arg.test_model,self.arg.ade)
            self.test_epoch()


    def train_epoch(self, epoch):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch))
        loader = self.data_loader['train']
        lr = self.arg.base_lr
        if self.arg.optimizer is not 'NoamOpt':            
            lr = self.adjust_learning_rate(epoch)
        loss_value = []
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        # for batch_idx, (features,masks,mean_xy,neighbors) in enumerate(loader):
        for batch_idx, batch_data in enumerate(loader):
            features,masks,mean,neighbors = batch_data
            batch_in = features,masks,neighbors
            time_horizon = features.shape[1]
            timer['dataloader'] += self.split_time()
         
            for current_frame in range(2, time_horizon):
                predicted, _ = self.model(
                    batch_in,current_frame,device,is_train = True)

                mask=masks[:, current_frame:].to(device)
                ground_truth=features[:, current_frame:,:,-2:].to(device)
                predict_traj = predicted * mask
                ground_truth = ground_truth * mask
                # backward
                self.optimizer.zero_grad()

                error_order=1
                error=torch.abs(predict_traj - ground_truth) ** error_order
                error = error.sum(dim=3).sum(dim=2)
                overall_mask = mask.sum(dim=3).sum(dim=2)
                loss = error.sum() / torch.max(overall_mask.sum(), torch.ones(1,).to(device))
                loss.backward()

                # total_loss.backward()
                if self.arg.optimizer == 'NoamOpt':
                    self.optim.step()
                else:
                    self.optimizer.step()

                timer['model'] += self.split_time()
                loss_value.append(loss.data.item())

                # record log
                if batch_idx % self.arg.log_interval == 0:
                    # x_writer.add_graph(self.model,input_to_model=(input_data,origin_A,False))
                    step = epoch * len(loader) + batch_idx
                    x_writer.add_scalar('loss-Train', loss.data.item(), step)
                    if self.arg.optimizer == 'NoamOpt':
                        self.print_log(
                        '\t|Epoch:{:>5}/{:>5}|\tIteration:{:>5}/{:>5}|\tLoss:{:.5f}|lr: {:.4f}|'.format(
                        epoch, self.arg.num_epoch,batch_idx, len(loader), loss.data.item(), self.optim._rate))
                    else:
                        self.print_log(
                        '\t|Epoch:{:>5}/{:>5}|\tIteration:{:>5}/{:>5}|\tLoss:{:.5f}|lr: {:.4f}|'.format(
                        epoch, self.arg.num_epoch,batch_idx, len(loader), loss.data.item(), lr))  
    
    @torch.no_grad()
    def val_epoch(self, epoch):
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch))

        loader = self.data_loader['val']

        sum_list = []
        number_list = []
        h_len = self.arg.history_len
        for batch_data in loader:
            features,masks,mean,neighbors = batch_data
            batch_in = features,masks,neighbors
            b,t,v,c = features.shape
            decoder_input = torch.zeros((b, 1, v, 2)).to(device) 
            for i in range(h_len):
                
                predicted,att = self.model(
                    batch_in, h_len, device, decoder_input, False)
                
                decoder_input = torch.cat((decoder_input, predicted[:, -1:]), 1)
            
            predicted_xy=decoder_input[:, 1:].cumsum(1)
            predicted_trajectory = predicted_xy + features[:,h_len-1:h_len, :, :2].to(device)
            ground_truth = features[:, h_len:,:,:2].to(device) * masks[:, h_len:].to(device)
            predict_traj = predicted_trajectory * masks[:, h_len:].to(device)

            error_order=2
            error=torch.abs(predict_traj - ground_truth) ** error_order
            error = error.sum(dim=3).sum(dim=2)
            overall_mask = masks[:, h_len:].sum(dim=3).sum(dim=2)

            number_list.extend(overall_mask.detach().cpu().numpy())
            sum_list.extend(error.detach().cpu().numpy())

        sum_time = np.sum(np.array(sum_list)**0.5, axis=0)
        num_time = np.sum(np.array(number_list), axis=0)
        overall_loss_time = (sum_time / num_time)
        overall_log = '[{:>15}] [FDE: {:.3f}] [ADE: {:.3f}] [best_FDE: {:.3f}] [best_ADE: {:.3f}] 7--12s: {}'.format(
            'Unweighted Sum', overall_loss_time[-1],np.mean(overall_loss_time),self.best_fde,self.best_ade,
            ' '.join(['{:.3f}'.format(x) for x in list(overall_loss_time) + [np.sum(overall_loss_time)]]))

        self.print_log(overall_log)

        WSADE=np.mean(overall_loss_time)
        WSFDE=overall_loss_time[-1]
        
        info = {
                'Ade': WSADE,
                'Fde': WSFDE
                }
        for tag, value in info.items():
            x_writer.add_scalar(tag, value, epoch)

        return WSADE,WSFDE

    def test_epoch(self):
        self.model.eval()

        with open(test_result_file, 'w') as writer:
            loader = self.data_loader['test']
            h_len = self.arg.history_len
            for batch_data in tqdm(loader):                
                features,masks,mean,origin,neighbors = batch_data
                batch_in = features,masks,neighbors
                b,t,v,c = features.shape
                decoder_input = torch.zeros((b, 1, v, 2)).to(device) 
                for i in range(h_len):

                    predicted,att = self.model(
                        batch_in, h_len, device, decoder_input, False)
                    
                    decoder_input = torch.cat((decoder_input, predicted[:, -1:]), 1)

                predicted_xy=decoder_input[:, 1:].cumsum(1) 
                predicted_trajectory = predicted_xy + features[:,h_len-1:h_len, :, :2].to(device)

                now_pred = predicted_trajectory.detach().cpu().numpy()
                now_mean_xy = mean.detach().cpu().numpy()
                now_mask = masks[:, -1].detach().cpu().numpy()
                origin=origin.detach().cpu().numpy()
                # batch
                for n_pred, n_mean_xy, n_data, n_mask in zip(now_pred, now_mean_xy, origin, now_mask):                
                    # time
                    for time_ind, n_pre in enumerate(n_pred):
                        #nodes 
                        for info, pred, mask in zip(n_data[-1], n_pre+n_mean_xy, n_mask):
                            if mask:
                                information = info.copy()
                                information[0] = information[0] + time_ind + 1
                                result = ' '.join(information.astype(str)) \
                                        + ' ' + ' '.join(pred.astype(str)) + '\n'
                                writer.write(result)

        with zipfile.ZipFile('prediction_result.zip', mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(test_result_file)

    def save_checkpoint(self, epoch,ade):        
        filename='epoch_{:04}_{:06.00f}.pt'.format(epoch,ade*10000)
        try:
            torch.save({'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict()
                        }, os.path.join(self.arg.work_dir, filename))

        except Exception as e:
            print("An error occurred while saving the checkpoint:")
            print(e)

    def load_checkpoint(self, best_epoch,ade):
        filename='epoch_{:04}_{:06.00f}.pt'.format(best_epoch,ade)
        # filename='epoch_{:04}_{:04.01f}.pt'.format(best_epoch,ade)
        ckpt_path = os.path.join(self.arg.work_dir, filename)

        checkpoint = torch.load(ckpt_path)
        self.arg.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Successfull loaded from {}'.format(ckpt_path))

    def load_data(self):

        self.data_loader = dict()
        self.trainLoader = Feeder(
            self.arg.train_data_path,self.arg.train_data_cache,self.arg.train_percent,'train')
        self.testLoader = Feeder(
            self.arg.test_data_path,self.arg.test_data_cache,self.arg.train_percent,'test')
        self.valLoader = Feeder(
            self.arg.train_data_path,self.arg.train_data_cache,self.arg.train_percent,'val')

        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=self.trainLoader,
            batch_size=self.arg.batch_size,
            shuffle=True,
            num_workers=self.arg.num_worker)
        self.data_loader['val'] = torch.utils.data.DataLoader(
            dataset=self.valLoader,
            batch_size=self.arg.val_batch_size,
            shuffle=True,
            num_workers=self.arg.num_worker)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.testLoader,
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker)

    def load_model(self):

        self.model = s2tnet(d_model=self.arg.d_model)
        self.model = nn.DataParallel(self.model)
        self.model.to(device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adamod':
            self.optimizer = adamod.AdaMod(
                self.model.parameters(), lr=self.arg.base_lr, beta3=0.999)
        elif self.arg.optimizer == 'NoamOpt':
            self.optim = NoamOpt(self.arg.d_model, self.arg.factor, self.arg.warmup, 
                torch.optim.Adam(self.model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
            self.optimizer=self.optim.optimizer
        else:
            raise ValueError()

    # save all arg in work directory with yaml format
    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        lr = self.arg.base_lr

        step = self.arg.step
        lr = self.arg.base_lr * (
            self.arg.base_lr ** np.sum(epoch >= np.array(step)))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    # print log in screen and save in log.txt
    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    # save current time
    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time
    # get time interval from last record time to now

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def display_result(self, pra_results, predict_result_lable='Train_epoch'):
        sum_list, num_list = pra_results
        sum_time = np.sum(sum_list**0.5, axis=0)
        num_time = np.sum(num_list, axis=0)
        overall_loss_time = (sum_time / num_time)
        overall_log = '[{:>15}] [FDE: {:.3f}] [ADE: {:.3f}] 7--12s: {}'.format(
            predict_result_lable, overall_loss_time[-1],
            np.mean(overall_loss_time),
            ' '.join(['{:.3f}'.format(x) for x in list(overall_loss_time) + [np.sum(overall_loss_time)]]))

        self.print_log(overall_log)
        return overall_loss_time


seed_torch()

if __name__ == '__main__':

    parser = get_parser()
    p = parser.parse_args()
    print(p.config)
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    processor = Processor(arg)
    processor.start()
