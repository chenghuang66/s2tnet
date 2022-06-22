import argparse
import sys
import os
from arg_types import arg_boolean, arg_dict

dataset_name = 'apollo'
home_dir = os.getcwd()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_parser():
    # parameter priority: command line > config > default

    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    
    parser.add_argument('--work_dir', default=home_dir+'/checkpoints/' + dataset_name)
    parser.add_argument('--config', default=home_dir+'/config/apolloscape/train.yaml')
    parser.add_argument('--phase', default='train')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='the interval for printing messages (#iteration)')
    parser.add_argument('--eval_interval', type=int, default=5,
                        help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print_log', type=str2bool, default=True,
                        help='print logging or not')
    parser.add_argument('--test_model', type=int, default=5)
    parser.add_argument('--load_checkpt', type=str2bool, default=False)

    parser.add_argument(
        '--feeder', default='feeder.Feeder', help='data loader will be used')

    parser.add_argument('--num_worker', type=int, default=10)
    parser.add_argument('--train_data_path', default=' ')
    parser.add_argument('--test_data_path', default=' ')
    parser.add_argument('--train_data_cache', default=' ')
    parser.add_argument('--test_data_cache', default=' ')
    parser.add_argument('--train_percent', type=float, default=0.8)

    parser.add_argument(
        '--base_lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--step', type=int, default=[20, 40, 60], nargs='+',
                        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--optimizer', default='Adam',help='type of optimizer')
    parser.add_argument(
        '--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument(
        '--test_batch_size', type=int, default=256, help='test batch size')
    parser.add_argument(
        '--val_batch_size', type=int, default=256, help='value batch size')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start training from which epoch')
    parser.add_argument('--num_epoch', type=int, default=120,
                        help='stop training in which epoch')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='weight decay for optimizer')
    parser.add_argument('--ade', type=float, default=800.0)
    parser.add_argument('--factor', type=float, default=1.)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--predict_len', type=int, default=6)
    parser.add_argument('--history_len', type=int, default=6)

    parser.add_argument('--val_test', type=str2bool, default=False)
    return parser
