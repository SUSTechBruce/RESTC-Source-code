from os import replace
import time
import argparse
import pickle
import numpy as np
from torch.utils.data.dataset import Dataset
from GAT import DualContrastiveModel, train_and_test
from utils import split_validation, Data, sample_adj, trans_to_cuda
import torch
import logging
import warnings
warnings.filterwarnings("ignore")


logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)
import wandb


def init_seed(seed=None):
    if seed == None:
        seed = int(time.time() * 1000//1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='diginetica', help='diginetica/Nowplaying/Tmall')
    parser.add_argument('--hiddenSize', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--activate', type=str, default='relu')
    parser.add_argument('--n_sample_all', type=int, default=12) # initial value is 12
    parser.add_argument('--n_sample', type=int, default=12) # initial value is 12
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_dc_step', type=float, default=3, help='the number of step after the learning rate decay')
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay')
    parser.add_argument('--l2', type=float, default=1e-5, help='the L2 pernalty')
    parser.add_argument('--n_iter', type=int, default=2)
    parser.add_argument('--dropout_gcn', type=float, default=0.2)
    parser.add_argument('--dropout_local', type=float, default=0.0)
    parser.add_argument('--dropout_global', type=float, default=0.5)
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for leaky_relu')
    parser.add_argument('--patience', type=int, default=10)

    opt = parser.parse_args()
    return opt

def train_and_valid(args):
    init_seed(2020)
    if args.dataset == 'diginetica':
        num_nodes = 43098
        args.n_iter = 2
        args.dropout_gcn = 0.2
        args.dropout_local = 0.0
    elif args.dataset == 'Tmall':
        num_nodes = 40728
        args.n_iter = 1
        args.dropout_gcn = 0.6
        args.dropout_local = 0.5
    elif args.dataset == 'Nowplaying':
        num_nodes = 60417
        args.n_iter = 1
        args.dropout_gcn = 0.0
        args.dropout_local = 0.0
    else:
        num_nodes = 310
    
    train_data = pickle.load(open('Dual_contrastive/datasets/'+ args.dataset +'/train.txt', 'rb'))
    if args.validation:
        print('Split the validation set from train set')
        train_data, valid_data = split_validation(train_data, args.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('Dual_contrastive/datasets/'+ args.dataset +'/test.txt', 'rb'))
    
    # print('train data [1][775]', train_data[0][775])
    adj = pickle.load(open('Dual_contrastive/datasets/'+ args.dataset +'/adj_'+ str(args.n_sample_all)+'.pkl', 'rb'))
    num = pickle.load(open('Dual_contrastive/datasets/'+ args.dataset +'/num_'+ str(args.n_sample_all)+'.pkl', 'rb'))
    train_data = Data(train_data)
    test_data = Data(test_data)
    # print(train_data)
    
    adj, num = sample_adj(adj, num_nodes, args.n_sample_all, num) #  adj, num: [n_nodes, sample_nums]

    #### Here is the main model of dual pipelines contrastive model:
    model = trans_to_cuda(DualContrastiveModel(args, num_nodes, adj, num))
    print('The arguments are:', args)
    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0

    ### start to train the model:

    wandb.watch(model, log="all")
    for epoch in range(args.epoch):
        print('-------------start to train-----------------')
        print('epoch:', epoch)
        hit, mrr, total_loss = train_and_test(model, train_data, test_data)
        # loging function
        wandb.log(
            {"Recall": hit,
            "MRR rate":mrr,
            "Total loss":total_loss}
        )

        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print("The Current result is:")
        print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f'%(hit, mrr))
        print("The Best result is :")
        print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d, \t%d'%(best_result[0], best_result[1], best_epoch[0], best_epoch[1]))

        bad_counter += 1 - flag
        if bad_counter >= args.patience:
            break
    print('-----------end of train-------------------')
    end = time.time()
    torch.save(model.state_dict(), 'model.h5')
    wandb.save('model.h5')
    print('The model has been')
    print("Run time:%fs", end - start)

if __name__ == '__main__':
    args = get_parser()
    # using wandb to visualize the train and test condition here
    # parser.add_argument('--dataset', default='diginetica', help='diginetica/Nowplaying/Tmall')
    # parser.add_argument('--hiddenSize', type=int, default=100)
    # parser.add_argument('--epoch', type=int, default=50)
    # parser.add_argument('--activate', type=str, default='relu')
    # parser.add_argument('--n_sample_all', type=int, default=14)
    # parser.add_argument('--n_sample', type=int, default=12)
    # parser.add_argument('--batch_size', type=int, default=100)
    # parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    # parser.add_argument('--lr_dc_step', type=float, default=3, help='the number of step after the learning rate decay')
    # parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay')
    # parser.add_argument('--l2', type=float, default=1e-5, help='the L2 pernalty')
    # parser.add_argument('--n_iter', type=int, default=1)
    # parser.add_argument('--dropout_gcn', type=float, default=0)
    # parser.add_argument('--dropout_local', type=float, default=0)
    # parser.add_argument('--dropout_global', type=float, default=0.5)
    # parser.add_argument('--validation', action='store_true')
    # parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
    # parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for leaky_relu')
    # parser.add_argument('--patience', type=int, default=10)

    wandb.init(project="Dual contrastive learning for session based")
    wandb.watch_called = False
    
    config = wandb.config
    config.dataset = args.dataset
    config.hiddenSize = args.hiddenSize
    config.epoch = args.epoch
    config.activate = args.activate
    config.n_sample_all = args.n_sample_all
    config.n_sample = args.n_sample
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.lr_dc_step = args.lr_dc_step
    config.lr_dc = args.lr_dc
    config.l2 = args.l2
    config.n_iter = args.n_iter
    config.dropout_gcn = args.dropout_gcn
    config.dropout_local = args.dropout_local
    config.dropout_global = args.dropout_global
    config.validation = args.validation
    config.valid_portion = args.valid_portion
    config.alpha = args.alpha
    config.patience = args.patience


    print("test", config)
    train_and_valid(config)