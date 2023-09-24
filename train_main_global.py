"""
train main for transformer/gcn fusion
new add global item embedding
start to add global item embedding to the last predict mutiplication

"""


"""
 train_data = pickle.load(open('Dual_contrastive/datasets/'+ args.dataset +'/train.txt', 'rb'))
    if args.validation:
        print('Split the validation set from train set')
        train_data, valid_data = split_validation(train_data, args.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('Dual_contrastive/datasets/'+ args.dataset +'/test.txt', 'rb'))
    global_data = pickle.load(open('Dual_contrastive/datasets/' + args.dataset + '/all_train_seq.txt', 'rb'))
"""
import time
import argparse
import pickle
import numpy as np
from torch.utils.data.dataset import Dataset
from dual_gnn_transformer import train_and_test, DualSSlGloablModel
from new_utils import split_validation, trans_to_cuda, DualModelData, get_maxlen # using new_utils instead of 
import torch
import logging
import warnings
import os
import random
torch.distributed.init_process_group(backend="nccl")

local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

warnings.filterwarnings("ignore")


logging.propagate = False
logging.getLogger().setLevel(logging.ERROR)


def init_seed(seed=None):
    if seed == None:
        seed = int(time.time() * 1000//1000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='RetailRocket', help='RetailRocket/Nowplaying/Tmall')
    parser.add_argument('--hiddenSize', type=int, default=200)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--activate', type=str, default='relu')
    parser.add_argument('--n_sample_all', type=int, default=12) # initial value is 12
    parser.add_argument('--n_sample', type=int, default=12) # initial value is 12
    parser.add_argument('--batch_size', type=int, default=256) # 300 for Tmall
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_dc_step', type=float, default=3, help='the number of step after the learning rate decay')
    parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay')
    parser.add_argument('--l2', type=float, default=1e-5, help='the L2 pernalty')
    parser.add_argument('--n_iter', type=int, default=2)
    parser.add_argument('--dropout_gcn', type=float, default=0.2)
    parser.add_argument('--dropout_local', type=float, default=0.2)
    parser.add_argument('--dropout_global', type=float, default=0.5)
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for leaky_relu')
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--data_dir', default='data_dir', help='The input training data')
    parser.add_argument('--output_dir', default='output_dir', help='The output log training data')
    parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')

    opt = parser.parse_args()
    return opt

def train_and_valid(args, local_rank):

    init_seed(2020)
    if args.dataset == 'diginetica': # Accieve SOTA
        num_nodes = 43098
        args.dropout_local = 0.0
    elif args.dataset == 'Tmall': # Acchieve SOTA
        num_nodes = 40728
        args.dropout_local = 0.5
    elif args.dataset == 'Nowplaying': # HR SOTA
        num_nodes = 60418
        args.dropout_local = 0.0
    elif args.dataset == 'yoochoose1_64': # HR SOTA 
        num_nodes = 37484
        args.dropout_local = 0.5
    elif args.dataset == 'yoochoose1_4':
        num_nodes = 37484
        args.dropout_local = 0.5
    elif args.dataset == 'RetailRocket': # Acchieve SOTA
        num_nodes = 36969
        args.dropout_local = 0.2
    elif args.dataset == 'lastfm': # Acchieve SOTA
        num_nodes = 38617
        args.dropout_local = 0.5
     
    else:
        num_nodes = 310
    data_path = args.data_dir
    train_data = pickle.load(open(data_path+ args.dataset +'/train.txt', 'rb'))
    if args.validation:
        print('Split the validation set from train set')
        train_data, valid_data = split_validation(train_data, args.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open(data_path+ args.dataset +'/test.txt', 'rb'))
    global_data = pickle.load(open(data_path+ args.dataset + '/all_train_seq.txt', 'rb'))

    max_len = get_maxlen(train_data, test_data)
    # if max_len > 200:
    #     max_len = 200

    train_data = DualModelData(train_data,  global_data, max_len, num_nodes, is_train = True)
    test_data = DualModelData(test_data,  global_data, max_len, num_nodes, is_train = False)
    
    adj = None
    num = None

    #### Here is the main model of dual pipelines contrastive model:
    global_graph = train_data.global_graph
    model = trans_to_cuda(DualSSlGloablModel(args, global_graph, num_nodes, adj, num))
    
    print('The arguments are:', args)
    start = time.time()
    best_result = [0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0]
    bad_counter = 0

    ### start to train the model:
    path = args.output_dir
    logging.basicConfig(level=logging.DEBUG,  
                    filename= path + 'train_global_2_layers_with_dim_368_512_gat_emb_glu_graph_0.005_' + args.dataset +'.log', 
                    filemode='a',  
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
    for epoch in range(args.epoch):
        print('-------------start to train-----------------')
        print('epoch:', epoch)
        if args.local_rank == 0:
           os.system("nvidia-smi")

        hit_20, mrr_20, hit_10, mrr_10, total_loss = train_and_test(local_rank, epoch, args, model, train_data, test_data)
    
        flag = 0
        if hit_20 >= best_result[0]:
            best_result[0] = hit_20
            best_epoch[0] = epoch
            flag = 1
        if mrr_20 >= best_result[1]:
            best_result[1] = mrr_20
            best_epoch[1] = epoch
            flag = 1

        if hit_10 >= best_result[2]:
            best_result[2] = hit_10
            best_epoch[2] = epoch

        if mrr_10 >= best_result[3]:
            best_result[3] = mrr_10
            best_epoch[3] = epoch
        if args.local_rank == 0:
            
            print("The Current result is:")
            print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f'%(hit_20, mrr_20))
            print('\tRecall@10:\t%.4f\tMRR@10:\t%.4f'%(hit_10, mrr_10))
            print("The Best result is :")
            print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d, \t%d'%(best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
            print('\tRecall@10:\t%.4f\tMRR@10:\t%.4f\tEpoch:\t%d, \t%d'%(best_result[2], best_result[3], best_epoch[2], best_epoch[3]))

            logging.info('The Current result is:')
            logging.info(f'Recall@20: {hit_20}, MRR@20: {mrr_20}')
            logging.info(f'Recall@20: {hit_10}, MRR@20: {mrr_10}')
            logging.info("The Best result is :")
            logging.info(f'Recall@20: {best_result[0]}, MRR@20: {best_result[1]}, Epoch: {best_epoch[0]}, Epoch: {best_epoch[1]}')
            logging.info(f'Recall@20: {best_result[2]}, MRR@20: {best_result[3]}, Epoch: {best_epoch[2]}, Epoch: {best_epoch[3]}')

        bad_counter += 1 - flag
        if bad_counter >= args.patience:
            break
    print('-----------end of train-------------------')
    end = time.time()
    # torch.save(model.state_dict(), 'model.h5')
    print('The model has been done')
    print("Run time:%fs", end - start)

if __name__ == '__main__':
    args = get_parser()
    config = args
    print("test", config)
    print('Start to training model ......')
    train_and_valid(config, local_rank)