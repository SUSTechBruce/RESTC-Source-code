import time
import argparse
import pickle
from model import *
from utils import *


def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='lastfm', help='diginetica/Nowplaying/Tmall')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=15)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=1)                                    # [1, 2]
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')       # [0, 0.2, 0.4, 0.6, 0.8]
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     # [0, 0.5]
parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--data_dir', default='data_dir', help='The input training data')
parser.add_argument('--output_dir', default='output_dir', help='The output log training data')
parser.add_argument('--local_rank', type=int, default=0, help='node rank for distributed training')

opt = parser.parse_args()


def main():
    init_seed(2020)

    if opt.dataset == 'diginetica':
        num_node = 43098
        opt.n_iter = 2
        opt.dropout_gcn = 0.2
        opt.dropout_local = 0.0
    elif opt.dataset == 'Nowplaying':
        num_node = 60417
        opt.n_iter = 1
        opt.dropout_gcn = 0.0
        opt.dropout_local = 0.0
    elif opt.dataset == 'Tmall':
        num_node = 40728
        opt.n_iter = 1
        opt.dropout_gcn = 0.6
        opt.dropout_local = 0.5
    elif opt.dataset == 'gowalla':
        num_node = 30000
        opt.dropout_local = 0.5
    elif opt.dataset == 'lastfm':
        num_node = 38616
        opt.dropout_local = 0.5
    elif opt.dataset == 'RetailRocket':
        num_node = 36969
        opt.dropout_local = 0.2
     

    else:
        num_node = 310
    data_path = opt.data_dir
    train_data = pickle.load(open(data_path + opt.dataset +'/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open(data_path + opt.dataset + '/test.txt', 'rb'))

    adj = pickle.load(open(data_path + opt.dataset + '/adj_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    num = pickle.load(open(data_path + opt.dataset + '/num_' + str(opt.n_sample_all) + '.pkl', 'rb'))
    train_data = Data(train_data)
    test_data = Data(test_data)

    adj, num = handle_adj(adj, num_node, opt.n_sample_all, num)
    model = trans_to_cuda(CombineGraph(opt, num_node, adj, num))
    print(model)
    print(opt)
    start = time.time()
    best_result = [0, 0, 0, 0]
    best_epoch = [0, 0, 0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit_20, mrr_20, hit_10, mrr_10, total_loss = train_test(model, train_data, test_data)
        # loging function
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

        
        
        print("The Current result is:")
        print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f'%(hit_20, mrr_20))
        print('\tRecall@10:\t%.4f\tMRR@10:\t%.4f'%(hit_10, mrr_10))
        print("The Best result is :")
        print('\tRecall@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d, \t%d'%(best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        print('\tRecall@10:\t%.4f\tMRR@10:\t%.4f\tEpoch:\t%d, \t%d'%(best_result[2], best_result[3], best_epoch[2], best_epoch[3]))

        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-----------end of train-------------------')
    end = time.time()
    torch.save(model.state_dict(), 'model.h5')
    print('The model has been')
    print("Run time:%fs", end - start)



if __name__ == '__main__':
    main()
