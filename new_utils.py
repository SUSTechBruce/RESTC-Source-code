import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.sparse import coo_matrix

def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set  # ([], []), len(train_data[0])=len(train_data[1])=719470
    n_samples = len(train_set_x)  # n_samples = 719470
    sidx = np.arange(n_samples, dtype='int32')  # [0..........719739]: type = 1 x 719470
    np.random.shuffle(sidx)  # 打乱sidx
    n_train = int(np.round(n_samples * (1. - valid_portion)))  # 取前百分之多少作为训练集，剩下的作为测试集
    #  随机取
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


def get_maxlen(train_data, test_data):
    train_sessions = train_data[0]
    test_sessions = test_data[0]
    len_train_data = [len(session) for session in train_sessions]
    len_test_data = [len(session) for session in test_sessions]
    total_data = len_train_data + len_test_data
    max_len = max(total_data)
    print('The max length of session data', max_len)

    return max_len
    

def get_graph_transformer_data(sessions, max_len, num_nodes, train_len=None):
    """
    返回填0并反转的session，掩码序列和最大长度
     return:
     graph_pois的idx
     graph_mask graph的mask
     transformer_seq, transformer输入的序列的idx
     transformer_pos transformer输入的绝对位置编码idx
     max_len 单个session的最大长度

    """
    len_data = [len(session) for session in sessions]
    
    # reverse the sequence
    graph_pois = []
    transformer_seq = []
    transformer_pos = []
    graph_mask = []
    count = 0
    for session, le in zip(sessions, len_data):
        count += 1
        if le < max_len: # 反转session 链表
            graph_pois.append(list(reversed(session)) + [0]*(max_len-le))
            transformer_seq.append([0]*(max_len - le) + list(session) + [num_nodes])
            transformer_pos.append([0]*(max_len - le) + list(range(1, len(session) +2))) # [0, 0, 0 .. + len(session) idx + 1]
            graph_mask.append([1]*le + [0] * (max_len-le))
            if count == 777:
                print(list(reversed(session)) + [0]*(max_len-le))
                print([1]*le + [0] * (max_len-le))
                print([0]*(max_len - le) + list(session) + [num_nodes])
                print([0]*(max_len - le) + list(range(1, len(session) + 2)))

    
        else:
            graph_pois.append(list(reversed(session[-max_len:])))
            graph_mask.append([1]*max_len)
            transformer_seq.append(list(session[-max_len:]) + [num_nodes])
            transformer_pos.append(list(range(1, max_len + 2)))

    return graph_pois, graph_mask, transformer_seq, transformer_pos, max_len

def handle_data(sessions, train_len=None):
    """
    返回填0并反转的session，掩码序列和最大长度
    """
    len_data = [len(session) for session in sessions]
    if train_len is None:
        max_len = max(len_data)
    else:
        max_len = train_len
    # reverse the sequence
    us_pois = []
    us_mask = []
    count = 0
    for session, le in zip(sessions, len_data):
        count += 1
        if le < max_len: # 反转session 链表
            us_pois.append(list(reversed(session)) + [0]*(max_len-le))
            us_mask.append([1]*le + [0] * (max_len-le))
            if count == 777:
                print(list(reversed(session)) + [0]*(max_len-le))
                print([1]*le + [0] * (max_len-le))

        else:
            us_pois.append(list(reversed(session[-max_len:])))
            us_mask.append([1]*max_len)

    return us_pois, us_mask, max_len

# handle_adj(adj, num_nodes, args.n_sample_all, num)

def sample_adj(adj_dict, n_entity, sample_num, num_dict=None):
    """
    adj_dict: adj
    n_entity: num_nodes
    sample_nums: args.n_sample_all
    num_dict: weights of adj
    """
    adj_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    weight_entity = np.zeros([n_entity, sample_num], dtype=np.int64)
    for entity in range(1, n_entity):
        neighbor = list(adj_dict[entity])
        neighbor_weight = list(num_dict[entity])
        n_neighbor = len(neighbor)
        if neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sample_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        else:
            sample_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        adj_entity[entity] = np.array([neighbor[i] for i in sample_indices])
        weight_entity[entity] = np.array([neighbor_weight[i] for i in sample_indices])
    
    return adj_entity, weight_entity

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable

def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

def get_graph_matrix(all_sessions, n_node):
    """
    
    处理全局sessions邻接图
    """
    adj = dict()
    for sess in all_sessions:
        for i, item in enumerate(sess):
            if i == len(sess)-1:
                break
            else:
                if sess[i] - 1 not in adj.keys():
                    adj[sess[i]-1] = dict() # 从0开始计算dict的index
                    adj[sess[i]-1][sess[i]-1] = 1
                    adj[sess[i]-1][sess[i+1]-1] = 1
                else:
                    if sess[i+1]-1 not in adj[sess[i]-1].keys():
                        adj[sess[i] - 1][sess[i + 1] - 1] = 1
                    else:
                        adj[sess[i]-1][sess[i+1]-1] += 1 # 计算邻接transition item的重复次数
    row, col, data = [], [], []
    for i in adj.keys():
        item = adj[i] # item: i的邻居
        for j in item.keys():
            row.append(i)
            col.append(j)    # row [i, i, i...] / col [j1, j2, j3]
            data.append(adj[i][j]) # data : [重复出现的次数]
    print('data shape', np.array(data).size)
    coo = coo_matrix((data, (row, col)), shape=(n_node, n_node))
    return coo # 返回一个压缩矩阵

class DualModelData(Dataset):
    def __init__(self, data, global_data, max_len, num_nodes, train_len=None, is_train=True):
        if num_nodes == None:
            num_nodes = 43098
        graph_inputs, graph_masks, transformer_seqs, transformer_pos, max_len = get_graph_transformer_data(data[0], max_len, num_nodes, train_len)
        self.inputs = np.asarray(graph_inputs)
        self.mask = np.asarray(graph_masks)
        self.transformer_seqs = np.asarray(transformer_seqs)
        self.transformer_pos = np.asarray(transformer_pos)
        self.max_len = max_len
        self.targets = np.asarray(data[1])
        self.length = len(data[0])
        # print('input_style', self.inputs[55])
        # print('input_len', len(self.inputs[55]))
        self.global_graph = None
        if is_train:
            self.global_graph = get_graph_matrix(global_data, num_nodes)
            print('Loading global graph matrix')
        print('target shape', self.targets.shape)

    def get_global_matrix(self, global_data, num_nodes):
        """
        Getting global item embedding.
        """
        adj = dict()
        for session in global_data:
            for i, node in enumerate(session):
                if i == len(session) - 1: # last item index
                    break
                else:
                    if session[i] - 1 not in adj.keys():
                        adj[session[i] - 1] = dict()
                        adj[session[i] - 1][session[i] - 1] = 1 # item itself
                        adj[session[i] - 1][session[i + 1] - 1] = 1 # next item
                    else:
                        if session[i + 1] -1 not in adj[session[i] - 1].keys():
                            adj[session[i] - 1][session[i + 1] - 1] = 1
                        else:
                            adj[session[i] - 1][session[i + 1]] += 1 # calculate repetition number
        row = []
        col = []
        value = []
        for i in adj.keys():
            node = adj[i] # neighbor of i item
            for j in node.keys():
                row.append(i)
                col.append(j)
                value.append(adj[i][j])
        global_graph = coo_matrix((value, (row, col)), shape=(num_nodes, num_nodes))

        return global_graph


    def __getitem__(self, index): # bug here
        u_input = self.inputs[index]
        mask = self.mask[index]
        label = self.targets[index]
        transformer_seq = self.transformer_seqs[index]
        transformer_po = self.transformer_pos[index]
        max_n_node = self.max_len
        node = np.unique(u_input)
        items = node.tolist()+(max_n_node - len(node)) * [0]
        adj = np.zeros((max_n_node, max_n_node))
        
        for i in np.arange(len(u_input) - 1):
            u = np.where(node == u_input[i])[0][0] # session对应的在session node中的索引
            adj[u][u] = 1
            if u_input[i + 1] == 0:
                break
            v = np.where(node == u_input[i+1])[0][0]
            if u == v or adj[u][v] == 4:
                continue
            adj[v][v] = 1 
            if adj[v][u] == 2:
                adj[u][v] = 4 # here are the bugs
                adj[v][u] = 4
            else:
                adj[u][v] = 2
                adj[v][u] = 3
        
        alias_inputs = [np.where(node == i)[0][0] for i in u_input]
        # change them to torch
        alias_inputs = torch.tensor(alias_inputs) # 单个session在 unique session中的index
        adj = torch.tensor(adj) # 单个session构alias_inputs建的邻接矩阵
        items = torch.tensor(items) # unique sessions + 0补位
        mask = torch.tensor(mask) 
        label = torch.tensor(label)
        u_input = torch.tensor(u_input) 
        transformer_seq = torch.tensor(transformer_seq)
        transformer_po = torch.tensor(transformer_po)

        return [alias_inputs, adj, items, mask, label, u_input, transformer_seq, transformer_po]   

    def __len__(self):
        return self.length


############## 代码测评标准

# Recall, also HR
def get_recall(pre, truth):
    """
    :param pre: (B,K) TOP-K indics predicted by the model
    :param truth: (B,1) the truth value of test samples
    :return: recall(Float), the recall score
    """
    truths = truth.expand_as(pre)
    hits = (pre == truths).nonzero()
    if len(hits) == 0:
        return 0
    n_hits = (pre == truths).nonzero().size(0)
    recall = n_hits / truths.size(0)
    return recall


# MRR
def get_mrr(pre, truth):
    """
    :param pre: (B,K) TOP-K indics predicted by the model
    :param truth: (B, 1) real label
    :return: MRR(Float), the mrr score
    """
    targets = truth.view(-1, 1).expand_as(pre)
    # ranks of the targets, if it appears in your indices
    hits = (targets == pre).nonzero()
    if len(hits) == 0:
        return 0
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    r_ranks = torch.reciprocal(ranks)  # reciprocal ranks
    mrr = torch.sum(r_ranks).data / targets.size(0)
    return mrr

