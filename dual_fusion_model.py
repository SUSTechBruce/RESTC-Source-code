"""
First edition
Dual fusion model: Combine transformer sequence model and GAT spacial model 
one model to learn temporal information of session data, and other GAT model learns the structual and spacial information of 
"""

import datetime
import math
import torch
from tqdm import tqdm
import torch.nn.functional as F
from utils import trans_to_cuda
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
from torch.functional import norm
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from entmax import entmax_bisect
from torch.nn.init import xavier_normal_

class LocalAggregator(nn.Module):

    def __init__(self, dim, alpha, dropout=0., name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout
        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1)) # 权重向量
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

        self.leakyrelu = nn.LeakyReLU(alpha)
    
    def forward(self, hidden, adj, mask_item=None):
        """
        hidden: the embeddings of single unique session
        adj: the adj matrix of single session
        mask_item: the mask 0 1 list of single session
        """
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]

        h_trans1 = h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
        h_trans2 = h.repeat(1, N, 1)

        h_mul = h_trans1 * h_trans2
        a_input = h_mul.view(batch_size, N, N, self.dim) # item 和 item之间两两相乘

        e_0 = torch.matmul(a_input, self.a_0) # [batch_size, N, N, dim] * [dim, 1]
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)
        
        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask) # r_in, r_self, r_out, r_in_out
        alpha = torch.where(adj.eq(2), e_1, alpha) # 建模session 的adj matrix
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)

        return output

class GlobalAggregator(nn.Module):
    def __init__(self, dim, dropout, act=torch.relu, name=None):
        super(GlobalAggregator, self).__init__() 
        self.dropout = dropout
        self.act = act
        self.dim = dim

        self.w_1 = nn.Parameter(torch.Tensor(self.dim + 1, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.w_3 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.bias = nn.Parameter(torch.Tensor(self.dim))

    def forward(self, self_vectors, neighbor_vector, batch_size, masks, neighbor_weight, extra_vector=None):
        if extra_vector is not None:
            alpha = torch.matmul(torch.cat([extra_vector.unsqueeze(2).repeat(1, 1, neighbor_vector.shape[2], 1)*neighbor_vector, neighbor_weight.unsqueeze(-1)], -1), self.w_1).squeeze(-1)
            alpha = F.leaky_relu(alpha, negative_slope=0.2)
            alpha = torch.matmul(alpha, self.w_2).squeeze(-1)
            alpha = torch.softmax(alpha, -1).unsqueeze(-1)
            neighbor_vector = torch.sum(alpha * neighbor_vector, dim=-2)
        else:
            neighbor_vector = torch.mean(neighbor_vector, dim=2)
        # self_vectors = F.dropout(self_vectors, 0.5, training=self.training)
        output = torch.cat([self_vectors, neighbor_vector], -1)
        output = F.dropout(output, self.dropout, training=self.training)
        output = torch.matmul(output, self.w_3)
        output = output.view(batch_size, -1, self.dim)
        output = self.act(output)
        return output


# model = trans_to_cuda(DualContrastiveModel(args, num_nodes, adj, num))
class DualTransGraphModel(torch.nn.Module):
    def __init__(self, args, num_nodes, adj_all, weight_all):
        super(DualTransGraphModel, self).__init__()
        self.num_nodes = num_nodes
        self.adj_all = adj_all
        self.weight_all = weight_all
        self.batch_size = args.batch_size
        self.dim = args.hiddenSize
        self.dropout_local = args.dropout_local
        self.dropout_global = args.dropout_global
        self.k_hop = args.n_iter
        self.sample_num = args.n_sample
        # self.adj_all = trans_to_cuda(torch.Tensor(adj_all)).long()
        # self.weight_all = trans_to_cuda(torch.Tensor(weight_all)).float()
    
        # aggregator
        self.local_agg = LocalAggregator(self.dim, args.alpha, dropout=0.0)
        self.global_agg = []
        for i in range(self.k_hop):
            if args.activate == 'relu':
                agg = GlobalAggregator(self.dim, args.dropout_gcn, act=torch.relu)
            else:
                 agg = GlobalAggregator(self.dim, args.dropout_gcn, act=torch.tanh)
            self.add_module('agg_gcn_{}'.format(i), agg)
            self.global_agg.append(agg)
            # self.global_agg_modellist = nn.ModuleList(self.global_agg)
        
        ## transformer layer
        item_dim = 100
        pos_dim = 100
        n_items = self.num_nodes
        n_pos = 300
        w = 20
        dropout = 0.5
        activate = 'relu'
        self.transformer_emb = SparseTransformer(item_dim, pos_dim, n_items, n_pos, w, dropout=dropout, activate=activate) #  # 512, 1, 100

        # item representation and position encoding
        self.embedding = nn.Embedding(self.num_nodes, self.dim) # [43097, 100]
        self.pos_embedding = nn.Embedding(300, self.dim) 

        # parameter of DualContrastiveModel
        self.w_1 = nn.Parameter(torch.Tensor(self.dim * 2, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(args.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.l2)
        self.scheduler_stepLR = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        self.scheduler_crosineAnneal = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=3)

        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)

        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    
    def sample(self, target): # [n_nodes, sample_nums]
        t_adj_list = self.adj_all[target.view(-1)] # 根据target选neighbor list
        t_weight_list = self.weight_all[target.view(-1)] # 根据target选weight list 
        # t_weight_list torch.Size([6900, 14])
        return t_adj_list, t_weight_list
    

    # 总的transformer encoding作为序列信息在  hs = hs / torch.sum(mask, 1) 替换
    # 提取transformer序列表征，在scores = DualContrastiveModel.compute_scores(seq_hidden, mask)前concate


    def compute_scores(self, hidden, mask, transformer_emb):
        """
        transformer_emb: 100, dim
        """
        mask = mask.float().unsqueeze(-1)
        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        # hs = torch.sum(hidden * mask, -2) # 第二个维度相加[batch, sequence, hidden]
        # hs = hs / torch.sum(mask, 1)  # batch, 1, seq 100, 1, 69
        # hs = hs.unsqueeze(-2).repeat(1, len, 1) # s' # 插入transformer 时序表征
        transformer_emb = transformer_emb.unsqueeze(-2).repeat(1, len, 1)
        # ('transformer_emb', transformer_emb.shape)

        concat_hs_pos = torch.cat([pos_emb, hidden], -1) # concat in dimension
        nh = torch.matmul(concat_hs_pos, self.w_1)
        nh = torch.tanh(nh) # (11)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(transformer_emb)) # (13)
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1) # (14)

        b = self.embedding.weight[1:]
        scores = torch.matmul(select, b.transpose(1, 0)) # (15) [batch ,seq, dim] * [batch, dim, seq]
        # print('score shape', scores.shape)
        return scores
     # code review here


    def forward(self, inputs, adj, mask_item, item, transformer_seqs, transformer_pos):
        """
        alias_input: 单个session在 unique session中的index
        adj: 单个session构成的领接矩阵
        items：unique session + 0 padding
        mask：单个session，存在item为1，其余补位0
        targets：label
        inputs： 单个反转session + (maxlen  -session len)补位0

        hidden = DualContrastiveModel(items, adj, mask, inputs) # forward function

        hidden = model(items, adj, mask, inputs)
        """
        
        batch_size = inputs.shape[0]
        seqs_len = inputs.shape[1]
        h = self.embedding(inputs)  # inputs == items

        # local graph
        h_local = self.local_agg(h, adj, mask_item) #  

        # # global graph
        # item_neighbors = [inputs]
        # weight_neighbors = [] # i -> sample 12 * i
        # support_size = seqs_len

        # for i in range(1, self.k_hop + 1):
        #     item_sample_i, weight_sample_i = self.sample(item_neighbors[-1]) 
        #     # print('item_neighbor shape', item_neighbors[-1].shape) # item_neighbor shape torch.Size([100, 69])
        #     # item_sample_i shape: [6900, 14]
        #     support_size = support_size * self.sample_num
        #     item_neighbors.append(item_sample_i.view(batch_size, support_size))
        #     weight_neighbors.append(weight_sample_i.view(batch_size, support_size))
        
        # entity_vectors = [self.embedding(i) for i in item_neighbors]
        # weight_vectors = weight_neighbors

        # session_info = []
        # # item == inputs
        # item_emb = self.embedding(item) * mask_item.float().unsqueeze(-1) # item_emb: [batch_size, seq_len, dim] * [batch_size, seq_len, 1]
        # sum_item_emb = torch.sum(item_emb, 1) / torch.sum(mask_item.float(), -1).unsqueeze(-1) # [batch_size, dim] / [batch_size, 1]
        # sum_item_emb = sum_item_emb.unsqueeze(-2) # batch_size, 1 ,dim

        # for i in range(self.k_hop):
        #     session_info.append(sum_item_emb.repeat(1, entity_vectors[i].shape[1], 1)) # [batch_size, support_size, dim] 当前的session平均值再扩展为和session’s neighbors 同一个length
        # for n_hop in range(self.k_hop):
        #     entity_vectors_next_iter = []
        #     shape = [batch_size, -1, self.sample_num, self.dim]
        #     for hop in range(self.k_hop - n_hop): # [2, 1] # 计算一个session和这个session item的邻居所构成的attention系数
        #         aggregator = self.global_agg[n_hop] # global_0
        #         vector = aggregator(self_vectors=entity_vectors[hop], # entity vector include session itself and its neighbors
        #                             neighbor_vector=entity_vectors[hop+1].view(shape), # neighbor_vector: [batch_size, -1, self.sample, dim]
        #                             masks=None,
        #                             batch_size=batch_size,
        #                             neighbor_weight=weight_vectors[hop].view(batch_size, -1, self.sample_num), # weight vector only include the weight of its neighbors
        #                             extra_vector=session_info[hop])
        #         entity_vectors_next_iter.append(vector)
        #     entity_vectors = entity_vectors_next_iter
        
        # # combine global and local
        # h_global = entity_vectors[0].view(batch_size, seqs_len, self.dim)
        # h_global = F.dropout(h_global, self.dropout_global, training=self.training)
        h_local = F.dropout(h_local, self.dropout_local, training=self.training)
    
        ## compute transformer embed
        transformer_emb = self.transformer_emb(transformer_seqs, transformer_pos) # 512 100
        
        output = h_local
        # print('seq_len: ', seqs_len) 69
        return output, transformer_emb

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

# the forward function of the DualContrastive model

def forward(DualTransGraphModel, data):
    """
    alias_input: 单个session在 unique session中的index
    adj: 单个session构成的领接矩阵
    items：unique session + 0 padding
    mask：单个session，存在item为1，其余补位0
    targets：label
    inputs： 单个反转session + (maxlen  -session len)补位0
    """
    alias_inputs, adj, items, mask, targets, inputs, transformer_seqs, transformer_pos = data
    alias_inputs = trans_to_cuda(alias_inputs).long()
    items = trans_to_cuda(items).long()
    adj = trans_to_cuda(adj).float()
    mask = trans_to_cuda(mask).long()
    inputs = trans_to_cuda(inputs).long()
    transformer_seqs = trans_to_cuda(transformer_seqs).long()
    transformer_pos = trans_to_cuda(transformer_pos).long()

    hidden, transformer_emb = DualTransGraphModel(items, adj, mask, inputs, transformer_seqs, transformer_pos) # forward function
    # print('transformer_pos', transformer_pos)
    get = lambda index: hidden[index][alias_inputs[index]]
    seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()]) #  根据单个session在unique session的index转换的seq embedding
    # print('seq_len: ', seq_hidden.shape)
    scores = DualTransGraphModel.compute_scores(seq_hidden, mask, transformer_emb)
    return targets, scores


def train_and_test(epoch, model, train_data, test_data):
    print('start to training', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_loader = Data.DataLoader(train_data, batch_size=model.batch_size, shuffle=True, pin_memory=True)

    for data in tqdm(train_loader): 
        model.optimizer.zero_grad()
        targets, scores = forward(model, data)
        targets = trans_to_cuda(targets).long()
        loss = model.loss_function(scores, targets - 1)
        loss.backward()
        model.optimizer.step()
        total_loss += loss
    print('\tLoss:\t%.3f'% total_loss)
    if epoch < 10:
        model.scheduler_stepLR.step()
    else:
        model.scheduler_crosineAnneal.step()

    print('Start to predict test data', datetime.datetime.now())
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=model.batch_size,shuffle=False, pin_memory=True)
    result = []
    hit, mrr = [], []

    # 测试，并计算召回率和mrr rate
    for data in test_loader:
        targets, scores = forward(model, data)
        sub_scores = scores.topk(20)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        targets = targets.numpy()
        for score, target, mask in zip(sub_scores, targets, test_data.mask):
            hit.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr.append(0)
            else:
                mrr.append(1 /(np.where(score == target - 1)[0][0] + 1))

    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100

    return hit, mrr, total_loss


class SparseTransformer(nn.Module):

    def __init__(self, item_dim, pos_dim, n_items, n_pos, w, atten_way='dot', decoder_way='bilinear', dropout=0,
                 activate='relu'):
        super(SparseTransformer, self).__init__()
        self.item_dim = item_dim
        self.pos_dim = pos_dim
        dim = item_dim + pos_dim
        self.dim = dim
        self.n_pos = n_pos
        self.n_items = n_items    # 所有session unique items的总数
        self.embedding = nn.Embedding(n_items + 1, self.item_dim, padding_idx=0,max_norm=1.5)
        self.pos_embedding = nn.Embedding(self.n_pos, self.pos_dim, padding_idx=0, max_norm=1.5)
        self.atten_way = atten_way
        self.decoder_way = decoder_way
        self.atten_w0 = nn.Parameter(torch.Tensor(1, self.dim))
        self.atten_w1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_w2 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_bias = nn.Parameter(torch.Tensor(self.dim))
        self.w_f = nn.Linear(2*self.dim, self.item_dim)
        self.dropout = nn.Dropout(dropout)
        self.self_atten_w1 = nn.Linear(self.dim, self.dim)
        self.self_atten_w2 = nn.Linear(self.dim, self.dim)
        
        self.LN = nn.LayerNorm(self.dim)
        self.LN2 = nn.LayerNorm(self.item_dim)
        self.is_dropout = True
        self.attention_mlp = nn.Linear(dim, dim) # the mlp of attention
        self.alpha_w = nn.Linear(dim, 1)
        self.w = w
        
        if activate == 'relu':
            self.activate = F.relu
        elif activate == 'selu':
            self.activate = F.selu

        self.initial_(initial_way='norm')

    def initial_(self, initial_way='norm'):
        if initial_way == 'norm':
            init.normal_(self.atten_w0, mean=0, std=0.05)
            init.normal_(self.atten_w1, mean=0, std=0.05)
            init.normal_(self.atten_w2, mean=0, std=0.05)
        elif initial_way == 'xavier':
            xavier_normal_(self.atten_w0.data, gain=1.414)
            xavier_normal_(self.atten_w1.data, gain=1.414)
            xavier_normal_(self.atten_w2.data, gain=1.414)
        init.constant_(self.atten_bias, 0)
        init.constant_(self.attention_mlp.bias, 0)
        init.constant_(self.embedding.weight[0], 0)
        init.constant_(self.pos_embedding.weight[0], 0)
    
    def get_alpha(self, session, flag):
        """"
        get alpha from sigmoid, the input is the last preference term of user.
        When the sigmoid output + 1.0001 = 1, the alpha = 1.0001 is too sparse 
        and need to be recorded
        input shape: [batch_size, dim]
        output shape: [batch_size, 70, 1]
        """
        max_len = 285
        if flag == 0:
            alpha_ent = torch.sigmoid(self.alpha_w(session)) + 1
            # alpha_ent = self.add_value(alpha_ent).unsqueeze(1)
            mask = (alpha_ent == 1).float()
            alpha_ent = alpha_ent.masked_fill(mask == 1, 1.00001)
            alpha_ent = alpha_ent.unsqueeze(1)
            alpha_ent = alpha_ent.expand(-1, max_len, -1)  # 单一维度扩展 70
            return alpha_ent
        if flag == 1:
            alpha_global = torch.sigmoid(self.alpha_w(session)) + 1
            mask = (alpha_global == 1).float()
            alpha_ent = alpha_global.masked_fill(mask == 1, 1.00001)
            return alpha_global

    def add_value(self, value):

        mask_value = (value ==1).float()
        value = value.masked_fill(mask_value == 1, 1.00001)
        return value
    

    def self_attention(self, Q, K, V, seq_mask, alpha_ent):
        """
        Single head attention with softmax
        """
        if self.is_dropout:
            # Q_ = self.dropout(self.activate(self.attention_mlp(Q)))
            Q_ = self.attention_mlp(Q)
            Q_ = self.activate(Q_)
            Q_ = self.dropout(Q_)
        else:
            # Q_ = self.activate(self.attention_mlp(Q))
            Q_ = self.attention_mlp(Q)
            Q_ = self.activate(Q_)
        attention_matrix = torch.matmul(Q_, K.transpose(1, 2)) / math.sqrt(self.dim)
        if seq_mask is not None:
            mask = seq_mask.unsqueeze(1).expand(-1, Q.size(1), -1)
            attention_matrix = attention_matrix.masked_fill(mask == 0, -np.inf)      
        # alpha = entmax_bisect(attention_matrix, alpha_ent, dim=-1)
        alpha = torch.softmax(attention_matrix, dim=-1)
        
        A = torch.matmul(alpha, V)  # B, seq, dim
        if self.is_dropout:
            # att_v = self.dropout(self.self_atten_w2(self.activate(self.self_atten_w1(A)))) + A
            X = self.self_atten_w1(A) # FFN part of transformer
            X = self.activate(X) # nn.relu()
            X = self.self_atten_w2(X) # nn.Linear()
            X = self.dropout(X)
            X = X + A   # the residual network of transformer
        else:
            X = self.self_atten_w1(A) # FFN part of transformer
            X = self.activate(X) # nn.relu()
            X = self.self_atten_w2(X) # nn.Linear()
            X = self.dropout(X)
            X = X + A   # the residual network of transformer
        X = self.LN(X)
        session_s = X[:, -1, :].unsqueeze(1)
        session_x = X[:, :-1, :]

        return session_s, session_x # session and last item of session



       
    def self_sparse_attention(self, Q, K, V, seq_mask, alpha_ent):
        """
        Single head attention with alpha-entmax to be inplace of softmax
        """
        if self.is_dropout:
            # Q_ = self.dropout(self.activate(self.attention_mlp(Q)))
            Q_ = self.attention_mlp(Q)
            Q_ = self.activate(Q_)
            Q_ = self.dropout(Q_)
        else:
            # Q_ = self.activate(self.attention_mlp(Q))
            Q_ = self.attention_mlp(Q)
            Q_ = self.activate(Q_)
        attention_matrix = torch.matmul(Q_, K.transpose(1, 2)) / math.sqrt(self.dim)
        if seq_mask is not None:
            mask = seq_mask.unsqueeze(1).expand(-1, Q.size(1), -1)
            attention_matrix = attention_matrix.masked_fill(mask == 0, -np.inf)      
        alpha = entmax_bisect(attention_matrix, alpha_ent, dim=-1)

        A = torch.matmul(alpha, V)  # B, seq, dim
        if self.is_dropout:
            # att_v = self.dropout(self.self_atten_w2(self.activate(self.self_atten_w1(A)))) + A
            X = self.self_atten_w1(A) # FFN part of transformer
            X = self.activate(X) # nn.relu()
            X = self.self_atten_w2(X) # nn.Linear()
            X = self.dropout(X)
            X = X + A   # the residual network of transformer
        else:
            X = self.self_atten_w1(A) # FFN part of transformer
            X = self.activate(X) # nn.relu()
            X = self.self_atten_w2(X) # nn.Linear()
            # X = self.dropout(X)
            X = X + A   # the residual network of transformer
        X = self.LN(X)
        session_s = X[:, -1, :].unsqueeze(1)
        session_x = X[:, :-1, :]

        return session_s, session_x # session and last item of session

 
    def global_attention(self, Q_hat, K_hat, V_hat, seq_mask=None, alpha_ent=1):

        """
        Q_hat: session_s
        K_hat: session_x
        """
        Q = Q_hat.matmul(self.atten_w2)
        K = K_hat.matmul(self.atten_w1)
        bias = self.atten_bias
        W_0 = self.atten_w0

        A = torch.relu(Q + K + bias)  # batch, seq, dim
        A = torch.matmul(A, W_0.t()) # batch, seq, 1
        if seq_mask is not None:
            seq_mask = seq_mask.unsqueeze(-1) # batch, seq, 1
            seq_mask = seq_mask[:, :-1, :]
            A = A.masked_fill(seq_mask == 0, -np.inf)
        A = entmax_bisect(A, alpha_ent, dim=1) 
        c = torch.matmul(A.transpose(1, 2), V_hat) # (batch, 1, seq) * (batch, seq, dim) # 直接concate 改 batch seq seq
        return c

    def decoder(self, global_c, self_c):
        if self.is_dropout:
            c = self.dropout(torch.selu(self.w_f(torch.cat((global_c, self_c), 2))))
        else:
            c = torch.selu(self.w_f(torch.cat((global_c, self_c), 2)))
        c = c.squeeze()
        l_c = (c/torch.norm(c, dim=-1).unsqueeze(1))
        # print('l_c shape', l_c.shape)
        l_emb = self.embedding.weight[1:-1]/torch.norm(self.embedding.weight[1:-1], dim=-1).unsqueeze(1)
        z = self.w * torch.matmul(l_c, l_emb.t())
        # print('result, shape', z.shape)

        return z


    def get_transformer_embed(self, session_vec, last_item):
        if self.is_dropout:
            transformer_emb = self.dropout(torch.selu(self.w_f(torch.cat((session_vec, last_item), 2))))
        else:
            transformer_emb = torch.selu(self.w_f(torch.cat((session_vec, last_item), 2)))
        transformer_emb = transformer_emb.squeeze()
        norm_emb = (transformer_emb/torch.norm(transformer_emb, dim=-1).unsqueeze(1))
        return norm_emb

    def forward(self, x, pos):
        self.is_dropout = True
        x_embeddings = self.embedding(x)  # B,seq,dim
        pos_embeddings = self.pos_embedding(pos)  # B, seq, dim 
        mask = (x != 0).float()  # B,seq
        x_ = torch.cat((x_embeddings, pos_embeddings), 2)  # B seq, 2*dim
        x_s = x_[:, :-1, :]  # B, seq-1, 2*dim 69
        alpha_ent = self.get_alpha(x_[:, -1, :], 0)
        m_s, x_n = self.self_sparse_attention(x_, x_, x_, mask, alpha_ent) # pay attention the input
        alpha_global = self.get_alpha(m_s, 1)
        global_c = self.global_attention(m_s, x_n, x_s, mask, alpha_global)  # B, 1, dim
        h_t = global_c
        transformer_emb = self.get_transformer_embed(h_t, m_s) #  # 512, 100
        return transformer_emb




            













        

        


    



        


    


    









    
    



