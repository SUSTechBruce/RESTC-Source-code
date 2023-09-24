"""
Using self supervised loss to assist GCN and Transformer fusion 
Using k-hop collaborative item embedding
Dual fusion model: Combine transformer sequence model and GAT spacial model 
one model to learn temporal information of session data, and other GAT model learns the structual and spacial information of data

"""
from torch.nn.modules.activation import Softmax
from tqdm import tqdm
import datetime
import math
import torch
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
from torch.utils.data.distributed import DistributedSampler

class GAT(nn.Module):
    """
    Graph attention network
    """

    def __init__(self, dim, alpha, dropout=0., name=None):
        super(GAT, self).__init__()
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
        alpha = torch.where(adj.eq(1), e_0, mask) # 
        alpha = torch.where(adj.eq(2), e_1, alpha) # 建模session 的adj matrix
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)

        return output

class Graph_embedding(nn.Module):
    """
    all items embedding through GCN to get global graph embedding
    improvement ideas：
    增加 relu/ leakyRelu激活函数
    增加 resnet网络结构查看效果
    """
    def __init__(self, layer_nums, dim=100):
        super(Graph_embedding, self).__init__()
        self.dim = dim
        self.layer_nums = layer_nums
        self.layers = {}
        self.activate_1 = nn.ReLU()
        self.activate_2 = nn.LeakyReLU(0.2)
        self.alpha = 0.2
        for i in range(self.layer_nums):
            self.layers['layer_%d' % (i)] = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, global_graph, nodes_feature):

        degree = global_graph.sum(axis=0).reshape(1, -1)

        degree = 1/degree
        graph_matrix = global_graph.multiply(degree) # A/D

        values = graph_matrix.data
        indices = np.vstack((graph_matrix.row, graph_matrix.col))
        indexs = torch.LongTensor(indices)
        values = torch.FloatTensor(values)
        shape = graph_matrix.shape
        graph_matrix = torch.sparse.FloatTensor(indexs, values, torch.Size(shape)) # ([43097, 43097])

        graph_embeddings = nodes_feature # [43097, dim]
        # embedding_list = [nodes_feature]
        for i in range(self.layer_nums):
            graph_embeddings = trans_to_cuda(self.layers['layer_%d' % (i)])(graph_embeddings) # X = AXW/D 谱图卷积神经网络 self.adacency已经包含A/D， A = A + I
            graph_embeddings = torch.sparse.mm(trans_to_cuda(graph_matrix), graph_embeddings)
            graph_embeddings = F.normalize(graph_embeddings, dim=-1, p=2)
            graph_embeddings = self.activate_1(graph_embeddings)
        graph_emb = graph_embeddings

        return graph_emb # [43097, dim]


# model = trans_to_cuda(DualContrastiveModel(args, num_nodes, adj, num))
class DualSSlGloablModel(torch.nn.Module):

    def __init__(self, args, global_graph, num_nodes, adj_all, weight_all):
        super(DualSSlGloablModel, self).__init__()
        self.args = args
        self.num_nodes = num_nodes
        self.adj_all = adj_all
        self.weight_all = weight_all
        self.batch_size = args.batch_size
        self.dim = args.hiddenSize
        self.dropout_local = args.dropout_local
        self.dropout_global = args.dropout_global
        self.k_hop = args.n_iter
        self.sample_num = args.n_sample
        self.beta = 0 # beta: 0.005 for diginetica  0.00025 for Tmall ssl task maginitude  
        self.global_graph = global_graph # all nodes graph in session based datasets

        # constrastive learning sampling
        self.K = 20
        self.num = 5000
        
        # Global graph embedding 

        self.layer_nums = 3
        self.graph_emb = Graph_embedding(self.layer_nums, self.dim)

        # GAT module
        self.GAT_model = GAT(self.dim, args.alpha, dropout=0.0)
  
        ## transformer layer module
        is_sparse = False
        n_layers = 2
        n_heads = 5
        item_dim = self.dim
        pos_dim = self.dim
        # n_items = 43098
        n_pos = 300
        w = 20
        dropout = 0.5
        activate = 'relu'
        self.transformer_emb = MHSparseTransformer(n_layers, n_heads, item_dim, pos_dim, self.num_nodes, n_pos, w, 
                          dropout=dropout, activate=activate, sparse=is_sparse) #  # 512, 1, 100
        
        # item representation and position encoding
        self.embedding = nn.Embedding(self.num_nodes, self.dim) # [43097, 100]
        self.pos_embedding = nn.Embedding(300, self.dim) 

        # parameter of DualContrastiveModel
        self.w_1 = nn.Parameter(torch.Tensor(self.dim * 2, self.dim))
        self.w_3_dim = nn.Parameter(torch.Tensor(self.dim * 3, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        
        self.w_3 = nn.Parameter(torch.Tensor(self.dim * 2, self.dim))
        self.w_4 = nn.Parameter(torch.Tensor(self.dim, 1))

        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)

        self.glu1_graph = nn.Linear(self.dim, self.dim)
        self.glu2_graph = nn.Linear(self.dim, self.dim, bias=False)

        self.linear_transform = nn.Linear(self.dim, self.dim, bias=False)

        self.leakyrelu = nn.LeakyReLU(args.alpha)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.l2)
        self.scheduler_stepLR = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
        self.scheduler_stepLR_1 = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.9)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        self.scheduler_crosineAnneal = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5)

        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)

        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
    
    # 总的transformer encoding作为序列信息在  hs = hs / torch.sum(mask, 1) 替换
    # 提取transformer序列表征，在scores = DualContrastiveModel.compute_scores(seq_hidden, mask)前concate


    def compute_loss_func(self, targets, reversed_sess_item, gat_hidden, mask, transformer_emb, trans_item_embedding, is_train=True):
        """
        transformer_emb: 100, dim
        """

        batch = gat_hidden.shape[0] # batch seq dim
        # 获取获取 session 的 k-hop邻接协同过滤信息表征 collaborative items embeddings
        global_graph = self.global_graph
        nodes_features = self.embedding.weight # 注意 区分 index=0 的情况 [43098， 43098] # 其实最大的点是1-43097
        global_graph_embedding = self.graph_emb(global_graph, nodes_features)  # [43098, 43098]
        
        zero_pads = torch.cuda.FloatTensor(1, self.dim).fill_(0)
        item_embeddings = torch.cat([zero_pads, global_graph_embedding], dim=0) # [43099, 43099]
        get = lambda i: item_embeddings[reversed_sess_item[i]] # 根据reversed session的index顺序从GCN items表征中选取 reversed item的表征
        k_hop_hidden = torch.cuda.FloatTensor(batch, list(reversed_sess_item.shape)[1], self.dim).fill_(0)
        for i in torch.arange(batch):
            k_hop_hidden[i] = get(i)
        
    
        # graph_hidden = gat_hidden + k_hop_hidden # add or concat
        graph_hidden = gat_hidden
        # 处理mask和获取 learnble position embedding
        mask = mask.float().unsqueeze(-1)
        batch_size = graph_hidden.shape[0]
        len = graph_hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        # 获取transformer的表征向量和原始节点的 nn.embedding.weight [batch, dim]，[n_nodes, dim]
        trans_items_emb = trans_item_embedding 
        trans_emb = transformer_emb
        

        transformer_emb = transformer_emb.unsqueeze(-2).repeat(1, len, 1) # 100, 100 -> 100, seq, 100
        if self.args.dataset == 'Nowplaying': # Nowplaying  # 300 300 0.0002
            graph_hidden = graph_hidden  # + k_hop_hidden
            concat_hs_pos = torch.cat([pos_emb, graph_hidden], -1) # concat in dimension
        else:

            concat_hs_pos = torch.cat([pos_emb, k_hop_hidden, graph_hidden], -1) # concat in dimension


        # 获取GCN的表征向量和原始节点 self.embedding.weight[1:]
        hs = torch.sum(graph_hidden * mask, -2) # 第二个维度相加[batch, sequence, hidden]
        hs = hs / torch.sum(mask, 1)  # batch, 1, seq 100, 1, 69
        hs = hs.unsqueeze(-2).repeat(1, len, 1) # s' # 插入transformer 时序表征

        if self.args.dataset == 'Nowplaying': # Nowplaying
            nh = torch.matmul(concat_hs_pos, self.w_1)
        else:
            nh = torch.matmul(concat_hs_pos, self.w_3_dim) # 此nh为graph_hidden的nh       
        nh = torch.tanh(nh) # (11)

        # 获取transformer emb + GAT emb的全局soft attention fusion表征
        trans_nh = torch.sigmoid(self.glu1(nh) + self.glu2(transformer_emb)) # (13)
        beta = torch.matmul(trans_nh, self.w_2)
        beta = beta * mask

        # concat gat and k-hop embedding
        if self.args.dataset == 'Tmall' :
            final_hidden = gat_hidden
        else:
            final_hidden = gat_hidden + k_hop_hidden # 试一试是否无敌

        select = torch.sum(beta * final_hidden, 1) # (14) # 此处使用concat信息表征

        final_item_emb = self.embedding.weight[1:]

        # 使用nn.Embedding获取的itmems表征
        scores = torch.matmul(select, final_item_emb.transpose(1, 0)) # (15) [batch ,seq, dim] * [batch, dim, seq]
        
        # 获取推断损失
        targets = trans_to_cuda(targets).long()
        labels = targets - 1
############################################## 对比学习部分代码 ##########################################################

        gat_hs = torch.sum(gat_hidden * mask, -2) # 第二个维度相加[batch, sequence, hidden]
        gat_hs = gat_hs / torch.sum(mask, 1)  # batch, 1, seq 100, 1, 69
        gat_hs = gat_hs.unsqueeze(-2).repeat(1, len, 1) # s' # 插入transformer 时序表征
        
        gat_concat_hs_pos = torch.cat([pos_emb, gat_hidden], -1) # concat in dimension
        gat_nh = torch.matmul(gat_concat_hs_pos, self.w_3) # 考虑 w1参数换掉
        gat_nh = torch.tanh(gat_nh) # (11)

        # 小小改动 注意！
        GAT_emb = torch.sigmoid(self.glu1_graph(gat_nh) + self.glu2_graph(gat_hs)) # 直接用gat hidden做表征
        # GAT_emb = torch.sigmoid(self.glu1(nh) + self.glu2(hs)) # 用k-hop hidden和gat hidden 相加后的表征作为 GAT_emb
        GAT_emb_beta = torch.matmul(GAT_emb, self.w_4) # 考虑 w2参数换掉
        GAT_emb_beta = GAT_emb_beta * mask  # element mul
        GAT_emb = torch.sum(GAT_emb_beta * graph_hidden, 1)
        if is_train:
            contrast_loss = self.dual_embs_SSL(GAT_emb, trans_emb)
            contrast_loss = self.beta * contrast_loss
            dual_fustion_loss = self.loss_function(scores, labels)
            # contrast_loss = 0
        else:
            dual_fustion_loss = 0
            contrast_loss = 0

        return scores, dual_fustion_loss, contrast_loss
   
    def dual_embs_SSL(self, gat_emb, trans_emb):
        """
        Batch sample constrastive learning 
        """
        def row_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            return corrupted_embedding
        def row_column_shuffle(embedding):
            corrupted_embedding = embedding[torch.randperm(embedding.size()[0])]
            corrupted_embedding = corrupted_embedding[:, torch.randperm(corrupted_embedding.size()[1])]
            return corrupted_embedding
        def score(x1, x2):
            return torch.sum(torch.mul(x1, x2), 1)

        pos = score(gat_emb, trans_emb)
        neg1 = score(trans_emb, row_column_shuffle(gat_emb))
        one = torch.cuda.FloatTensor(neg1.shape[0]).fill_(1)
        # one = zeros = torch.ones(neg1.shape[0])
        con_loss = torch.sum(-torch.log(1e-8 + torch.sigmoid(pos))-torch.log(1e-8 + (one - torch.sigmoid(neg1))))
        return con_loss
       

    def forward(self, inputs, adj, mask_item, item, alias_inputs, targets, transformer_seqs, transformer_pos, is_train=True):
        """
        forward function for pytorch Datadistribution
        input: items, adj, mask, inputs, alias_inputs, transformer_seqs, transformer_pos, is_train=is_train
        """
        h = self.embedding(inputs)  # inputs == items
        # GAT emb
        gat_emb = self.GAT_model(h, adj, mask_item) #  

        gat_hidden = F.dropout(gat_emb, self.dropout_local, training=self.training)
    
        ## compute transformer embed
        transformer_emb, trans_item_embeddings = self.transformer_emb(transformer_seqs, transformer_pos) # 512 100
        get = lambda index: gat_hidden[index][alias_inputs[index]] # index 就是 batch
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()]) #  根据单个session在unique session的index转换的seq embedding
        reversed_sess_item = item
        mask = mask_item
        scores, dual_fusion_loss, contrast_loss = self.compute_loss_func(targets, reversed_sess_item, seq_hidden, mask, transformer_emb, trans_item_embeddings, is_train=is_train)
        output = {}
        output['scores'] = scores
        output['dual_fusion_loss'] = dual_fusion_loss
        output['contrast_loss'] = contrast_loss
        output['total_loss'] = dual_fusion_loss + contrast_loss

        return output
# /home/intern2/NunoPesikin/anaconda3/envs/wan/lib/python3.7/site-packages (from scipy) (1.21.2)
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
 
def train_and_test(local_rank, epoch, args, model, train_data, test_data):
    """
    alias_input: 单个session在 unique session中的index
    adj: 单个session构成的领接矩阵
    items：unique session + 0 padding
    mask：单个session，存在item为1，其余补位0
    targets：label
    inputs： 单个反转session + (maxlen  -session len)补位0
    """
    print('start to training', datetime.datetime.now())
    model.train()
    total_loss = 0.0
    train_sampler = DistributedSampler(train_data).set_epoch(epoch)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=int(model.batch_size/4), sampler=train_sampler, shuffle=False, pin_memory=True)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)

    count = 0
    model = model.module if hasattr(model, "module") else model
    for data in train_loader: 

        alias_inputs, adj, items, mask, targets, inputs, transformer_seqs, transformer_pos = data
        alias_inputs = trans_to_cuda(alias_inputs).long()
        items = trans_to_cuda(items).long()
        adj = trans_to_cuda(adj).float()
        mask = trans_to_cuda(mask).long()
        inputs = trans_to_cuda(inputs).long()
        transformer_seqs = trans_to_cuda(transformer_seqs).long()
        transformer_pos = trans_to_cuda(transformer_pos).long()
        targets = trans_to_cuda(targets).long()
        count += 1
        model.optimizer.zero_grad()
        # self, inputs, adj, mask_item, item, alias_inputs, targets, transformer_seqs, transformer_pos, is_train=True
        outputs = model(items, adj, mask, inputs, alias_inputs, targets, transformer_seqs, transformer_pos, is_train=True)
        # targets, scores, dual_fustion_loss, contrast_loss = forward(model, data, is_train=True) # 在forward里计算loss并返回最后的总的loss
        # loss = model.loss_function(scores, targets - 1)
        dual_fusion_loss = outputs['dual_fusion_loss']
        contrast_loss = outputs['contrast_loss']
        if count % 500 == 0 and args.local_rank == 0:
            print('The current local rank', args.local_rank)
            print('The train model loss is: {} The contrastive loss is: {}'.format(dual_fusion_loss, contrast_loss))
        loss = outputs['total_loss']
        loss = loss.mean()
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()
    print('\tLoss:\t%.3f'% total_loss)
    if epoch < 11:
        model.scheduler_stepLR.step()
    else:
        model.scheduler_crosineAnneal.step()

    print('Start to predict test data', datetime.datetime.now())
    model.eval()
    test_loader = Data.DataLoader(test_data, batch_size=int(model.batch_size/4),shuffle=False, pin_memory=True)
    result = []
    hit_20, mrr_20, hit_10, mrr_10 = [], [], [], []
    
    # 测试，并计算召回率和mrr rate
    for data in test_loader: # 此处要改，注意区分训练和推断测试的区别
        # targets, scores, dual_fustion_loss, contrast_loss = forward(model, data, is_train=False)
        alias_inputs, adj, items, mask, targets, inputs, transformer_seqs, transformer_pos = data
        alias_inputs = trans_to_cuda(alias_inputs).long()
        items = trans_to_cuda(items).long()
        adj = trans_to_cuda(adj).float()
        mask = trans_to_cuda(mask).long()
        inputs = trans_to_cuda(inputs).long()
        transformer_seqs = trans_to_cuda(transformer_seqs).long()
        transformer_pos = trans_to_cuda(transformer_pos).long()
        targets = trans_to_cuda(targets).long()
        outputs = model(items, adj, mask, inputs, alias_inputs, targets, transformer_seqs, transformer_pos, is_train=False)
        scores = outputs['scores']
        sub_scores_20 = scores.topk(20)[1]
        sub_scores_10 = scores.topk(10)[1]
        sub_scores_20 = trans_to_cpu(sub_scores_20).detach().numpy()
        sub_scores_10 = trans_to_cpu(sub_scores_10).detach().numpy()
        targets = trans_to_cpu(targets).numpy()
        # 计算HR 20,10 和 MRR 20， 10
        for score, target, mask in zip(sub_scores_20, targets, test_data.mask):
            hit_20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_20.append(0)
            else:
                mrr_20.append(1 /(np.where(score == target - 1)[0][0] + 1))
     
        for score, target, mask in zip(sub_scores_10, targets, test_data.mask):
            hit_10.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_10.append(0)
            else:
                mrr_10.append(1 /(np.where(score == target - 1)[0][0] + 1))

    hit_20 = np.mean(hit_20) * 100
    mrr_20 = np.mean(mrr_20) * 100

    hit_10 = np.mean(hit_10) * 100
    mrr_10 = np.mean(mrr_10) * 100

    return hit_20, mrr_20, hit_10, mrr_10, total_loss


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention_dropout_rate = attention_dropout_rate
        self.dim = dim

        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attention_mlp = nn.Linear(self.dim, self.dim)
        
        self.linear_Q = nn.Linear(dim, num_heads * self.head_dim)
        self.linear_K = nn.Linear(dim, num_heads * self.head_dim)
        self.linear_V = nn.Linear(dim, num_heads * self.head_dim)

        self.att_dropout = nn.Dropout(self.attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * self.head_dim, self.dim)
    
    def forward(self, Q, K, V, seq_mask,  alpha_ent, sparse):
        orig_q_size = Q.size() # batch, seq, dim
        seq_len = Q.size(1)
        batch_size = Q.size(0)
        d_q = self.head_dim 
        d_k = self.head_dim
        d_v = self.head_dim

        Q = self.linear_Q(Q).view(batch_size, -1, self.num_heads, d_q) # batch, seq, head_num, head_dim
        K = self.linear_K(K).view(batch_size, -1, self.num_heads, d_k) 
        V = self.linear_V(V).view(batch_size, -1, self.num_heads, d_v)
        

        Q = Q.transpose(1, 2) 
        V = V.transpose(1, 2) # batch, h, q_len, d_k
        K = K.transpose(1, 2).transpose(2, 3) # batch, h ,dk, k_len
        # batch, h, q_len, q_len
        attention_matrix = torch.matmul(Q, K) / math.sqrt(self.dim)

        if seq_mask is not None:
            mask = seq_mask.unsqueeze(1).expand(-1, seq_len, -1) # batch, len, len
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1) # batch, h, len ,len
            attention_matrix = attention_matrix.masked_fill(mask == 0, -np.inf)
            if sparse == False:
                alpha = torch.softmax(attention_matrix, dim=-1)
            else:
                alpha = entmax_bisect(attention_matrix, alpha_ent, dim=-1)
        x = torch.matmul(alpha, V)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * d_v)
        assert x.size() == orig_q_size
        return x



class FeedForwardNetwork(nn.Module):
    def __init__(self, dim, ffn_dim):
        super(FeedForwardNetwork, self).__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.layer1 = nn.Linear(self.dim, self.ffn_dim)
        self.relu = torch.relu
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(self.ffn_dim, self.dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x) # try torch.relu
        x = self.layer2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(dim)
        self.self_attention = MultiHeadAttention(dim, attention_dropout_rate, num_heads)
        self.self_attention_dropout = nn.Dropout(dropout_rate)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForwardNetwork(dim, ffn_dim)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, seq_mask, alpha_ent, sparse):
        # y = self.self_attention_norm(x)                       # 传入sparse
        y = x
        y = self.self_attention(y, y, y, seq_mask,  alpha_ent, sparse) # sparse = False 
        y = self.self_attention_dropout(y)
        x = x + y

        # y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        x = self.ffn_norm(x)
        output = x
        return output


class MHSparseTransformer(nn.Module):

    def __init__(self, n_layers, n_heads, item_dim, pos_dim, n_items, n_pos, w, atten_way='dot', decoder_way='bilinear', dropout=0,
                 activate='relu', sparse=False):
        super(MHSparseTransformer, self).__init__()
        self.item_dim = item_dim
        self.pos_dim = pos_dim
        dim = item_dim + pos_dim
        self.dim = dim
        self.ffn_dim = dim
        self.n_pos = n_pos
        self.n_heads = n_heads  
        self.n_items = n_items  # 所有session unique items的总数
        self.embedding = nn.Embedding(n_items + 1, self.item_dim, padding_idx=0,max_norm=1.5)
        self.pos_embedding = nn.Embedding(self.n_pos, self.pos_dim, padding_idx=0, max_norm=1.5)
        self.atten_way = atten_way
        self.decoder_way = decoder_way
        self.atten_w0 = nn.Parameter(torch.Tensor(1, self.dim))
        self.atten_w1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_w2 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_bias = nn.Parameter(torch.Tensor(self.dim))
        self.w_f = nn.Linear(2*self.dim, self.item_dim)
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        self.self_atten_w1 = nn.Linear(self.dim, self.dim)
        self.self_atten_w2 = nn.Linear(self.dim, self.dim)
        
        self.LN = nn.LayerNorm(self.dim)
        self.LN2 = nn.LayerNorm(self.item_dim)
        self.is_dropout = True
        self.attention_mlp = nn.Linear(dim, dim) # the mlp of attention
        self.alpha_w = nn.Linear(dim, 1)
        self.w = w
        
        # new add
        self.sparse = sparse
        self.attention_dropout_rate = dropout
        # dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, seq_mask,  alpha_ent
        encoders = [EncoderLayer(self.dim, self.ffn_dim, self.dropout_rate, 
        self.attention_dropout_rate, self.n_heads) for _ in range(n_layers)]
        self.encoder_layers = nn.ModuleList(encoders)
        
        if activate == 'relu':
            self.activate = F.relu
        elif activate == 'selu':
            self.activate = F.selu

        self.initial_(initial_way='norm')

    def initial_(self, initial_way='xavier'):
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

    
    def get_alpha(self, session, seq_len, head_nums, flag):
        """"
        get alpha from sigmoid, the input is the last preference term of user.
        When the sigmoid output + 1.0001 = 1, the alpha = 1.0001 is too sparse 
        and need to be recorded
        input shape: [batch_size, dim]
        output shape: [batch_size, 70, 1]
        """     
        
        if flag == 0:
            alpha_ent = torch.sigmoid(self.alpha_w(session)) + 1
            # alpha_ent = self.add_value(alpha_ent).unsqueeze(1)
            mask = (alpha_ent == 1).float()
            alpha_ent = alpha_ent.masked_fill(mask == 1, 1.00001)
            alpha_ent = alpha_ent.unsqueeze(1)
            alpha_ent = alpha_ent.expand(-1, seq_len, -1)  # 单一维度扩展 70
            return alpha_ent
        if flag == 1:
            alpha_global = torch.sigmoid(self.alpha_w(session)) + 1
            mask = (alpha_global == 1).float()
            alpha_ent = alpha_global.masked_fill(mask == 1, 1.00001)
            return alpha_global
        if flag == 3:
            alpha_ent = torch.sigmoid(self.alpha_w(session)) + 1
            # alpha_ent = self.add_value(alpha_ent).unsqueeze(1)
            mask = (alpha_ent == 1).float()
            alpha_ent = alpha_ent.masked_fill(mask == 1, 1.00001)
            alpha_ent = alpha_ent.unsqueeze(1)
            alpha_ent = alpha_ent.expand(-1, seq_len, -1)  # 单一维度扩展 70
            alpha_ent = alpha_ent.unsqueeze(1)
            alpha_ent = alpha_ent.expand(-1, head_nums, -1, -1) # batch num, head_dim, seq_len, 1
           
            return alpha_ent

    def add_value(self, value):

        mask_value = (value ==1).float()
        value = value.masked_fill(mask_value == 1, 1.00001)
        return value
        
    def self_attention(self,X, seq_mask, alpha_ent):
        """
        multi-head self attention with alpha-entmax to be inplace of softmax
        """
        if self.sparse:
            for enc_layer in self.encoder_layers:
                X = enc_layer(X, seq_mask, alpha_ent, True)
        else:
            for enc_layer in self.encoder_layers:
                X = enc_layer(X, seq_mask, alpha_ent, False)
        session_s = X[:, -1, :].unsqueeze(1)
        session_x = X[:, :-1, :]

        return session_s, session_x # session and last item of session

 
    def global_attention(self, Q_hat, K_hat, V_hat, seq_mask=None, alpha_ent=1):

        """
        Q_hat: session_s  last one 
        K_hat: session_x
        """
        Q = Q_hat.matmul(self.atten_w2)
        K = K_hat.matmul(self.atten_w1)
        bias = self.atten_bias
        W_0 = self.atten_w0
        

        alpha = torch.relu(Q + K + bias)  # batch, seq, dim   直接使用这里？
        alpha = torch.matmul(alpha, W_0.t()) # batch, seq, 1   改这 batch seq dim
        if seq_mask is not None:
            seq_mask = seq_mask.unsqueeze(-1) # batch, seq, 1
            seq_mask = seq_mask[:, :-1, :]
            alpha = alpha.masked_fill(seq_mask == 0, -np.inf)
        alpha = entmax_bisect(alpha, alpha_ent, dim=1) 
        # alpha = torch.softmax(alpha, dim=1)
        c = torch.matmul(alpha.transpose(1, 2), V_hat) # (batch, 1, seq) * (batch, seq, dim)
        # print('c shape', c.shape)
        return c

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
        
        input_embed = torch.cat((x_embeddings, x_embeddings), 2)  # B seq, 2*dim
        x_s = input_embed[:, :-1, :]  # B, seq-1, 2*dim
        last_item =  input_embed[:, -1, :]
        # set seq_len
        seq_len = input_embed.shape[1]
        # set head_num
        head_nums = self.n_heads
        alpha_ent = self.get_alpha(last_item, seq_len, head_nums, 3)
        input_x = input_embed
        m_s, x_n = self.self_attention(input_x, mask, alpha_ent) # pay attention the input
        alpha_global = self.get_alpha(m_s, seq_len, head_nums, 1)
        global_c = self.global_attention(m_s, x_n, x_s, mask, alpha_global)  # B, 1, dim
        h_t = global_c
        result = self.get_transformer_embed(h_t, m_s)
        items_embeddings = self.embedding.weight[1: -1]/ torch.norm(self.embedding.weight[1:-1], dim=-1).unsqueeze(1)
        return result, items_embeddings # 去除第一个idx = 0的，去除最后一个transformer item
                        # 考虑一下L2 norm