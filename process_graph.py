import pickle
import argparse

# # # train_seqs是一个二维list,[[1,52,38]，[957,25,87,98,57,45],[21,33]...]表示每个session中的item的被赋予的id
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='diginetica/Tmall/Nowplaying')
parser.add_argument('--sample_num', type=int, default=12)

args = parser.parse_args()

dataset = args.dataset
sample_num = args.sample_num

seq = pickle.load(open('Dual_contrastive/datasets/'+ dataset +'/all_train_seq.txt', 'rb'))

# print(seq)
if args.dataset == 'diginetica':
    num_nodes = 43098

elif args.dataset == 'Tmall':
    num_nodes = 40728

elif args.dataset == 'Nowplaying':
    num_nodes = 60417
else:
    num_nodes = 310

relation = []
neighbor = [] * num_nodes

all_test = set()

adj1 = [dict() for _ in range(num_nodes)]
adj = [dict() for _ in range(num_nodes)]
weight = [[] for _ in range(num_nodes)]

for i in range(len(seq)): # 相邻K个item之间进行关系提取
    data = seq[i]
    for k in range(1, 4):
        for j in range(len(data)-k):
            relation.append([data[j], data[j+k]])
            relation.append([data[j+k], data[j]])

for tup in relation:
    if tup[1] in adj1[tup[0]].keys():
        adj1[tup[0]][tup[1]] += 1
    else:
        adj1[tup[0]][tup[1]] = 1
# print(adj1[15])
"""
{14: 1, 13: 1, 12: 1, 10635: 2, 13001: 3, 
379: 6, 18008: 3, 20194: 1, 15: 2, 16436: 1, 
10576: 2, 11351: 2, 7158: 2, 17756: 2, 26905: 1, 
13859: 1, 7941: 1, 20786: 2, 5237: 1, 10: 3, 28793: 1, 
13061: 1, 2703: 1, 8416: 1, 25037: 1, 7940: 2, 39359: 1,
 1267: 1, 22146: 4, 2770: 1, 28458: 1, 10582: 1, 16929: 1, 
 1814: 1, 19617: 1, 3867: 1}
"""
for t in range(num_nodes):
    x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
    adj[t] = [v[0] for v in x ]
    weight[t] = [v[1] for v in x]

# print('adj', adj[15])
"""
adj [379, 22146, 13001, 18008, 10, 
10635, 15, 10576, 11351, 7158, 17756, 
20786, 7940, 14, 13, 12, 20194, 16436, 
26905, 13859, 7941, 5237, 28793, 13061, 
2703, 8416, 25037, 39359, 1267, 2770, 
28458, 10582, 16929, 1814, 19617, 3867]
"""
for i in range(num_nodes):
    adj[i] = adj[i][:sample_num]
    weight[i] = weight[i][:sample_num]

pickle.dump(adj, open('Dual_contrastive/datasets/' + dataset + '/adj_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(weight, open('Dual_contrastive/datasets/' + dataset + '/num_' + str(sample_num) + '.pkl', 'wb'))

