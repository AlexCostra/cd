# -*- coding: utf-8 -*-

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_add_pool

from ddi_layers import GINConv, MLP, SubExtractor, InterExtractor
class Attention1(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Attention1, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z), beta
class Attention2(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Attention2, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z), beta
class Attention3(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Attention3, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z), beta

class Attention4(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(Attention4, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z), beta
def get_gnn_model(args):
    gnn_model = args.gnn_model
    hidden_dim = args.hidden_dim
    gnn_num_layers = args.gnn_num_layers

    if gnn_model == "GCN":
        return nn.ModuleList([GCNConv(
            hidden_dim, hidden_dim
        ) for i in range(gnn_num_layers)])

    elif gnn_model == "GAT":
        gat_num_heads = args.gat_num_heads
        gat_to_concat = args.gat_to_concat

        if gat_to_concat:
            return nn.ModuleList([GATConv(
                hidden_dim, hidden_dim // gat_num_heads, gat_num_heads
            ) for i in range(gnn_num_layers)])
        else:
            return nn.ModuleList([GATConv(
                hidden_dim, hidden_dim, gat_num_heads, False
            ) for i in range(gnn_num_layers)])

    elif gnn_model == "GIN":
        gin_nn_layers = args.gin_nn_layers
        return nn.ModuleList([GINConv(
            hidden_dim, gin_nn_layers
        ) for i in range(gnn_num_layers)])

    else:
        raise NotImplementedError("Unsupported GNN Model")


class InteractionPredictor(nn.Module):
    def __init__(self, args):
        super().__init__()

        hidden_dim = args.hidden_dim
        num_node_feats = args.num_node_feats
        num_ddi_types = args.num_ddi_types
        pred_mlp_layers = args.pred_mlp_layers

        num_patterns = args.num_patterns
        dropout = args.dropout
        self.sub_drop_freq = args.sub_drop_freq
        self.sub_drop_mode = args.sub_drop_mode
        self.device = args.device

        self.hidden_dim = hidden_dim
        self.num_patterns = num_patterns
        self.num_ddi_types = num_ddi_types
        self.nn1 = nn.Sequential(
            nn.Linear(3600, 256),
            nn.BatchNorm1d(256),
            nn.Sigmoid(),
            nn.Dropout(dropout)

        )
        self.attention1=Attention1(128,64)
        self.attention2 = Attention2(128, 64)
        self.attention3 = Attention3(128, 64)
        self.attention4 = Attention4(128, 64)
        self.p=nn.Linear(64, 1)
        self.node_fc = nn.Linear(num_node_feats, hidden_dim)
        self.gnn = get_gnn_model(args)
        self.pool = SubExtractor(hidden_dim, num_patterns, args.attn_out_residual)
        self.inter = InterExtractor(hidden_dim, num_patterns, args.attn_out_residual)
        if args.dataset == "drugbank":
            self.mlp = nn.Sequential(
                nn.Linear(128, hidden_dim),
                MLP(hidden_dim, pred_mlp_layers, dropout),
                nn.Linear(hidden_dim, 64)#原始参数是1
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(128, hidden_dim),  #
                MLP(hidden_dim, pred_mlp_layers, dropout),
                nn.Linear(hidden_dim, 64)#原始参数是1
            )

        self.drop_rate_list = []

        if args.dataset != "drugbank":
            self.forward_func = self.forward_wo_type
        elif args.inductive:
            self.forward_func = self.forward_inductive
        else:
            self.forward_func = self.forward_transductive

    def do_gnn(self, x, edge_index):
        for gnn in self.gnn:
            x = gnn(x, edge_index)

        return x

    @torch.no_grad()
    def get_sub_to_drop(self, sub_cnt):
        if self.sub_drop_mode == "rand_per_graph":
            drop_prob = torch.rand(sub_cnt.size(0), self.num_patterns).to(self.device)
            drop_prob[sub_cnt == 0] = 0
            sub_to_drop = drop_prob.argmax(dim=-1, keepdim=True)

        elif self.sub_drop_mode == "rand_per_batch":
            sub_total = sub_cnt.sum(dim=0)
            # (num_patterns, )
            sub_non_zero = sub_total.nonzero().squeeze(-1).tolist()
            sub_to_drop = random.choice(sub_non_zero)

        elif self.sub_drop_mode == "smallest":
            sub_cnt[sub_cnt == 0] = sub_cnt.max().item() + 1
            sub_to_drop = sub_cnt.argmin(dim=-1, keepdim=True)
            # (batch_size, 1)

        elif self.sub_drop_mode == "biggest":
            sub_to_drop = sub_cnt.argmax(dim=-1, keepdim=True)

        else:
            raise NotImplementedError("Unsupported Sub drop mode")

        return sub_to_drop

    def compute_mahalanobis_distance(self, x, mu, sigma_inv):
        """计算马氏距离"""
        delta = x - mu  # 计算偏差
        dist = torch.sqrt(torch.sum(delta @ sigma_inv * delta, dim=1))  # 按行计算距离
        return dist

    @torch.no_grad()
    def do_sub_drop(self, x, edge_index, batch):
        num_nodes = x.size(0)
        if num_nodes == 1:
            return x
        h = self.node_fc(x)
        h = self.do_gnn(h, edge_index)
        _, sub_assign, mask = self.pool(h, batch)
        sub_assign[~mask] = self.num_patterns
        sub_one_hot = F.one_hot(sub_assign, self.num_patterns + 1)
        sub_cnt = sub_one_hot.sum(dim=1)[:, : self.num_patterns]
        sub_to_drop = self.get_sub_to_drop(sub_cnt)
        drop_mask = (sub_assign == sub_to_drop)
        drop_mask = drop_mask[mask]
        x[drop_mask] = 0
        drop_rate = drop_mask.sum().item() / drop_mask.size(0)
        return x, drop_rate

    @torch.no_grad()
    def do_node_drop(self, x, drop_rate=0.2):
        if x.size(0) == 1:
            return x
        prob = torch.rand(x.size(0), 1).to(self.device)
        drop_mask = (prob < drop_rate)
        keep_mask = ~drop_mask
        keep_mask = keep_mask.float()
        x_out = keep_mask * x
        return x_out, drop_rate
    def encode_graph(self, graph_batch_1, graph_batch_2):
        to_drop = False
        inter1, inter2, interaction_features_1, interaction_features_2 = self.process_drug_interactions(graph_batch_1,
                                                                                                        graph_batch_2)
        if self.training:
            if self.sub_drop_freq == "half":
                to_drop = (random.random() > 0.5)
            elif self.sub_drop_freq == "always":
                to_drop = True
        x1 = graph_batch_1.x
        edge_index_1 = graph_batch_1.edge_index
        batch1 = graph_batch_1.batch
        if to_drop:
            x, drop_rate = self.do_sub_drop(x1, edge_index_1, batch1)
            self.drop_rate_list.append(drop_rate)
        x1 = self.node_fc(x1)
        x1 = self.do_gnn(x1, edge_index_1)
        out1 = global_add_pool(x1, batch1)
        pool1, *_ = self.pool(x1, batch1)
        pool1 = F.normalize(pool1, dim=-1)
        to_drop = False
        if self.training:
            if self.sub_drop_freq == "half":
                to_drop = (random.random() > 0.5)
            elif self.sub_drop_freq == "always":
                to_drop = True
        x2 = graph_batch_2.x
        edge_index_2 = graph_batch_2.edge_index
        batch2 = graph_batch_2.batch
        if to_drop:
            x, drop_rate = self.do_sub_drop(x2, edge_index_2, batch2)
            self.drop_rate_list.append(drop_rate)
        x2 = self.node_fc(x2)
        x2 = self.do_gnn(x2, edge_index_2)
        out2 = global_add_pool(x2, batch2)
        pool2, *_ = self.pool(x2, batch2)
        pool2 = F.normalize(pool2, dim=-1)
        return out1, pool1, out2, pool2, inter1, inter2, interaction_features_1, interaction_features_2

    def process_drug_interactions(self, graph_batch_1, graph_batch_2):
        x1 = graph_batch_1.x
        edge_index_1 = graph_batch_1.edge_index
        batch1 = graph_batch_1.batch
        x1 = self.node_fc(x1)
        x1 = self.do_gnn(x1, edge_index_1)
        x2 = graph_batch_2.x
        edge_index_2 = graph_batch_2.edge_index
        batch2 = graph_batch_2.batch
        x2 = self.node_fc(x2)
        x2 = self.do_gnn(x2, edge_index_2)
        interaction_features_1, interaction_features_2 = self.inter(x1, x2, batch1, batch2)
        interaction_features_1_mean = interaction_features_1.mean(dim=1)  # 结果维度 [256, 128]
        interaction_features_2_mean = interaction_features_2.mean(dim=1)  # 结果维度 [256, 128]
        mean1 = interaction_features_1_mean.mean(dim=1, keepdim=True)
        std1 = interaction_features_1_mean.std(dim=1, keepdim=True)
        mean2 = interaction_features_2_mean.mean(dim=1, keepdim=True)
        std2 = interaction_features_2_mean.std(dim=1, keepdim=True)
        inter1 = (interaction_features_1_mean - mean1) / std1
        inter2 = (interaction_features_2_mean - mean2) / std2
        interaction_features_1=interaction_features_1_mean
        interaction_features_2=interaction_features_2_mean
        return inter1, inter2, interaction_features_1, interaction_features_2

    def forward_wo_type(self, graph_batch_1, graph_batch_2, dummy_ddi_type, cnm, cnm1,mode=1):
        # Step 1: 图特征提取
        out1, pool1, out2, pool2, inter1, inter2, interaction_features_1, interaction_features_2 = self.encode_graph(
            graph_batch_1, graph_batch_2
        )
        pool1, _ = torch.max(pool1, dim=1)  # 对 pool1 提取最大值
        pool2, _ = torch.max(pool2, dim=1)  # 对 pool2 提取最大值

        # Step 2: 注意力机制处理
        pool1, _ = self.attention3(pool1)  # 处理 pool1
        pool2, _ = self.attention4(pool2)  # 处理 pool2
        inter11, _ = self.attention1(inter1)  # inter1 注意力处理
        inter21, _ = self.attention2(inter2)  # inter2 注意力处理

        # Step 3: 合并特征张量
        out = inter1 + inter2 + pool1 + pool2  # 将处理后的特征张量相加
        out= self.mlp(out)
        if mode==1:
            x111 = out[cnm]  # 正常样本的特征

            # Step 4: 计算均值和协方差矩阵
            mu = torch.mean(x111, dim=0)  # 正常样本的均值
            cov_matrix = torch.cov(x111.T)
            cov_matrix += 1e-6 * torch.eye(cov_matrix.size(0))
            sigma_inv = torch.inverse(cov_matrix)
            score1 = self.compute_mahalanobis_distance(out, mu, sigma_inv)
            self.mu=mu
            self.sigma_inv=sigma_inv
        else:
            score1 = self.compute_mahalanobis_distance(out, self.mu, self.sigma_inv)
        score = self.p(out).squeeze(-1)

        return score,score1


