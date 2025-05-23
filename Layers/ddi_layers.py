# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch
from scipy.stats import chi2
def concat_edge_embeddings(adj_matrix,embeddings):
    indices=torch.nonzero(adj_matrix)
    indices=indices[indices[:,1]<indices[:,2]]
    i,j=indices[:,1],indices[:,2]
    node1_embeddings=embeddings[:,i]
    node2_embeddings=embeddings[:,j]
    edge_embeddings = torch.cat((node1_embeddings, node2_embeddings), dim=2)
    edge_embeddings = edge_embeddings.view(edge_embeddings.shape[1], -1)
    chi_square_distance_threshold = chi2.isf(q=0.95, df=edge_embeddings.shape[0] - (
                edge_embeddings.shape[1] + edge_embeddings.shape[0] + 1) / 2)
    location_layer = torch.mean(edge_embeddings, dim=0)  # location layer
    scatter_layer = torch.matmul((edge_embeddings - location_layer).T,
                                 edge_embeddings - location_layer) / (edge_embeddings.shape[0] - 1)
    # scatter_layer
    inv_conv = scatter_layer.inverse()
    total_maha_distance = torch.matmul(torch.matmul(edge_embeddings - location_layer, inv_conv),
                                       (edge_embeddings - location_layer).T)
    loss = torch.sum(torch.nn.ReLU()(total_maha_distance.diag() -
                                     chi_square_distance_threshold)) / edge_embeddings.shape[
               0]  # chi_square based loss function
    return loss
class mah_distance_loss(torch.nn.Module):
    def __init__(self):
        super(mah_distance_loss, self).__init__()
    def forward(self, d, adj):
        loss = concat_edge_embeddings(d,adj)
        return loss

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
        return (beta * z).sum(1), beta

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
        return (beta * z).sum(1), beta

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
        return (beta * z).sum(1), beta

class MLP(nn.Sequential):
    def __init__(self, hidden_dim, num_layers, dropout=0.5):
        def build_block(input_dim, output_dim):
            return [
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]

        m = build_block(hidden_dim, 2 * hidden_dim)
        for i in range(1, num_layers - 1):
            m += build_block(2 * hidden_dim, 2 * hidden_dim)
        m.append(nn.Linear(2 * hidden_dim, hidden_dim))

        super().__init__(*m)

class GINConv(MessagePassing):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.mlp = MLP(hidden_dim, num_layers, 0.0)

    def forward(self, x, edge_index, size=None):
        out = self.propagate(edge_index, x=(x, x), size=size)
        out = x + self.mlp(out)
        return out

    def message(self, x_j):
        return x_j

class SubExtractor(nn.Module):
    def __init__(self, hidden_dim, num_clusters, residual=False):
        super().__init__()

        self.Q = nn.Parameter(torch.Tensor(1, num_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.Q)

        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.W_O = nn.Linear(hidden_dim, hidden_dim)

        self.residual = residual

    def forward(self, x, batch):
        K = self.W_K(x)
        V = self.W_V(x)

        K, mask = to_dense_batch(K, batch)
        V, _ = to_dense_batch(V, batch)

        attn_mask = (~mask).float().unsqueeze(1)
        attn_mask = attn_mask * (-1e9)

        Q = self.Q.tile(K.size(0), 1, 1)
        Q = self.W_Q(Q)

        A = Q @ K.transpose(-1, -2) / (Q.size(-1) ** 0.5)
        A = A + attn_mask
        A = A.softmax(dim=-2)

        threshold = 1
        top_values, _ = torch.topk(A.reshape(-1), int(A.numel() * threshold), sorted=False)
        mask = A >= top_values.min()
        A = A * mask
        out = Q + A @ V

        if self.residual:
            out = out + self.W_O(out).relu()
        else:
            out = self.W_O(out).relu()

        return out, A.detach().argmax(dim=-2), mask
class InterExtractor(nn.Module):
    def __init__(self, hidden_dim, num_clusters, residual=False):
        super().__init__()
        # 初始化查询矩阵
        self.Q = nn.Parameter(torch.randn(1, num_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.Q)

        # 定义线性变换
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.W_O = nn.Linear(hidden_dim, hidden_dim)
        self.residual = residual

    def forward(self, x1, x2, batch1, batch2):
        # 对第一个药物计算键和值
        K1, V1 = self.W_K(x1), self.W_V(x1)
        K1, mask1 = to_dense_batch(K1, batch1)
        V1, _ = to_dense_batch(V1, batch1)

        # 对第二个药物计算键和值
        K2, V2 = self.W_K(x2), self.W_V(x2)
        K2, mask2 = to_dense_batch(K2, batch2)
        V2, _ = to_dense_batch(V2, batch2)

        # 生成查询
        Q1 = self.W_Q(self.Q.tile(K1.size(0), 1, 1))
        Q2 = self.W_Q(self.Q.tile(K2.size(0), 1, 1))

        # 计算注意力分数
        A1 = torch.softmax(Q1 @ K2.transpose(-1, -2) / (Q1.size(-1) ** 0.5), dim=-2)
        A2 = torch.softmax(Q2 @ K1.transpose(-1, -2) / (Q2.size(-1) ** 0.5), dim=-2)

        # 计算输出
        threshold = 0.75  # 根据需要调整阈值

        top_values1, _ = torch.topk(A1.reshape(-1), int(A1.numel() * threshold), sorted=False)
        top_values2, _ = torch.topk(A2.reshape(-1), int(A2.numel() * threshold), sorted=False)

        mask1 = A1 >= top_values1.min()
        mask2 = A2 >= top_values2.min()
        A1 = A1 * mask1
        A2 = A2 * mask2

        out1 = A1 @ V2
        out2 = A2 @ V1

        out1 = self.W_O(out1).relu()
        out2 = self.W_O(out2).relu()

        return out1, out2