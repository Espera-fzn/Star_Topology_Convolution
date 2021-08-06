import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import random

# =============================================================================
# =============================================================================
# This is a Pytorch implementation of Star Topology Convolution.
# =============================================================================
# =============================================================================
# 
# Copyright (c) 2020, Chong WU & Zhenan FENG All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# 
# Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# 
# Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 
# Neither the name of City University of Hong Kong nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# =============================================================================
# =============================================================================
# Please cite our paper if you use this code in your own work:
# Wu, Chong; Feng, Zhenan; Zheng, Jiangbin; Zhang, Houwang; Cao, Jiawang; YAN, Hong (2020): Star Topology Convolution for Graph Representation Learning. TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.12805799.v4 
# =============================================================================
# =============================================================================


class STC_encoder(nn.Module):
    """
    Encodes a node's using Star Topology Convolution (STC)
    """
    def __init__(self, features, feature_dim, embedding_dim, adjacent_list, STC_layer, device, dropout_param, dropout_a, filtersize=10, base_model = None): 
        super(STC_encoder, self).__init__()
        """
        features        -- function mapping LongTensor of node ids to FloatTensor of feature values
        feature_dim     -- feature dimension
        embedding_dim   -- embedding dimension
        adjacent_list   -- adjacent list
        STC_layer       -- Star Topology Convolution (STC) layer
        device          -- GPU or CPU
        dropout_param   -- Dropout rate
        dropout_a       -- Use dropout or not
        filtersize      -- number of neighbors to sample
        """
        self.features = features
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.adjacent_list = adjacent_list
        self.STC_layer = STC_layer
        self.device = device  
        self.dropout = dropout_a
        if self.dropout == True:
            self.dropout_param = dropout_param
        if base_model != None:
            self.base_model = base_model
        self.filter_size = filtersize
        self.detaching_weight = nn.Parameter(torch.FloatTensor(self.feature_dim*2 + self.embedding_dim, self.embedding_dim))
        init.xavier_uniform_(self.detaching_weight)
        self.W = nn.Parameter(torch.FloatTensor(size=(self.feature_dim, self.embedding_dim)))  
        nn.init.xavier_uniform_(self.W)
        self.alpha = 0.2
        self.a = nn.Parameter(torch.FloatTensor(size=(1,2*self.embedding_dim)))
        nn.init.xavier_uniform_(self.a)
        self.leakyrelu = nn.LeakyReLU(self.alpha)       

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        num_sample = self.filter_size
        neigh_nodes = [self.adjacent_list[int(node)] for node in nodes]
        neigh_feats = self.STC_layer.forward(neigh_nodes)   
        nodes2 = [int(node) for node in nodes]
        _set = set

        _sample = random.sample
        sampled_neighbors = [_set(_sample(batch_neighbor, num_sample,)) if len(batch_neighbor) >= num_sample else batch_neighbor for batch_neighbor in neigh_nodes]        
        
        unique_nodes_list = list(set.union(*sampled_neighbors))  
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)} 
        unique_nodes2 = {n:i for i,n in enumerate(nodes2)} 
        embedding_matrix = self.features(torch.LongTensor(unique_nodes_list).to(self.device))
        embedding_matrix2 = self.features(torch.LongTensor(nodes2).to(self.device))
        column_indices = [unique_nodes[n] for sampled_neighbor in sampled_neighbors for n in sampled_neighbor]   

        edge1 = [[unique_nodes2[nodes2[i]] for i,sampled_neighbor in enumerate(sampled_neighbors) for n in sampled_neighbor]]
        edge2 = column_indices
        edge1.append(edge2)
        edge = torch.LongTensor(edge1).to(self.device)
        edge[1,:] = edge[1,:] + len(unique_nodes2)
        input1 = torch.cat((embedding_matrix2,embedding_matrix),dim=0)
        
        h = torch.mm(input1, self.W)
        N = h.size()[0]
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        a = torch.sparse_coo_tensor(edge, edge_e, torch.Size([len(nodes2), N])).to(self.device)
        e_rowsum = torch.sparse.mm(a,torch.ones(size=(N,1), device=self.device))
        if self.dropout == True:
            edge_e = F.dropout(edge_e, self.dropout_param, training=self.training)
        h_prime = torch.nan_to_num(torch.sparse.mm(a,h))
        assert not torch.isnan(h_prime).any()
        h_prime = torch.nan_to_num(h_prime.div(e_rowsum))
        assert not torch.isnan(h_prime).any()
        temp_feat = h_prime
        combined_feats = torch.cat([embedding_matrix2, temp_feat, neigh_feats], dim=1)        
        combined_feats = F.relu(combined_feats.mm(self.detaching_weight))
        if self.dropout == True:
            combined_feats = F.dropout(combined_feats, self.dropout_param, training=self.training)

        return combined_feats        
        
