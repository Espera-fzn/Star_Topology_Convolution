import torch
import torch.nn as nn
from torch.nn import init
import random
import numpy as np


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
# Wu, Chong; Feng, Zhenan; Zheng, Jiangbin; Zhang, Houwang; Cao, Jiawang; Yan, Hong (2020): Star Topology Convolution for Graph Representation Learning. TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.12805799.v5 
# =============================================================================
# =============================================================================


"""
Vanilla Star Topology Convolution (Vanilla STC or STC (conv only)) Layer.
"""

class STC_layer(nn.Module):
    """
    Do spectral convolution on the star topology subgraphs
    """
    def __init__(self, features, filter_size, feature_dim, hidden_dim, device): 
        """
        Initializes the Vanilla Star Topology Convolution (Vanilla STC or STC (conv only)) for a specific graph.

        features        -- function mapping LongTensor of node ids to FloatTensor of feature values
        filter_size     -- filter size (fiter size in paper + 2)
        feature_dim     -- feature dimension  
        hidden_dim      -- hidden unit dimension which is not used in Vanilla STC
        device          -- GPU or CPU
        """

        super(STC_layer, self).__init__()

        self.features = features
        self.filter_size = filter_size
        self.feature_dim = feature_dim
        self.device = device
        self.hidden_dim = hidden_dim
        self.Laplacian_mat = np.zeros((filter_size,filter_size))
        ADJ = np.ones(filter_size-1)   
        A = np.zeros((filter_size,filter_size))
        D = np.diag(np.ones(filter_size))
        I = np.diag(np.ones(filter_size))
        D[0,0] = (filter_size-1)**(-1/2)
        for i in range(filter_size-1):
            A[0,i+1] = ADJ[i]
            A[i+1,0] = ADJ[i]

        self.Laplacian_mat = I - np.matmul(D,np.matmul(A,D))
        s,u = np.linalg.eigh(self.Laplacian_mat)
        self.eigen_vec = torch.FloatTensor(u).to(self.device)
        self.weight = nn.Parameter(torch.FloatTensor(self.filter_size,1))
        init.xavier_uniform_(self.weight)
        self.avgweight = nn.Parameter(torch.FloatTensor(self.filter_size,1))
        init.xavier_uniform_(self.avgweight)

    def forward(self, batch_neighbors):
        """
        batch_neighbors    -- list of sets, each set is the set of neighbors for node in batch
        """
        
        _set = set
        num_sample = self.filter_size-2
        _sample = random.sample
        sampled_neighbors = [_set(_sample(batch_neighbor, num_sample,)) if len(batch_neighbor) >= num_sample else batch_neighbor for batch_neighbor in batch_neighbors]        
        
        unique_nodes_list = list(set.union(*sampled_neighbors))                
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)} 
        column_indices = [unique_nodes[n] for sampled_neighbor in sampled_neighbors for n in sampled_neighbor]   
        row_indices = [i for i in range(len(sampled_neighbors)) for j in range(len(sampled_neighbors[i]))]
        feat_matrix = self.features(torch.LongTensor(unique_nodes_list).to(self.device))
        
        fdim = feat_matrix.size(1)
        result = {}
        result = {int(i):row_indices.count(i) for i in set(row_indices)}
        num_Count = result
        
        mask1 = torch.zeros(len(sampled_neighbors)*self.filter_size,fdim).to(self.device)
        
        index_c1 = []        
        index_c1 = [i*self.filter_size+j+1 for i in range(len(sampled_neighbors)) for j in range(num_Count[i])]  
        mask1[index_c1] = feat_matrix[column_indices,:]   
        
        U = self.eigen_vec
        UT = torch.transpose(U, 0, 1)
        mask1 = mask1.view(len(sampled_neighbors), self.filter_size,fdim)
        mask1 = torch.transpose(mask1, 1, 2)
        mask2 = torch.matmul(mask1,U)
        mask3 = torch.mul(mask2, self.weight.view(1,1,-1))
        temp_feat = torch.matmul(mask3,UT)
        temp_feat = torch.matmul(temp_feat,self.avgweight)
        temp_feat = torch.transpose(temp_feat, 1, 2)
        temp_feat = temp_feat.contiguous().view(-1,fdim)
        batch_features = temp_feat

        return batch_features
