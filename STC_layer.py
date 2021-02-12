import torch
import torch.nn as nn
from torch.autograd import Variable
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
# Wu, Chong; Feng, Zhenan; Zheng, Jiangbin; Zhang, Houwang; Cao, Jiawang; Yan, Hong (2020): Star Topology Convolution for Graph Representation Learning. TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.12805799.v3
# =============================================================================
# =============================================================================


"""
Star Topology Convolution (STC) Layer.
"""

class STC_layer(nn.Module):
    """
    Do spectral convolution on the star topology subgraphs
    """
    def __init__(self, features, filter_size, feature_dim, device): 
        """
        Initializes the Star Topology Convolution (STC) for a specific graph.

        features        -- function mapping LongTensor of node ids to FloatTensor of feature values
        filter_size     -- filter size (fiter size in paper + 2)
        feature_dim     -- feature dimension  
        device          -- GPU or CPU
        """

        super(STC_layer, self).__init__()

        self.batch_norm = torch.nn.BatchNorm1d(feature_dim)
        self.features = features
        self.filter_size = filter_size
        self.feature_dim = feature_dim
        self.device = device
        self.embedding_dim = feature_dim
        self.filter = np.zeros((filter_size,filter_size))
        ADJ = np.ones(filter_size-1)   
        A = np.zeros((filter_size,filter_size))
        D = np.diag(np.ones(filter_size))
        I = np.diag(np.ones(filter_size))
        D[0,0] = (filter_size-1)**(-1/2)
        for i in range(filter_size-1):
            # self.filter[0,i+1] = -ADJ[i]
            # self.filter[i+1,0] = -ADJ[i]
            
            A[0,i+1] = ADJ[i]
            A[i+1,0] = ADJ[i]
            
            # if i == 0:
            #     self.filter[0,0] = filter_size-1
            #     self.filter[-1,-1] = 1
            # else:
            #     self.filter[i,i] = 1
        self.filter = I - np.matmul(D,np.matmul(A,D))
        s,u = np.linalg.eigh(self.filter)
        self.filter2 = torch.FloatTensor(u).to(self.device)
        self.weight = nn.Parameter(torch.FloatTensor(self.filter_size,1))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes, batch_neighbors):
        """
        nodes              -- list of nodes in a batch
        batch_neighbors    -- list of sets, each set is the set of neighbors for node in batch
        """
        
        # Local pointers to functions (speed hack)
        _set = set
        num_sample = self.filter_size-2
        _sample = random.sample
        sampled_neighbors = [_set(_sample(batch_neighbor, num_sample,)) if len(batch_neighbor) >= num_sample else batch_neighbor for batch_neighbor in batch_neighbors]        
        
        
        unique_nodes_list = list(set.union(*sampled_neighbors))                
        
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)} 



              
        mask = torch.zeros(len(sampled_neighbors), len(unique_nodes))
        column_indices = [unique_nodes[n] for sampled_neighbor in sampled_neighbors for n in sampled_neighbor]   
        row_indices = [i for i in range(len(sampled_neighbors)) for j in range(len(sampled_neighbors[i]))]
        mask[row_indices, column_indices] = 1 
        
        
        mask = mask.to(self.device)
        
        num_neighors = mask.sum(1, keepdim=True)
        mask = mask.div(num_neighors)
        

        embedding_matrix = self.features(torch.LongTensor(unique_nodes_list).to(self.device))
        embedding_matrix1 = torch.FloatTensor(len(unique_nodes_list),self.embedding_dim).to(self.device)                
        
        fdim = embedding_matrix.size(1)
        result = {}
        for i in set(row_indices):
            result[i]=row_indices.count(i)
        num_Count = result
        end_num = 0
        start_num = 0
        
        #mask2 = Variable(torch.zeros(self.filter_size, fdim*len(sampled_neighbors))).to(self.device)
        mask2 = torch.zeros(self.filter_size, fdim*len(sampled_neighbors)).to(self.device)
        #mask3 = torch.zeros(self.filter_size, fdim*len(sampled_neighbors)).to(self.device)
        
        index_c1 = []
        index_r = []
        for i in range(len(sampled_neighbors)):
            start_num = end_num
            end_num = end_num + num_Count[i]
            index_c = i*fdim
            mask2[1:num_Count[i]+1,index_c:index_c+fdim] = embedding_matrix[column_indices[start_num:end_num],:]        
        #print(mask2)
        U = self.filter2
        UT = torch.transpose(U, 0, 1)
        weight2 = torch.mm(UT,mask2)
        mask3 = torch.mul(self.weight,weight2)        
        temp_feat = torch.mm(U, mask3)
        end_num = 0
        start_num = 0
        for i in range(len(sampled_neighbors)):
            start_num = end_num
            end_num = end_num + num_Count[i]
            index_c = i*fdim
            embedding_matrix1[column_indices[start_num:end_num],:] = temp_feat[1:num_Count[i]+1,index_c:index_c+fdim]

        batch_features = mask.mm(embedding_matrix1)
        batch_features = self.batch_norm(batch_features)
        return batch_features
