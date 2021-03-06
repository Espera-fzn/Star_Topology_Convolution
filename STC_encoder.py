import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F


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


class STC_encoder(nn.Module):
    """
    Encodes a node's using Star Topology Convolution (STC)
    """
    def __init__(self, features, feature_dim, embedding_dim, adjacent_list, STC_layer, device, num_sample=10, base_model = None): 
        super(STC_encoder, self).__init__()
        """
        features        -- function mapping LongTensor of node ids to FloatTensor of feature values
        feature_dim     -- feature dimension
        embedding_dim   -- embedding dimension
        adjacent_list   -- adjacent list
        STC_layer       -- Star Topology Convolution (STC) layer
        device          -- GPU or CPU
        num_sample      -- number of neighbors to sample
        """
        self.features = features
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.adjacent_list = adjacent_list
        self.STC_layer = STC_layer
        self.device = device
        self.num_sample = num_sample     
        if base_model != None:
            self.base_model = base_model
        self.detaching_weight = nn.Parameter(torch.FloatTensor(self.embedding_dim, 2 * self.feature_dim))
        init.xavier_uniform_(self.detaching_weight)
        self.batch_norm = torch.nn.BatchNorm1d(self.feature_dim)            
            


    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        neigh_feats = self.STC_layer.forward(nodes, [self.adjacent_list[int(node)] for node in nodes])   
        nodes2 = [int(node) for node in nodes]
        self_feats = self.features(torch.LongTensor(nodes2).to(self.device))
        self_feats = self.batch_norm(self_feats)
        combined_feats = torch.cat([self_feats, neigh_feats], dim=1)        
        combined_feats = F.relu(self.detaching_weight.mm(combined_feats.t()))
        return combined_feats        
        

