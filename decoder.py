import torch
from torch.nn import init
import torch.nn as nn

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

class STC_decoder(nn.Module):
    """
    The decoder for classification
    """
    def __init__(self, num_classes, STC_encoder, device):
        super(STC_decoder, self).__init__()
        self.STC_encoder = STC_encoder
        self.device = device
        self.xent_loss = nn.CrossEntropyLoss()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, STC_encoder.embedding_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):        
        embeddings = self.STC_encoder(nodes)
        output = self.weight.mm(embeddings)    
        return output.t()

    def loss(self, nodes, labels):        
        output = self.forward(nodes)        
        return self.xent_loss(output, labels.squeeze().to(self.device))