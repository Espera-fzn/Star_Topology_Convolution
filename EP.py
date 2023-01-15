import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict
from STC_layer import STC_layer
from decoder import STC_decoder
from STC_encoder import STC_encoder
from scipy.io import loadmat


# ==========================================================================================
# ==========================================================================================
# A simple demo of using Star Topology Convolution (STC) to predict essential proteins.
# ==========================================================================================
# ==========================================================================================
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
# =======================================================================================================================================================================================================================================
# =======================================================================================================================================================================================================================================
# Please cite our paper if you use this code in your own work:
# Wu, Chong; Feng, Zhenan; Zheng, Jiangbin; Zhang, Houwang; Cao, Jiawang; Yan, Hong (2020): Star Topology Convolution for Graph Representation Learning. TechRxiv. Preprint. https://doi.org/10.36227/techrxiv.12805799.v6
# =======================================================================================================================================================================================================================================
# =======================================================================================================================================================================================================================================


# Running settings
num_classes = 2
batchsize = 20
lr_ = 0.0001
filter_size1 = 20
filter_size2 = 20
max_iteration = 200
start_iteration = 0
eval_val_every = 10
dropout_flag = True
dropout_param = 0
hidden_dim1 = 256
hidden_dim2 = 256
times = []
f1mic_best = 0
f1_test_cor = 0
best_iter = 0
np.random.seed(1)
random.seed(1)


def get_parameter_number(model):
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable parameters: ', trainable_num)


def evaluation(STC_Decoder, val_nodes, labels, test_nodes):

    sz = 5
    ind = int(len(val_nodes)/sz)
    score_f = 0
    for i in range(ind):        
        val_nodes1 = val_nodes[i*sz:(i+1)*sz]
        val_output = STC_Decoder.forward(val_nodes1)
        score_f = score_f + f1_score(labels[val_nodes1], val_output.data.cpu().numpy().argmax(axis=1), average="micro")     
    score_f = score_f/ind 
                     
    ind = int(len(test_nodes)/sz)
    score_f1 = 0
    for i in range(ind):        
        test_nodes1 = test_nodes[i*sz:(i+1)*sz]
        test_output = STC_Decoder.forward(test_nodes1)
        score_f1 = score_f1 + f1_score(labels[test_nodes1], test_output.data.cpu().numpy().argmax(axis=1), average="micro")    
    score_f1 = score_f1/ind                        

    return score_f, score_f1


def load_protein():

    mat1 = loadmat("protein/new_features.mat")
    mat2 = loadmat("protein/new_labels.mat")
    mat3 = loadmat("protein/new_edge.mat")
    mat4 = loadmat("protein/new_nodes.mat")

    nodes = mat4['new_nodes']
    num_nodes = len(nodes[0,:])
    num_features = 36
    labels = mat2['new_essential_list']
    features_data = mat1['new_gene_exp']
    labels = labels.squeeze()
    edges = mat3['new_edge']
    edges = edges.tolist()
    adjacent_list = defaultdict(set)
    
    for i in range(len(edges)):
        ind = edges[i]
        protein1 = ind[0]
        protein2 = ind[1]
        adjacent_list[protein1].add(protein2)
        adjacent_list[protein2].add(protein1)

    rand_indices = np.random.permutation(num_nodes)
    test_nodes = rand_indices[:1000]
    val_nodes = rand_indices[1000:1500]
    train_nodes = list(rand_indices[1500:])   
    adjacent_lists = [adjacent_list]
    
    return features_data, labels, adjacent_lists, num_nodes, num_features, train_nodes, test_nodes, val_nodes


features_data, labels, adjacent_lists, num_nodes, num_features, train_nodes, test_nodes, val_nodes = load_protein()
features = nn.Embedding(num_nodes, num_features)   
features.weight = nn.Parameter(torch.FloatTensor(features_data), requires_grad=False)
feature_dim = num_features
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

STC_layer01 = STC_layer(features, filter_size1, feature_dim, hidden_dim1, device)
STC_encoder01 = STC_encoder(features, feature_dim, hidden_dim1, adjacent_lists, STC_layer01, device, dropout_param, dropout_flag, filter_size1-2)
STC_layer02 = STC_layer(lambda nodes : STC_encoder01(nodes), filter_size2, STC_encoder01.hidden_dim, hidden_dim2, device)
STC_encoder02 = STC_encoder(lambda nodes : STC_encoder01(nodes), STC_encoder01.hidden_dim, hidden_dim2, adjacent_lists, STC_layer02, device, dropout_param, dropout_flag, filter_size2-2, base_model = STC_encoder01)

STC_encoder_use = STC_encoder02
STC_Decoder = STC_decoder(num_classes, STC_encoder_use, device)
STC_Decoder.to(device)  

get_parameter_number(STC_Decoder)
optimizer = torch.optim.Adam(STC_Decoder.parameters(), lr=lr_)

for iteration in  range(start_iteration + 1 ,max_iteration):
    
    STC_Decoder.train()
    batch_nodes = train_nodes[:batchsize]
    random.shuffle(train_nodes)
    start_time = time.time()
    optimizer.zero_grad()
    loss = STC_Decoder.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
    loss.backward()
    optimizer.step()
    end_time = time.time()
    times.append(end_time - start_time)
    print("Iteration: ", iteration, ", loss: ", loss.item(),", average batch time: ", np.mean(times))       
    
    if (iteration + 1)%eval_val_every == 0:
        with torch.no_grad():
            STC_Decoder.eval()        
            f1mic_val, f1mic_test = evaluation(STC_Decoder, np.array(val_nodes), np.array(labels), np.array(test_nodes))
            if f1mic_val > f1mic_best:
                f1mic_best, f1_test_cor, best_iter = f1mic_val, f1mic_test, iteration
            print('Best f1 val: ', f1mic_best) 
            print('Corresponding f1 test: ', f1_test_cor) 
            print('Best iteration: ', best_iter)            
            
print("Optimization Finished!")
print('Final best f1 val: ', f1mic_best) 
print('Final corresponding f1 test: ', f1_test_cor) 
print('Final best epoch: ', best_iter)    
 
