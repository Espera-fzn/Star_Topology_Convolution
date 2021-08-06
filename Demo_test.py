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
import os
from STC_encoder import STC_encoder

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


#############################################################################################
# Parameter settings
#############################################################################################
num_classes = 349
batchsize = 200
lr_ = 0.001
filter_size1 = 15
filter_size2 = 15
maxgen = 30000
dropout_a = True
dropout_param = 0.5
np.random.seed(1)
random.seed(1)

embedding_dim1 = 256
embedding_dim2 = 256

save_dir = './raw_mag/'


#############################################################################################
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total', total_num)
    print('Trainable', trainable_num)

def load_MAG():
    num_nodes = 736389
    num_features = 128
    

    features_data = np.load(save_dir+'feats.npy')
    train_idx = np.load(save_dir+'train_sp.npy')
    test_idx = np.load(save_dir+'test_sp.npy')
    valid_idx = np.load(save_dir+'val_sp.npy')
    
    labels1 = np.load(save_dir+'labels.npy') 
    labels = labels1[:,0]
    
    adjacent_list = defaultdict(set)
    print('Dataset name: Ogbn-MAG')
    edge_index = np.load(save_dir+'edge.npy')
    
    ind = len(edge_index[:,0])
    
    for i in range(ind):
        paper1 = edge_index[i,0]
        paper2 = edge_index[i,1]
        adjacent_list[paper1].add(paper2)
        adjacent_list[paper2].add(paper1) 

    train_idx_set = train_idx.tolist()
    test_idx_set = test_idx.tolist()
    val_idx_set = valid_idx.tolist()
    return features_data, labels, adjacent_list, num_nodes, num_features, train_idx_set, test_idx_set, val_idx_set

def evaluate_full_batch(STC_Decoder, val_nodes, labels, test_nodes):

    sz = 5
    #valid
    ind2 = int(len(val_nodes)/sz)
    score_f = 0
    for i in range(ind2):        
        val_nodes1 = val_nodes[i*sz:(i+1)*sz]
        val_output = STC_Decoder.forward(val_nodes1)
        score_f = score_f + f1_score(labels[val_nodes1], val_output.data.cpu().numpy().argmax(axis=1), average="micro")    
    score_f = score_f/ind2 

    #test
    ind2 = int(len(test_nodes)/sz)
    score_f1 = 0
    for i in range(ind2):        
        test_nodes1 = test_nodes[i*sz:(i+1)*sz]
        test_output = STC_Decoder.forward(test_nodes1)
        score_f1 = score_f1 + f1_score(labels[test_nodes1], test_output.data.cpu().numpy().argmax(axis=1), average="micro")    
    score_f1 = score_f1/ind2                        

    return score_f, score_f1


def run_MAG():

    features_data, labels, adjacent_list, num_nodes, num_features, train_nodes, test_nodes, val_nodes = load_MAG()
    features = nn.Embedding(num_nodes, num_features)
    features.weight = nn.Parameter(torch.FloatTensor(features_data), requires_grad=False)
    feature_dim = num_features 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    STC_layer1 = STC_layer(features, filter_size1, feature_dim, device)
    STC_encoder1 = STC_encoder(features, feature_dim, embedding_dim1, adjacent_list, STC_layer1, device, dropout_param, dropout_a, filter_size1-2)
    STC_layer2 = STC_layer(lambda nodes : STC_encoder1(nodes), filter_size2, STC_encoder1.embedding_dim, device)
    STC_encoder2 = STC_encoder(lambda nodes : STC_encoder1(nodes), STC_encoder1.embedding_dim, embedding_dim2, adjacent_list, STC_layer2, device, dropout_param, dropout_a, filter_size2-2, base_model = STC_encoder1)

    STC_Decoder = STC_decoder(num_classes, STC_encoder2, device)
    STC_Decoder.to(device)  

    get_parameter_number(STC_Decoder)
    optimizer = torch.optim.Adam(STC_Decoder.parameters(), lr=lr_)
    eval_val_every = 10
    times = []
    f1mic_best = 0
    f1_test_cor = 0
    epb = 0
    #training
    for epoch in range(maxgen):
        STC_Decoder.train()
        batch_nodes = train_nodes[:batchsize]
        random.shuffle(train_nodes)
        start_time = time.time()
        optimizer.zero_grad()
        loss = STC_Decoder.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print("Epoch: ", epoch, ", loss: ", loss.item(),", average batch time: ", np.mean(times))   
        if (epoch+1)%eval_val_every == 0:
            with torch.no_grad():
                STC_Decoder.eval()
                f1mic_val, f1mic_test = evaluate_full_batch(STC_Decoder, np.array(val_nodes), np.array(labels),np.array(test_nodes))
                if f1mic_val > f1mic_best:
                    f1mic_best, f1_test_cor, epb = f1mic_val, f1mic_test, epoch
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    
                    path_saver = save_dir+ '/saved_model.pkl'
                    torch.save(STC_Decoder.state_dict(), path_saver)
                print('Best f1 val: ', f1mic_best) 
                print('Corresponding f1 test: ', f1_test_cor) 
                print('Best epoch: ', epb)
                
    print("Optimization Finished!")
    print('Final best f1 val: ', f1mic_best) 
    print('Final f1 test: ', f1_test_cor) 
    print('Final best epoch: ', epb)

if __name__ == "__main__":
    run_MAG() 
