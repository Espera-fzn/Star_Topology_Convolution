import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict


from STC_encoder import STC_encoder
from STC_layer import STC_layer
from decoder import STC_decoder


import scipy.io as scio
from scipy.io import loadmat
import os

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

def load_MAG():
    num_nodes = 736389
    num_features = 128
    
    save_dir = './MAG/'
    features_data = np.load(save_dir+'feats.npy')
    train_idx = np.load(save_dir+'train_sp.npy')
    test_idx = np.load(save_dir+'test_sp.npy')
    valid_idx = np.load(save_dir+'val_sp.npy')
    
    train_idx_set = train_idx.tolist()
    test_idx_set = test_idx.tolist()
    val_idx_set = valid_idx.tolist()
    
    labels1 = np.load(save_dir+'labels.npy') 
    labels = labels1[:,0]
    
    adjacent_list = defaultdict(set)
       
    edge_index = np.load(save_dir+'edge.npy')
    
    ind = len(edge_index[0,:])
    
    for i in range(ind):
        paper1 = edge_index[0,i]
        paper2 = edge_index[1,i]
        adjacent_list[paper1].add(paper2)
        adjacent_list[paper2].add(paper1) 
 
    return features_data, labels, adjacent_list, num_nodes, num_features, train_idx_set, test_idx_set, val_idx_set


def load_Arxiv():
    num_nodes = 169343
    num_features = 128
    
    save_dir = './Arxiv/'
    features_data = np.load(save_dir+'feats.npy')
    train_idx = np.load(save_dir+'train_sp.npy')
    test_idx = np.load(save_dir+'test_sp.npy')
    valid_idx = np.load(save_dir+'val_sp.npy')
    
    train_idx_set = train_idx.tolist()
    test_idx_set = test_idx.tolist()
    val_idx_set = valid_idx.tolist()
    
    labels1 = np.load(save_dir+'labels.npy') 
    labels = labels1[:,0]
    
    adjacent_list = defaultdict(set)
       
    edge_index = np.load(save_dir+'edge.npy')
    
    ind = len(edge_index[0,:])
    
    for i in range(ind):
        paper1 = edge_index[0,i]
        paper2 = edge_index[1,i]
        adjacent_list[paper1].add(paper2)
        adjacent_list[paper2].add(paper1) 
 
    return features_data, labels, adjacent_list, num_nodes, num_features, train_idx_set, test_idx_set, val_idx_set



def load_cora():
    num_nodes = 2708
    num_features = 1433
    features_data = np.zeros((num_nodes, num_features))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            features_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]
            
    adjacent_list = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adjacent_list[paper1].add(paper2)
            adjacent_list[paper2].add(paper1)
    return features_data, labels, adjacent_list, num_nodes, num_features



def load_citeseer():
    num_nodes = 3312
    num_features = 3703
    features_data = np.zeros((num_nodes, num_features))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("citeseer/citeseer.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            features_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adjacent_list = defaultdict(set)
    with open("citeseer/citeseer.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            if info[0] in node_map and info[1] in node_map:
                paper1 = node_map[info[0]]
                paper2 = node_map[info[1]]
                adjacent_list[paper1].add(paper2)
                adjacent_list[paper2].add(paper1)

    

    return features_data, labels, adjacent_list, num_nodes, num_features


def load_pubmed():
    num_nodes = 19717
    num_features = 500
    features_data = np.zeros((num_nodes, num_features))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                features_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adjacent_list = defaultdict(set)
    with open("pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adjacent_list[paper1].add(paper2)
            adjacent_list[paper2].add(paper1)
    return features_data, labels, adjacent_list, num_nodes, num_features

def load_protein():

    m = loadmat("protein/new_features.mat")
    n = loadmat("protein/new_labels.mat")
    e = loadmat("protein/new_edge.mat")

    d = loadmat("protein/new_nodes.mat")

    nodes = d['new_nodes']
    num_nodes = len(nodes[0,:])
    num_features = 36
    labels = n['new_essential_list']
    feat_data = m['new_gene_exp']
    labels = labels.squeeze()

    a_list = e['new_edge']
    a_list = a_list.tolist()


    adj_lists = defaultdict(set)
    for i in range(len(a_list)):
        ind = a_list[i]
        protein1 = ind[0]
        protein2 = ind[1]
        adj_lists[protein1].add(protein2)

        adj_lists[protein2].add(protein1)

  


    return feat_data, labels, adj_lists, num_nodes, num_features


def run_MAG1():
 
    np.random.seed(2)
    random.seed(2)
    features_data, labels, adjacent_list, num_nodes, num_features, train_nodes, test_nodes, val_nodes = load_MAG()
    features = nn.Embedding(num_nodes, num_features)
    features.weight = nn.Parameter(torch.FloatTensor(features_data), requires_grad=False)
    feature_dim = num_features 
    filter_size1 = 20    
    filter_size2 = 20
    embedding_dim1 = 256
    embedding_dim2 = 256
    
    num_classes = 349
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    STC_layer1 = STC_layer(features, filter_size1, feature_dim, device)
    STC_encoder1 = STC_encoder(features, feature_dim, embedding_dim1, adjacent_list, STC_layer1, device, num_sample = filter_size1-2)
    STC_layer2 = STC_layer(lambda nodes : STC_encoder1(nodes).t(), filter_size2, STC_encoder1.embedding_dim, device)
    STC_encoder2 = STC_encoder(lambda nodes : STC_encoder1(nodes).t(), STC_encoder1.embedding_dim, embedding_dim2, adjacent_list, STC_layer2, device, num_sample = filter_size2-2, base_model = STC_encoder1)
    
    
    STC_Decoder = STC_decoder(num_classes, STC_encoder2, device)
 
    STC_Decoder.to(device)  
    
         
    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, STC_Decoder.parameters()), lr=0.7)
    times = []
    #training
    for batch in range(2000):
        batch_nodes = train_nodes[:200]
        random.shuffle(train_nodes)
        start_time = time.time()
        optimizer.zero_grad()
        loss = STC_Decoder.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
  
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.item())   
    #test
    ind2 = 163
    score_f = 0
    for i in range(ind2):        
        test_nodes1 = test_nodes[i*256:(i+1)*256]
        test_output = STC_Decoder.forward(test_nodes1) 
        score_f = score_f + f1_score(labels[test_nodes1], test_output.data.cpu().numpy().argmax(axis=1), average="micro")    
    score_f = score_f/ind2                        
    print("TEST F1 micro:", score_f)


def run_Arxiv1():
 
    np.random.seed(2)
    random.seed(2)
    features_data, labels, adjacent_list, num_nodes, num_features, train_nodes, test_nodes, val_nodes = load_Arxiv()
    features = nn.Embedding(num_nodes, num_features)
    features.weight = nn.Parameter(torch.FloatTensor(features_data), requires_grad=False)
    feature_dim = num_features 
    filter_size1 = 20    
    filter_size2 = 20
    embedding_dim1 = 128
    embedding_dim2 = 128
    
    num_classes = 40
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    STC_layer1 = STC_layer(features, filter_size1, feature_dim, device)
    STC_encoder1 = STC_encoder(features, feature_dim, embedding_dim1, adjacent_list, STC_layer1, device, num_sample = filter_size1-2)
    STC_layer2 = STC_layer(lambda nodes : STC_encoder1(nodes).t(), filter_size2, STC_encoder1.embedding_dim, device)
    STC_encoder2 = STC_encoder(lambda nodes : STC_encoder1(nodes).t(), STC_encoder1.embedding_dim, embedding_dim2, adjacent_list, STC_layer2, device, num_sample = filter_size2-2, base_model = STC_encoder1)
    
    
    STC_Decoder = STC_decoder(num_classes, STC_encoder2, device)
 
    STC_Decoder.to(device)  
    
         
    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, STC_Decoder.parameters()), lr=0.7)
    times = []
    #training
    for batch in range(200):
        batch_nodes = train_nodes[:1000]
        random.shuffle(train_nodes)
        start_time = time.time()
        optimizer.zero_grad()
        loss = STC_Decoder.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
  
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.item())   
    #test
    ind2 = 189
    score_f = 0
    for i in range(ind2):        
        test_nodes1 = test_nodes[i*256:(i+1)*256]
        test_output = STC_Decoder.forward(test_nodes1) 
        score_f = score_f + f1_score(labels[test_nodes1], test_output.data.cpu().numpy().argmax(axis=1), average="micro")    
    score_f = score_f/ind2                        
    print("TEST F1 micro:", score_f)
    




def run_cora1():
 
    np.random.seed(2)
    random.seed(2)
    features_data, labels, adjacent_list, num_nodes, num_features = load_cora()
    features = nn.Embedding(num_nodes, num_features)
    features.weight = nn.Parameter(torch.FloatTensor(features_data), requires_grad=False)
    feature_dim = num_features 
    filter_size1 = 20    
    filter_size2 = 20
    embedding_dim1 = 128
    embedding_dim2 = 128
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    STC_layer1 = STC_layer(features, filter_size1, feature_dim, device)
    STC_encoder1 = STC_encoder(features, feature_dim, embedding_dim1, adjacent_list, STC_layer1, device, num_sample = filter_size1-2)
    STC_layer2 = STC_layer(lambda nodes : STC_encoder1(nodes).t(), filter_size2, STC_encoder1.embedding_dim, device)
    STC_encoder2 = STC_encoder(lambda nodes : STC_encoder1(nodes).t(), STC_encoder1.embedding_dim, embedding_dim2, adjacent_list, STC_layer2, device, num_sample = filter_size2-2, base_model = STC_encoder1)
    
    
    STC_Decoder = STC_decoder(7, STC_encoder2, device)
 
    STC_Decoder.to(device)  
    
    rand_indices = np.random.permutation(num_nodes)
    test_nodes = rand_indices[:1000]
    val_nodes = rand_indices[1000:1500]
    train_nodes = list(rand_indices[1500:])           

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, STC_Decoder.parameters()), lr=0.7)
    times = []
    #training
    for batch in range(200):
        batch_nodes = train_nodes[:100]
        random.shuffle(train_nodes)
        start_time = time.time()
        optimizer.zero_grad()
        loss = STC_Decoder.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
  
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.item())   
    #test
    test_output = STC_Decoder.forward(test_nodes) 
    f1mic_test = f1_score(labels[test_nodes], test_output.data.cpu().numpy().argmax(axis=1), average="micro")
    f1mac_test = f1_score(labels[test_nodes], test_output.data.cpu().numpy().argmax(axis=1), average="macro") 
    print("TEST F1 micro:", f1mic_test)
    print("TEST F1 macro:", f1mac_test)
    #validation
    val_output = STC_Decoder.forward(val_nodes) 
    f1mic_val = f1_score(labels[val_nodes], val_output.data.cpu().numpy().argmax(axis=1), average="micro")
    f1mac_val = f1_score(labels[val_nodes], val_output.data.cpu().numpy().argmax(axis=1), average="macro")                       
    print("VAL F1 micro:", f1mic_val)
    print("VAL F1 macro:", f1mac_val)
    print("Average batch time:", np.mean(times))    



def run_citeseer1():
    
    np.random.seed(2)
    random.seed(2)
    features_data, labels, adjacent_list, num_nodes, num_features = load_citeseer()
    features = nn.Embedding(num_nodes, num_features)
    features.weight = nn.Parameter(torch.FloatTensor(features_data), requires_grad=False)
    feature_dim = num_features 
    filter_size1 = 20    
    filter_size2 = 20
    embedding_dim1 = 128
    embedding_dim2 = 128
  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    
    rand_indices = np.random.permutation(num_nodes)
    test_nodes = rand_indices[:1000]
    val_nodes = rand_indices[1000:1500]
    train_nodes = list(rand_indices[1500:])
    STC_layer1 = STC_layer(features, filter_size1, feature_dim, device)
    STC_encoder1 = STC_encoder(features, feature_dim, embedding_dim1, adjacent_list, STC_layer1, device, num_sample = filter_size1-2)
    STC_layer2 = STC_layer(lambda nodes : STC_encoder1(nodes).t(), filter_size2, STC_encoder1.embedding_dim, device)
    STC_encoder2 = STC_encoder(lambda nodes : STC_encoder1(nodes).t(), STC_encoder1.embedding_dim, embedding_dim2, adjacent_list, STC_layer2, device, num_sample = filter_size2-2, base_model = STC_encoder1)
    
    
    STC_Decoder = STC_decoder(6, STC_encoder2, device)
 
    STC_Decoder.to(device)  
    
         

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, STC_Decoder.parameters()), lr=0.7)
    times = []
    #training
    for batch in range(200):
        batch_nodes = train_nodes[:100]
        random.shuffle(train_nodes)
        start_time = time.time()
        optimizer.zero_grad()
        loss = STC_Decoder.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
  
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.item()) 
    #test
    test_output = STC_Decoder.forward(test_nodes) 
    f1mic_test = f1_score(labels[test_nodes], test_output.data.cpu().numpy().argmax(axis=1), average="micro")
    f1mac_test = f1_score(labels[test_nodes], test_output.data.cpu().numpy().argmax(axis=1), average="macro") 
    print("TEST F1 micro:", f1mic_test)
    print("TEST F1 macro:", f1mac_test)

    #validation
    val_output = STC_Decoder.forward(val_nodes) 
    f1mic_val = f1_score(labels[val_nodes], val_output.data.cpu().numpy().argmax(axis=1), average="micro")
    f1mac_val = f1_score(labels[val_nodes], val_output.data.cpu().numpy().argmax(axis=1), average="macro")                       
    print("VAL F1 micro:", f1mic_val)
    print("VAL F1 macro:", f1mac_val)
    print("Average batch time:", np.mean(times)) 


def run_pubmed1():
    
    np.random.seed(2)
    random.seed(2)
    features_data, labels, adjacent_list, num_nodes, num_features = load_pubmed()
    features = nn.Embedding(num_nodes, num_features)
    features.weight = nn.Parameter(torch.FloatTensor(features_data), requires_grad=False)
    feature_dim = num_features 
    filter_size1 = 20    
    filter_size2 = 20
    embedding_dim1 = 128
    embedding_dim2 = 128
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    
    rand_indices = np.random.permutation(num_nodes)
    test_nodes = rand_indices[:1000]
    val_nodes = rand_indices[1000:1500]
    train_nodes = list(rand_indices[1500:])
    STC_layer1 = STC_layer(features, filter_size1, feature_dim, device)
    STC_encoder1 = STC_encoder(features, feature_dim, embedding_dim1, adjacent_list, STC_layer1, device, num_sample = filter_size1-2)
    STC_layer2 = STC_layer(lambda nodes : STC_encoder1(nodes).t(), filter_size2, STC_encoder1.embedding_dim, device)
    STC_encoder2 = STC_encoder(lambda nodes : STC_encoder1(nodes).t(), STC_encoder1.embedding_dim, embedding_dim2, adjacent_list, STC_layer2, device, num_sample = filter_size2-2, base_model = STC_encoder1)
    
    
    STC_Decoder = STC_decoder(3, STC_encoder2, device)
 
    STC_Decoder.to(device)  
    
         

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, STC_Decoder.parameters()), lr=0.7)
    times = []
    #training
    for batch in range(200):
        batch_nodes = train_nodes[:512]
        random.shuffle(train_nodes)
        start_time = time.time()
        optimizer.zero_grad()
        loss = STC_Decoder.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
  
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.item())
    #test    
    test_output = STC_Decoder.forward(test_nodes) 
    f1mic_test = f1_score(labels[test_nodes], test_output.data.cpu().numpy().argmax(axis=1), average="micro")
    f1mac_test = f1_score(labels[test_nodes], test_output.data.cpu().numpy().argmax(axis=1), average="macro") 
    print("TEST F1 micro:", f1mic_test)
    print("TEST F1 macro:", f1mac_test)
    
    #validation
    val_output = STC_Decoder.forward(val_nodes) 
    f1mic_val = f1_score(labels[val_nodes], val_output.data.cpu().numpy().argmax(axis=1), average="micro")
    f1mac_val = f1_score(labels[val_nodes], val_output.data.cpu().numpy().argmax(axis=1), average="macro")                       
    print("VAL F1 micro:", f1mic_val)
    print("VAL F1 macro:", f1mac_val)
    print("Average batch time:", np.mean(times)) 


def run_protein1():
    
    np.random.seed(2)
    random.seed(2)
    features_data, labels, adjacent_list, num_nodes, num_features = load_protein()
    features = nn.Embedding(num_nodes, num_features)
    features.weight = nn.Parameter(torch.FloatTensor(features_data), requires_grad=False)
    feature_dim = num_features 
    filter_size1 = 20    
    filter_size2 = 20
    embedding_dim1 = 128
    embedding_dim2 = 128
    


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    
    rand_indices = np.random.permutation(num_nodes)
    test_nodes = rand_indices[:1000]
    val_nodes = rand_indices[1000:1500]
    train_nodes = list(rand_indices[1500:])
    STC_layer1 = STC_layer(features, filter_size1, feature_dim, device)
    STC_encoder1 = STC_encoder(features, feature_dim, embedding_dim1, adjacent_list, STC_layer1, device, num_sample = filter_size1-2)
    STC_layer2 = STC_layer(lambda nodes : STC_encoder1(nodes).t(), filter_size2, STC_encoder1.embedding_dim, device)
    STC_encoder2 = STC_encoder(lambda nodes : STC_encoder1(nodes).t(), STC_encoder1.embedding_dim, embedding_dim2, adjacent_list, STC_layer2, device, num_sample = filter_size2-2, base_model = STC_encoder1)
    
    
    STC_Decoder = STC_decoder(2, STC_encoder2, device)
 
    STC_Decoder.to(device)  
    
         

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, STC_Decoder.parameters()), lr=0.7)
    times = []
    #training
    for batch in range(200):
        batch_nodes = train_nodes[:100]
        random.shuffle(train_nodes)
        start_time = time.time()
        optimizer.zero_grad()
        loss = STC_Decoder.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
  
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.item())   
    #test    
    test_output = STC_Decoder.forward(test_nodes) 
    f1mic_test = f1_score(labels[test_nodes], test_output.data.cpu().numpy().argmax(axis=1), average="micro")
    f1mac_test = f1_score(labels[test_nodes], test_output.data.cpu().numpy().argmax(axis=1), average="macro") 
    print("TEST F1 micro:", f1mic_test)
    print("TEST F1 macro:", f1mac_test)
    
    #validation
    val_output = STC_Decoder.forward(val_nodes) 
    f1mic_val = f1_score(labels[val_nodes], val_output.data.cpu().numpy().argmax(axis=1), average="micro")
    f1mac_val = f1_score(labels[val_nodes], val_output.data.cpu().numpy().argmax(axis=1), average="macro")                       
    print("VAL F1 micro:", f1mic_val)
    print("VAL F1 macro:", f1mac_val)
    print("Average batch time:", np.mean(times)) 
    
    
if __name__ == "__main__":
    run_Arxiv1()    