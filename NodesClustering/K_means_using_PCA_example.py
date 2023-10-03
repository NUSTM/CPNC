#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: Siwei Wu
@file: K_means.py
@time: 2022/10/15
@contact: wusiwei@njust.edu.cn
"""
import torch
import numpy as np
from kmeans_pytorch import kmeans
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
import json
from sklearn.decomposition import PCA

f = open('/root/commonsense-kg-completion-master/data/atomic/atomic_training_nodes.txt', 'r', encoding = 'utf-8')
data = f.readlines()
nodes = []
for line in data:
    nodes.append(line.replace('\n', ''))
f.close()

EB_forward = torch.load('/root/commonsense-kg-completion-master/bert_model_embeddings/nodes-lm-atomic/atomic_bert_embeddings.pt')


pca = PCA(n_components=250)
        
cluster_EB = pca.fit_transform(EB_forward)
cluster_EB = torch.tensor(cluster_EB, dtype = torch.float32)

num_clusters = 100
x = cluster_EB

# kmeans
nodes_index, _ = kmeans(
   X=x, num_clusters=num_clusters, distance='cosine', device=torch.device('cuda:0')
)

nodes_cluster = {}
for i in range(nodes_index.size(0)):
   node = nodes[i]
   nodes_cluster[node] = nodes_index[i].item()

nodes_index_to_cluster_index = []
cluster_centers = {}
for i, node in enumerate(nodes):
    if nodes_cluster[node] not in cluster_centers:
        cluster_centers[nodes_cluster[node]] = []
    cluster_centers[nodes_cluster[node]].append(EB_forward[i, :])
    nodes_index_to_cluster_index.append(nodes_cluster[node])

for cluster_index in cluster_centers:
    cluster_centers[cluster_index] = torch.stack(cluster_centers[cluster_index])
    cluster_centers[cluster_index] = cluster_centers[cluster_index].mean(dim = 0)

cluster_centers_tensor = []
for cluster_index in cluster_centers:
    cluster_centers_tensor.append(cluster_centers[cluster_index])
cluster_centers_tensor = torch.stack(cluster_centers_tensor)

f = open('/root/commonsense-kg-completion-master/data/atomic/Allen_nodes_cluster.json', 'w')
json.dump(nodes_cluster, f)
f.close()

f = open('/root/commonsense-kg-completion-master/data/atomic/Allen_nodes_index_to_cluster_index.json', 'w')
json.dump(nodes_index_to_cluster_index, f)
f.close()

# cluster_centers
torch.save(cluster_centers_tensor, '/root/commonsense-kg-completion-master/data/atomic/Allen_cluster_centers.pt')

print(cluster_centers_tensor)