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

def load_json(path):
    f = open(path, 'r')
    data = json.load(f)
    f.close()

    return data

f = open('./CPNC-I/data/atomic/atomic_node_names.txt', 'r', encoding = 'utf-8')
data = f.readlines()
nodes = []
for line in data:
    nodes.append(line.replace('\n', ''))
    #print(line.replace('\n'))
f.close()

print(len(nodes))


EB_forward = torch.load('./bert_model_embeddings/nodes-lm-atomic/atomic_bert_embeddings.pt')


num_clusters = 900
x = EB_forward

# kmeans
nodes_index, cluster_centers = kmeans(
   X=x, num_clusters=num_clusters, distance='cosine', device=torch.device('cpu')
)

print(nodes_index.size())
print(cluster_centers)

nodes_cluster = {}
for i in range(nodes_index.size(0)):
   node = nodes[i]
   nodes_cluster[node] = nodes_index[i].item()

nodes_index_to_cluster_index = []
for node in nodes:
    nodes_index_to_cluster_index.append(nodes_cluster[node])

f = open('./Concept_Centre/atomic/nodes_cluster.json', 'w')
json.dump(nodes_cluster, f)
f.close()

f = open('./Concept_Centre/atomic/nodes_index_to_cluster_index.json', 'w')
json.dump(nodes_index_to_cluster_index, f)
f.close()

# cluster_centers
torch.save(cluster_centers, './Concept_Centre/atomic/cluster_centers.pt')

print(cluster_centers)