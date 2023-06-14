import copy
import json
import pytorch_lightning as pl
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer, BertModel
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
import numpy as np
import os

f = open('../CPNC-I/data/ConceptNet/cn_node_names.txt', 'r', encoding = 'utf-8')
data_nodes = f.readlines()
nodes = []
for line in data_nodes:
    nodes.append(line.replace('\n', ''))
f.close()

model_save_path = '../CP_model/ConceptNet/'
sentence_model = SentenceTransformer(model_save_path)
all_nodes_features = sentence_model.encode(nodes)
all_nodes_features = torch.from_numpy(all_nodes_features)
#all_nodes_features = torch.nn.Embedding.from_pretrained(all_nodes_features)
print(type(all_nodes_features))
print(all_nodes_features.size())
torch.save(all_nodes_features, '../bert_model_embeddings/nodes-lm-conceptnet/conceptnet_bert_embeddings.pt', _use_new_zipfile_serialization=False)