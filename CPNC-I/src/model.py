import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
import torch.nn.functional as F

import encoder
from decoder import ConvTransE

from sklearn.decomposition import PCA

import json

def load_json(path):
    f = open(path, 'r')
    data = json.load(f)
    f.close()

    return data

class IndexEB(nn.Module):
    def __init__(self, index_EB):
        self.cluster_index = index_EB
    
    def forward(self, index):
        return self.cluster_index[index]

def mask_by_schedule(tensor, epoch, epoch_cutoff=100):  
    if epoch < epoch_cutoff:
        cuda_check = tensor.is_cuda

        if cuda_check:
            mask = torch.zeros((tensor.size(0), tensor.size(1)), device='cuda')
        else:
            mask = torch.zeros((tensor.size(0), tensor.size(1)))

        k = int((epoch / epoch_cutoff) * tensor.size(1))
        perm = torch.randperm(tensor.size(1))
        indices = perm[:k]
        mask[:, indices] = 1
        return tensor * mask
    else:
        return tensor

class LinkPredictor(nn.Module):
    def __init__(self, args):
        super(LinkPredictor, self).__init__()
        
        #self.fc = torch.nn.Linear(1024, 500)
        #self.feature_map_drop_fusion = torch.nn.Dropout(0.2)
        
        # Parameters
        self.num_nodes = args.num_nodes
        #self.decoder_rels = args.num_rels * 2 # account for inverse relations
        self.decoder_rels = args.num_rels * 2 + 1 # account for inverse relations

        self.decoder_embedding_dim = args.decoder_embedding_dim
        self.rel_regularization = args.rel_regularization
        self.device = args.device

        self.encoder_name = args.encoder
        self.decoder_name = args.decoder

        # Entity & Rel Embedding
        self.w_relation = torch.nn.Embedding(self.decoder_rels, self.decoder_embedding_dim, padding_idx=0)
        self.entity_embedding = None
        self.entity_cluster_embedding = None
        
        cluster_EB = torch.load(f'{args.concept_center_path}cluster_centers.pt')

        self.cluster_embedding = torch.nn.Embedding.from_pretrained(cluster_EB)
        
        cluster_index = load_json(f'{args.concept_center_path}nodes_index_to_cluster_index.json')
        cluster_index_copy= []
        for i in range(len(cluster_index)):
            cluster_index_copy.append(0)
        self.register_buffer('cluster_index' , torch.tensor(cluster_index) )
        
        # Encoder
        if self.encoder_name == 'RWGCN_NET':
            self.encoder = encoder.RWGCN_NET(args)
        elif self.encoder_name == 'identity_emb':
            self.encoder = encoder.identity_emb(args)
        else:
            print('Encoder not found: ', self.encoder_name)

        # Decoder
        if self.decoder_name == "ConvTransE":
            self.decoder = ConvTransE(self.num_nodes, self.decoder_rels, args)
        else:
            print('Decoder not found: ', self.decoder_name)

        self.loss = torch.nn.BCELoss(reduction='none')
        self.init()

    def forward(self, e1, rel, target=None, sample_normaliaztion=None):

        e1_embedding = self.entity_embedding[e1]
        e1_cluster_embedding = self.entity_cluster_embedding[e1]
        rel_embedding = self.w_relation(rel)

        # Decoder
        pred = self.decoder(e1_embedding, rel_embedding, self.entity_embedding,  e1_cluster_embedding, self.entity_cluster_embedding)

        if target is None:
            return pred
        else:
            pred_loss = self.loss(pred, target)
            pred_loss = torch.mean(pred_loss, dim=1)
            
            if sample_normaliaztion != None:
                pred_loss = pred_loss * sample_normaliaztion
            pred_loss = torch.mean(pred_loss, dim=0)

            reg_loss = self.regularization_loss()
            if self.rel_regularization != 0.0:
                reg_loss = self.regularization_loss()
                return pred_loss + self.rel_regularization * reg_loss
            else:
                return pred_loss
            

    def init(self):

        xavier_normal_(self.w_relation.weight.data)

    def regularization_loss(self):
        
        reg_w_relation = self.w_relation.weight.pow(2)
        reg_w_relation = torch.mean(reg_w_relation)

        return reg_w_relation + self.encoder.en_regularization_loss()


    def update_whole_embedding_matrix(self, g, node_id_copy, epoch = None):
        
        if self.encoder_name == 'RWGCN_NET':
            gnn_embs = self.encoder.forward(g, node_id_copy)
            self.entity_embedding = gnn_embs
        elif self.encoder_name == 'identity_emb':
            self.entity_embedding = self.encoder.entity_embedding.weight
        else:
            print('Encoder not found: ', self.encoder_name)
        entity_index = self.cluster_index[node_id_copy]
        
        self.entity_cluster_embedding = self.cluster_embedding(entity_index)
        if epoch != None:
            self.entity_embedding = mask_by_schedule(self.entity_embedding, epoch)
            self.entity_cluster_embedding = mask_by_schedule(self.entity_cluster_embedding, epoch)