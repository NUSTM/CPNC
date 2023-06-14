import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True

class ConvTransE(nn.Module):
    def __init__(self, num_entities, num_relations, args):
        super(ConvTransE, self).__init__()

        """
        Difference from ConvE: no reshaping after stacking e_1 and e_r
        """
        # Hyperparameters
        self.embedding_dim = args.decoder_embedding_dim        

        # Dropouts
        self.inp_drop = torch.nn.Dropout(args.input_dropout)
        self.hidden_drop = torch.nn.Dropout(args.dropout)
        self.feature_map_drop = torch.nn.Dropout(args.feature_map_dropout)

        # Convolution
        self.kernel_size = args.dec_kernel_size
        self.channels = args.dec_channels

        self.num_groups = 1
        self.conv1 = nn.Conv1d(2, self.channels, self.kernel_size, stride=1, padding= int(math.floor(self.kernel_size/2)), groups=1)
        
        self.feature_map_drop_fusion = torch.nn.Dropout(0.1)
        
        self.bn0 = torch.nn.BatchNorm1d(2)
        
        self.bn1 = torch.nn.BatchNorm1d(self.channels)
        self.bn2 = torch.nn.BatchNorm1d(self.embedding_dim)

        self.fc = torch.nn.Linear(self.channels * self.embedding_dim * self.num_groups, self.embedding_dim)
        self.fc_fusion_R_C_hidden = torch.nn.Linear(500 + 1024, 500)
        self.fc_fusion_T_C_hidden = torch.nn.Linear(500 + 1024, 500)
        

        self.random_permutation = torch.LongTensor([np.random.permutation(self.embedding_dim) for _ in range(self.num_groups)])



    def forward(self, e1_embedding, rel_embedding, entity_embedding, e1_cluster_embedding, entity_cluster_embedding, encoding=False):
        
        batch_size = e1_embedding.shape[0]

        e1_embedding = e1_embedding#.unsqueeze(1)
        rel_embedding = rel_embedding#.unsqueeze(1)

        e1_embedding = e1_embedding[:,self.random_permutation]        
        rel_embedding = rel_embedding[:,self.random_permutation]

        e1_embedding = e1_embedding.reshape((batch_size,-1)).unsqueeze(1)
        rel_embedding = rel_embedding.reshape((batch_size,-1)).unsqueeze(1)
        
        
        rel_embedding = rel_embedding.squeeze(1)
        
        
        rel_clu_embedded = torch.cat([rel_embedding, e1_cluster_embedding/15] , 1)
        rel_embedded = self.fc_fusion_R_C_hidden(rel_clu_embedded)
        rel_embedded = self.feature_map_drop_fusion(rel_embedded)
        rel_embedding = rel_embedded.unsqueeze(1)
        
        stacked_inputs = torch.cat([e1_embedding, rel_embedding], dim=1)

        stacked_inputs = self.bn0(stacked_inputs)


        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)

        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)

        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)

        if encoding == True:
            return x
            
        entity_embedding = entity_embedding.squeeze(1)
        tail_cluster_embedding = torch.cat([entity_embedding, entity_cluster_embedding/15], 1)
        entity_embedding = self.fc_fusion_T_C_hidden(tail_cluster_embedding)
        entity_embedding = self.feature_map_drop_fusion(entity_embedding)
        
        x = torch.mm(x, entity_embedding.t())

        pred = torch.sigmoid(x)

        return pred
