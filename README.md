# Commonsense Knowledge Graph Completion Via Contrastive Pretraining and Node Clustering


Codes for the paper [Commonsense Knowledge Graph Completion Via Contrastive Pretraining and Node Clustering](https://arxiv.org/pdf/2305.17019.pdf). 

## Requirements

- python=3.10.6
- cuda=11.6
- conda install -c dglteam dgl-cuda11.6 
- Run `pip install -r requirements.txt` to install the required packages.

## Training 

The dataset are provided in [CPNC-S-data](https://pan.baidu.com/s/1bHQT9fHtvlgUHf-4NhEmcA?pwd=jgym  ) and [CPNC-I-data ](https://pan.baidu.com/s/1K7pFff0zrxMzpDhmBSHR8Q?pwd=rxpv ). Plese download it and unzip the file under CPNC-S and CPNC-I, respectively.

### Constrastive Pretraining

Train Constrastive Pretraining Model for ATOMIC and CN-100K datasets:

```bash
python Contrastive_Pretraining_ATOMIC.py  # training CP model for ATOMIC  datasets
python Contrastive_Pretraining_ConceptNet.py  # training CP model for CN-100K datasets
```

The trained model are saved under the `./CP_model/ATOMIC/`  and `./CP_model/ConceptNet/` directory. Use the trained model to get the semantic embeddings of nodes :

```bash
python get_nodes_embedding_Atomic.py  # get the semantic embeddings of nodes in ATOMIC 
python get_nodes_embedding_ConceptNet.py  # get the semantic embeddings of nodes in CN-100K
```

the semantic embeddings of nodes are saved in `./bert_model_embeddings/nodes-lm-atomic/`  and `./bert_model_embeddings/nodes-lm-conceptnet/` directory.

The CPNC-I model need the extra fasttext embeddings of nodes, and need to transfor the semantic embeddings of nodes to a dict format.

We provide those embeddings: [saved_entity_embedding]( https://pan.baidu.com/s/1tb_VDern8FO2NI8FwNaNpg?pwd=c8c5).

###  Node Clustering

Then, performing nodes clustering:

```bash
python K_means_Atomic.py  # get the semantic embeddings of nodes in ATOMIC
python K_means_ConceptNet.py  # get the semantic embeddings of nodes in CN-100K
```

The nodes clustering results are saved in `./Concept_Centre/atomic/`  and `./Concept_Centre/ConceptNet/`.

We provide our nodes clustering results: [Concept_Centre](https://pan.baidu.com/s/1AiX-wfZTDiB9lcaZJ2pSsQ?pwd=legy ).

### Completion CSKG

For reproducing the results in our paper, please download the semantic embedding of nodes and nodes clustering result data and unzip it under CPNC-S and CPNC-I, respectively.

#### CPNC-S Model

To train the CPNC-S model ,  you need to run the following command:

```bash
cd CPNC/CPNC-S
```

Finally, in order to train the CPNC-S model on CN-100K, run the following command:

```bash
python -u src/run_kbc_subgraph.py --dataset conceptnet --evaluate-every 10 --n-layers 2 --graph-batch-size 60000  --bert_concat --Concept_center_path '../Concept_Centre/ConceptNet/'
```

In order to train the CPNC-S model on ATOMIC, run the following command:

```bash
python -u src/run_kbc_subgraph.py --dataset atomic  --evaluate-every 10 --n-layers 2 --graph-batch-size 20000  --bert_concat --Concept_center_path '../Concept_Centre/atomic/'
```

This trains the model and saves the model under the`./saved_models_ConceptNet/` and  `./saved_models_ATOMIC/` directory.

#### CPNC-I Model

To train the CPNC-I model ,  you need to run the following command:

```bash
cd CPNC/CPNC-I
```

Then, in order to train the CPNC-I model on CN-100K, run the following command:

```bash
bash train.sh conceptnet-100k 15 saved/saved_ckg_model saved_entity_embedding/conceptnet/cn_bert_emb_dict.pkl 500 256 100 ConvTransE 10 1234 1e-20 0.25 0.25 0.25 0.0003 1024 Adam 5 300 RWGCN_NET 50000 1324 ../bert_model_embeddings/nodes-lm-conceptnet/cn_fasttext_dict.pkl 300 0.2 5 100 50 0.1 ../Concept_Centre/ConceptNet/
```

In order to train the CPNC-S model on ATOMIC, run the following command:

```bash
bash train.sh atomic 500 saved/saved_ckg_model saved_entity_embedding/atomic/at_bert_emb_dict.pkl 500 256 100 ConvTransE 10 1234 1e-20 0.20 0.20 0.20 0.0001 1024 Adam 5 300 RWGCN_NET 50000 1324 ../bert_model_embeddings/nodes-lm-atomic/at_fasttext_dict.pkl 300 0.2 3 100 50 0.1 ../Concept_Centre/atomic/
```

This trains the model and saves the model under the `./saved/saved_ckg_model/` directory.

## Citation

```
@article{DBLP:journals/corr/abs-2305-17019,
  author       = {Siwei Wu and
                  Xiangqing Shen and
                  Rui Xia},
  title        = {Commonsense Knowledge Graph Completion Via Contrastive Pretraining
                  and Node Clustering},
  journal      = {CoRR},
  volume       = {abs/2305.17019},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2305.17019},
  doi          = {10.48550/arXiv.2305.17019},
  eprinttype    = {arXiv},
  eprint       = {2305.17019},
  timestamp    = {Wed, 07 Jun 2023 14:31:13 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2305-17019.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
