# Commonsense Knowledge Graph Completion Via Contrastive Pretraining and Node Clustering


Code for the paper [Commonsense Knowledge Graph Completion Via Contrastive Pretraining and Node Clustering](https://arxiv.org/pdf/2305.17019.pdf). 

## Bibtex

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

## Requirements

- PyTorch
- Run `pip install -r requirements.txt` to install the required packages.



## Training 

### Data

The dataset, semantic embedding of node and nodes clustering data are provided in [CPNC-S-data](https://pan.baidu.com/s/1AyUGZlUT2-ri2Sj5w_6Yzg?pwd=om9e ) and [CPNC-I-data ](https://pan.baidu.com/s/1DyMJs_lumfAf5GsPELsEEg?pwd=xe4j ). Plese donwload it and unzip the file under CPNC-S and CPNC-I, respectively. Because we didn't set the random seed in Constrastive Pretraining and Nodes Clustering, so we advice you use the embedding we have obtained.

### Semantic Embedding and Node Clustering

 You can rerun the result as following:

```bash
cd CPNC/CPNC-S
```

Train Constrastive Pretraining Model for ATOMIC and CN-100K datasets:

```bash
python Contrastive_Pretraining_ATOMIC.py
python Contrastive_Pretraining_ConceptNet.py
```

This trains the model and saves the model under the `/CP_model/ATOMIC/`  and `/CP_model/ConceptNet/` directory. Based on it, you can get the semantic embedding of node :

```bash
python get_nodes_embedding_Atomic.py
python get_nodes_embedding_ConceptNet.py
```

the semantic embedding of node are saved in `/bert_model_embeddings/nodes-lm-atomic/`  and `/bert_model_embeddings/nodes-lm-conceptnet/` directory.

Then, clustering nodes to obtain the latent concept:

```bash
python K_means_Atomic.py
python K_means_ConceptNet.py
```

The nodes cluster result are saved in `/Concept_Centre/atomic/`  and `/Concept_Centre/ConceptNet/`.

### CPNC-S Model

To train the CPNC-S model ,  you need to run the following command:

```bash
cd CPNC/CPNC-S
```

Finally, in order to train the CPNC-S model on CN-100K, run the following command:

```bash
python -u src/run_kbc_subgraph.py --dataset conceptnet --evaluate-every 10 --n-layers 2 --graph-batch-size 60000  --bert_concat --Concept_center_path './Concept_Centre/ConceptNet/'
```

In order to train the CPNC-S model on ATOMIC, run the following command:

```bash
python -u src/run_kbc_subgraph.py --dataset atomic  --evaluate-every 10 --n-layers 2 --graph-batch-size 20000  --bert_concat --Concept_center_path './Concept_Centre/atomic/'
```

This trains the model and saves the model under the`./saved_models_ConceptNet/` and  `./saved_models_ATOMIC/` directory.

### CPNC-I Model

To train the CPNC-I model ,  you need to run the following command:

```bash
cd CPNC/CPNC-I
```

Then, in order to train the CPNC-I model on CN-100K, run the following command:

```bash
bash train.sh conceptnet-100k 15 saved/saved_ckg_model data/saved_entity_embedding/conceptnet/cn_bert_emb_dict.pkl 500 256 100 ConvTransE 10 1234 1e-20 0.25 0.25 0.25 0.0003 1024 Adam 5 300 RWGCN_NET 50000 1324 data/saved_entity_embedding/conceptnet/cn_fasttext_dict.pkl 300 0.2 5 100 50 0.1 ./Concept_Centre/ConceptNet/
```

In order to train the CPNC-S model on ATOMIC, run the following command:

```bash
bash train.sh atomic 500 saved/saved_ckg_model data/saved_entity_embedding/atomic/at_bert_emb_dict.pkl 500 256 100 ConvTransE 10 1234 1e-20 0.20 0.20 0.20 0.0001 1024 Adam 5 300 RWGCN_NET 50000 1324 data/saved_entity_embedding/atomic/at_fasttext_dict.pkl 300 0.2 3 100 50 0.1 ./Concept_Centre/atomic/python -u src/run_kbc_subgraph.py --dataset atomic  --evaluate-every 10 --n-layers 2 --graph-batch-size 20000  --bert_concat --Concept_center_path './Concept_Centre/atomic/'
```

This trains the model and saves the model under the `saved/saved_ckg_model/` directory.

