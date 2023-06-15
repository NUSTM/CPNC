

"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with MultipleNegativesRankingLoss. Entailnments are poisitive pairs and the contradiction on AllNLI dataset is added as a hard negative.
At every 10% training steps, the model is evaluated on the STS benchmark dataset
Usage:
python training_nli_v2.py
OR
python training_nli_v2.py pretrained_transformer_model_name
"""
import math
from sentence_transformers import models, losses, datasets
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import random

datasets_name = 'ConceptNet'
#datasets_name = 'Atomic'
neg_num = 1
repeated_neg_num = 1

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#model_name = sys.argv[1] if len(sys.argv) > 1 else 'distilroberta-base'
model_name = 'bert-large-uncased'
train_batch_size = 128          #The larger you select this, the better the results (usually). But it requires more GPU memory
max_seq_length = 50
num_epochs = 1

# Save path of the model
model_save_path = f'../CP_model/{datasets_name}'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

# Here we define our SentenceTransformer model
print(model_name)
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


# Read the AllNLI.tsv.gz file and create the training dataset
logging.info("Read AllNLI train dataset")

print('read_train_data')
def add_to_samples(sent1, sent2, label):
    if sent1 not in train_data:
        train_data[sent1] = {'contradiction': set(), 'entailment': set()}
    train_data[sent1][label].add(sent2)


nodes = []
head_nodes = {}
if datasets_name == 'ConceptNet':
    f = open('../CPNC-I/data/ConceptNet/train.txt', 'r', encoding = 'utf-8')
elif datasets_name == 'Atomic':
    f = open('../CPNC-I/data/atomic/train.preprocessed.txt', 'r', encoding = 'utf-8')
train_txt = f.readlines()
for line in train_txt:
    line = line.replace('\n', '')
    line = line.split('\t')[:3]
    if datasets_name == 'ConceptNet':
            head = line[1]
    elif datasets_name == 'Atomic':
            head = line[0]
    tail = line[2]
    
    nodes.append(head)
    nodes.append(tail)
    
    if head not in head_nodes:
        head_nodes[head] = []
    
    head_nodes[head].append(tail)


#remove repeated nodes
for head in head_nodes:
    head_nodes[head] = set(head_nodes[head])

nodes = list(set(nodes))


train_data = {}

print('neg')

if datasets_name == 'ConceptNet':
    file_name = '../CPNC-I/data/ConceptNet/train.txt'
    val_file_name = '../CPNC-I/data/ConceptNet/valid.txt'
    test_file_name = '../CPNC-I/data/ConceptNet/test.txt'
elif datasets_name == 'Atomic':
    file_name = '../CPNC-I/data/atomic/train.preprocessed.txt'
    val_file_name = '../CPNC-I/data/atomic/valid.preprocessed.txt'
    test_file_name = '../CPNC-I/data/atomic/test.preprocessed.txt'
 
with open(file_name, 'r', encoding = 'utf-8') as f:
    train_txt = f.readlines()
    count = 0
    for line in train_txt:
        #print(count)
        count +=1
        line = line.replace('\n', '')
        line = line.split('\t')[:3]
        
        if datasets_name == 'ConceptNet':
            head = line[1]
        elif datasets_name == 'Atomic':
            head = line[0]
        tail = line[2]
        add_to_samples(head, tail, 'entailment')
        add_to_samples(tail, head, 'entailment')
        
        negs = random.choices(nodes, k = neg_num)
        for i, neg in enumerate(negs):
            if neg in head_nodes[head]:
                while 1:
                    neg = random.choice(nodes)
                    if neg not in set(head_nodes[head]):
                        break
            negs[i] = neg
            
        for neg in negs:
            add_to_samples(head, neg, 'contradiction')
            add_to_samples(neg, head, 'contradiction')
        
        
        #add_to_samples(sent2, sent1, row['label'])  #Also add the opposite


train_samples = []
for sent1, others in train_data.items():
    if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
        tails = list(others['entailment'])
        for i in range(len(tails)):
            for j in range(repeated_neg_num):
                train_samples.append(InputExample(texts=[sent1, tails[i], random.choice(list(others['contradiction']))]))
                train_samples.append(InputExample(texts=[tails[i], sent1, random.choice(list(others['contradiction']))]))


logging.info("Train samples: {}".format(len(train_samples)))


print('Special data loader that avoid duplicates within a batch')
# Special data loader that avoid duplicates within a batch
train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=train_batch_size)


print('Our training loss')
# Our training loss
train_loss = losses.MultipleNegativesRankingLoss(model)



print('read_test_data')

#Read STSbenchmark dataset and use it as development set
logging.info("Read STSbenchmark dev dataset")
dev_samples = []
with open(val_file_name, 'r', encoding = 'utf-8') as f:
    train_txt = f.readlines()
    for line in train_txt:
        line = line.replace('\n', '')
        line = line.split('\t')[:3]
        
        if datasets_name == 'Atomic':
            head = line[0]
        elif datasets_name == 'ConceptNet':
            head = line[1]
        tail = line[2]
        
        dev_samples.append(InputExample(texts=[head, tail], label=float(1)))
        dev_samples.append(InputExample(texts=[tail, head], label=float(1)))
        neg = random.choice(nodes)
        dev_samples.append(InputExample(texts=[head, neg], label=float(0)))
        neg = random.choice(nodes)
        dev_samples.append(InputExample(texts=[tail, neg], label=float(0)))

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')

# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

print('start training')
# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=int(len(train_dataloader)*0.05),
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=False          #Set to True, if your GPU supports FP16 operations
          )



##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################
test_samples = []
with open(test_file_name, 'r', encoding = 'utf-8') as f:
    train_txt = f.readlines()
    for line in train_txt:
        line = line.replace('\n', '')
        line = line.split('\t')[:3]
        
        if datasets_name == 'Atomic':
            head = line[0]
        elif datasets_name == 'ConceptNet':
            head = line[1]
        tail = line[2]
        
        test_samples.append(InputExample(texts=[head, tail], label=float(1)))
        neg = random.choice(nodes)
        test_samples.append(InputExample(texts=[head, neg], label=float(0)))

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')
test_evaluator(model, output_path=model_save_path)