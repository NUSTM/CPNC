import numpy as np
import torch

import math
import json
import logging
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt


#######################################################################
# Utility functions for evaluation
#######################################################################


def sort_and_rank(score, target):
    sorted, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


def get_filtered_ranks(score, target, batch_a, batch_r, e1_to_multi_e2):
    filtered_scores = score.clone().detach()

    for i, t in enumerate(target):
        filter_ents = e1_to_multi_e2[(batch_a[i].item(), batch_r[i].item())]

        # these filters contain ALL labels
        target_value = filtered_scores[i][t].clone()
        # zero all known cases => corresponds to the filtered setting
        filtered_scores[i][filter_ents] = 0.0
        assert t in filter_ents
        # write base the saved values
        filtered_scores[i][t] = target_value

    return sort_and_rank(filtered_scores, target)


def perturb_and_get_rank(model, embedding, w, a, r, b, e1_to_multi_e2,
                         num_entity, batch_size=128, perturbed="subj"):
    """
        Perturb one element in the triplets
    """

    num_triples = len(a)
    n_batch = math.ceil(num_triples / batch_size)
    gold_scores = []
    ranks = []
    filtered_ranks = []

    for idx in range(n_batch):
        print("batch {} / {}".format(idx, n_batch), end="\r")
        batch_start = idx * batch_size
        batch_end = min(num_triples, (idx + 1) * batch_size)
        batch_a = a[batch_start: batch_end]
        batch_r = r[batch_start: batch_end]
        emb_ar = embedding[batch_a] * w[batch_r]
        emb_ar = emb_ar.transpose(0, 1).unsqueeze(2)  # size: D x E x 1
        emb_c = embedding.transpose(0, 1).unsqueeze(1)  # size: D x 1 x V
        # out-prod and reduce sum
        out_prod = torch.bmm(emb_ar, emb_c)  # size D x E x V
        score = torch.sum(out_prod, dim=0)  # size E x V
        score = torch.sigmoid(score)
        target = b[batch_start: batch_end]
        gold_score = torch.FloatTensor([score[i][idx] for i, idx in enumerate(target)])
        ranks.append(sort_and_rank(score, target))
        gold_scores.append(gold_score)
        filtered_ranks.append(get_filtered_ranks(score, target, batch_a, batch_r, e1_to_multi_e2, perturbed))

    return torch.cat(ranks), torch.cat(filtered_ranks), torch.cat(gold_scores)


def perturb_and_get_rank_conv(model, embedding, w, a, r, b, e1_to_multi_e2,
                              num_entity, batch_size=128, perturbed="subj"):
    """
        Perturb one element in the triplets for a convolution-based decoder
    """

    num_triples = len(a)
    n_batch = math.ceil(num_triples / batch_size)
    gold_scores = []
    ranks = []
    filtered_ranks = []

    for idx in range(n_batch):
        print("batch {} / {}".format(idx, n_batch), end="\r")
        batch_start = idx * batch_size
        batch_end = min(num_triples, (idx + 1) * batch_size)
        batch_a = a[batch_start: batch_end]
        batch_r = r[batch_start: batch_end]
        with torch.no_grad():
            score = model.calc_score(batch_a, batch_r)

        target = b[batch_start: batch_end]
        gold_score = torch.FloatTensor([score[i][idx] for i, idx in enumerate(target)])
        ranks.append(sort_and_rank(score, target))
        gold_scores.append(gold_score)
        filtered_ranks.append(get_filtered_ranks(score, target, batch_a, batch_r, e1_to_multi_e2, perturbed))

    return torch.cat(ranks), torch.cat(filtered_ranks), torch.cat(gold_scores)


def ranking_and_hits(test_graph, model, test_triplets, e1_to_multi_e2, network, fusion="graph-only",
                     sim_relations=False, write_results=False, debug=False, epoch=None):
    print(model)
    
    model.eval()

    s = test_triplets[:, 0]
    r = test_triplets[:, 1]
    o = test_triplets[:, 2]
    
    
    if fusion == "sum":
        gembedding, tail_index = model.evaluate(test_graph)
        #gembedding = gembedding.cuda()
        #init_embedding = model.rgcn.layers[0].embedding.weight
        with torch.no_grad():
            #embedding = gembedding + init_embedding
            embedding = gembedding

    elif fusion == "init":
        embedding = model.rgcn.layers[0].embedding.weight

    elif fusion == "graph-only":
        embedding, tail_index = model.evaluate(test_graph, epoch)
        embedding = embedding.cuda()

    if sim_relations:
        rel_offset = model.num_rels - 1
    else:
        rel_offset = model.num_rels

    #model.decoder.module.cur_embedding = embedding
    model.decoder.cur_embedding = embedding
    model.decoder.tail_index = tail_index

    #hits_left = []
    #hits_right = []
    hits = []
    ranks = 0
    ranks_left = 0
    ranks_right = 0
    scores = []
    node_mrr = {}
    mrr = 0
    
    count_ranks = 0
    count_mrr = 0
    count_left_right = 0
    

    for i in range(10):
        #hits_left.append([])
        #hits_right.append([])
        hits.append(0)

    batch_size = 128

    if debug:
        end = min(5000, len(test_triplets))
    else:
        end = len(test_triplets)

    # for i in range(0, len(test_triplets), batch_size):
    batch_size = 32
    for i in range(0, end, batch_size):
        hits_tmp = []
        ranks_tmp = []
        ranks_left_tmp = []
        ranks_right_tmp = []
        scores_tmp = []
        for l in range(10):
            hits_tmp.append([])
    
        e1 = s[i: i + batch_size]
        e2 = o[i: i + batch_size]
        rel = r[i: i + batch_size]
        rel_reverse = rel + rel_offset
        cur_batch_size = len(e1)

        e2_multi1 = [torch.LongTensor(e1_to_multi_e2[(e.cpu().item(), r.cpu().item())]).cuda() for e, r in zip(e1, rel)] 
        e2_multi2 = [torch.LongTensor(e1_to_multi_e2[(e.cpu().item(), r.cpu().item())]).cuda() for e, r in
                     zip(e2, rel_reverse)] 

        with torch.no_grad():
            pred1 = model.calc_score(e1, rel.cuda())
            pred2 = model.calc_score(e2, rel_reverse.cuda())
        #pred1, pred2 = pred1.cpu(), pred2.cpu()
        
        pred1, pred2 = pred1.data, pred2.data
        scores_tmp.append(pred1.cpu())
        e1, e2 = e1.data, e2.data

        for j in range(0, cur_batch_size):
            # these filters contain ALL labels
            filter1 = e2_multi1[j].long()
            filter2 = e2_multi2[j].long()
            # save the prediction that is relevant
            target_value1 = pred1[j, e2[j].item()].item()
            target_value2 = pred2[j, e1[j].item()].item()
            # zero all known cases (this are not interesting)
            # this corresponds to the filtered setting
            pred1[j][filter1] = 0.0
            pred2[j][filter2] = 0.0

            # EXP: also remove self-connections
            pred1[j][e1[j].item()] = 0.0
            pred2[j][e2[j].item()] = 0.0

            # write base the saved values
            pred1[j][e2[j]] = target_value1
            pred2[j][e1[j]] = target_value2
            
            del target_value1
            del target_value2
        del e2_multi1
        del e2_multi2
        
        # sort and rank
        max_values, argsort1 = torch.sort(pred1, 1, descending=True)
        del max_values
        max_values, argsort2 = torch.sort(pred2, 1, descending=True)
        del max_values

        for j in range(0, cur_batch_size):

            # find the rank of the target entities
            # rank1 = np.where(argsort1[i]==e2[i].item())[0][0]
            # rank2 = np.where(argsort2[i]==e1[i].item())[0][0]
            rank1 = (argsort1[j] == e2[j]).nonzero().cpu().item()
            rank2 = (argsort2[j] == e1[j]).nonzero().cpu().item()

            # rank+1, since the lowest rank is rank 1 not rank 0
            ranks_tmp.append(rank1 + 1)
            ranks_left_tmp.append(rank1 + 1)
            ranks_tmp.append(rank2 + 1)
            ranks_right_tmp.append(rank2 + 1)


            for hits_level in [0, 2, 9]:
                #if hits_level in [0, 2, 9]:
                if rank1 <= hits_level:
                    hits_tmp[hits_level].append(1.0)
                    #hits_left[hits_level].append(1.0)
                else:
                    hits_tmp[hits_level].append(0.0)
                    #hits_left[hits_level].append(0.0)

                if rank2 <= hits_level:
                    hits_tmp[hits_level].append(1.0)
                    #hits_right[hits_level].append(1.0)
                else:
                    hits_tmp[hits_level].append(0.0)
                    #hits_right[hits_level].append(0.0)
        
        for hits_level in [0, 2, 9]:
             hits[hits_level] += sum(hits_tmp[hits_level])
        ranks += sum(ranks_tmp)
        ranks_left += sum(ranks_left_tmp)
        ranks_right += sum(ranks_right_tmp)
        mrr += np.sum(1 / np.array(ranks_tmp))
        
        argsort1, argsort2 = argsort1.cpu(), argsort2.cpu()
        
        count_ranks += len(ranks_tmp)
        count_mrr += len(ranks_tmp)
        count_left_right += len(ranks_right_tmp)
        
        
        del argsort1
        del argsort2
        del rank1
        del rank2
        del pred1
        del pred2
        del e1
        del e2
        del rel
        
    for k in range(0, 10):
        if k in [0, 2, 9]:
        #logging.info('Hits left @{0}: {1}'.format(k + 1, np.mean(hits_left[k])))
        #logging.info('Hits right @{0}: {1}'.format(k + 1, np.mean(hits_right[k])))
            logging.info('Hits @{0}: {1}'.format(k + 1, hits[k] / count_ranks))
    logging.info('Mean rank left: {0}'.format(ranks_left / count_left_right ))
    logging.info('Mean rank right: {0}'.format(ranks_right / count_left_right ))
    logging.info('Mean rank: {0}'.format(ranks / count_ranks))
    #logging.info('Mean rank: {0}'.format(mrr / count_mrr))
    logging.info('Mean reciprocal rank: {0}'.format(mrr / count_mrr))
    print('MRR:', mrr / count_mrr)
    for k in range(0, 10):
        if k in [0, 2, 9]:
            print("hit@", k, ':', hits[k] / count_ranks)

    #if write_results:
    #    write_topk_tuples(torch.cat(scores, dim=0).cpu().numpy(), test_triplets, network)

    # plot_degree_mrr(node_mrr)

    return mrr / count_mrr


def plot_degree_mrr(node_ranks):
    degree_rank = {}
    for node, rank in node_ranks.items():
        node_degree = node.get_degree()
        if node_degree not in degree_rank:
            degree_rank[node_degree] = []
        degree_rank[node_degree].append(sum(rank) / len(rank))

    degrees = []
    ranks = []
    for k in sorted(degree_rank.keys()):
        if k < 20:
            # degrees.append(k)
            # ranks.append(sum(degree_rank[k])/len(degree_rank[k]))
            for rank in degree_rank[k]:
                if rank < 100:
                    degrees.append(k)
                    ranks.append(rank)

    fig, ax = plt.subplots()

    ax.scatter(degrees, ranks, marker='.')
    ax.set(xlabel="degree", ylabel="mean ranks")
    ax.grid()
    fig.savefig("comet_cn_degree_ranks.png")


def write_topk_tuples(scores, input_prefs, network, k=50):
    out_lines = []

    argsort = [np.argsort(-1 * np.array(score)) for score in np.array(scores)]

    for i, sorted_scores in enumerate(argsort):

        pref = input_prefs[i]
        e1 = pref[0].cpu().item()
        rel = pref[1].cpu().item()
        e2 = pref[2].cpu().item()
        cur_point = {}
        cur_point['gold_triple'] = {}
        cur_point['gold_triple']['e1'] = network.graph.nodes[e1].name
        cur_point['gold_triple']['e2'] = network.graph.nodes[e2].name
        cur_point['gold_triple']['relation'] = network.graph.relations[rel].name

        topk_indices = sorted_scores[:k]
        topk_tuples = [network.graph.nodes[elem] for elem in topk_indices]
        # if golds[i] in topk_tuples:
        #    topk_indices = argsort[i][:k+1]
        #    topk_tuples = [input_batch[i][elem] for elem in topk_indices if input_batch[i][elem]!=golds[i]]
        cur_point['candidates'] = []

        for j, node in enumerate(topk_tuples):
            tup = {}
            tup['e1'] = network.graph.nodes[e1].name
            tup['e2'] = node.name
            tup['relation'] = network.graph.relations[rel].name
            tup['score'] = str(scores[i][topk_indices[j]])
            cur_point['candidates'].append(tup)

        out_lines.append(cur_point)

    with open("topk_candidates.jsonl", 'w') as f:
        for entry in out_lines:
            json.dump(entry, f)
            f.write("\n")
