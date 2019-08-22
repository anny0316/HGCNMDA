from __future__ import division
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import random
import os, sys, pdb, math, csv
import scipy.io as sio
import warnings
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
import pandas as pd
import seaborn as sns

warnings.simplefilter('ignore', sp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s/software/node2vec/src' % cur_dir)
import node2vec

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_bio_data(dataset_str):
    data = sio.loadmat('data/NS.mat')
    net = data['net']


def sample_neg(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None):
    # get upper triangular matrix
    net_triu = sp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = sp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None or test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num, test_num = len(train_pos[0]), len(test_pos[0])
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    while len(neg[0]) < train_num + test_num:
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        if i < j and net[i, j] == 0:
            neg[0].append(i)
            neg[1].append(j)
        else:
            continue
    train_neg  = (neg[0][:train_num], neg[1][:train_num])
    test_neg = (neg[0][train_num:], neg[1][train_num:])
    return train_pos, train_neg, test_pos, test_neg


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        test = np.vstack((mx.row, mx.col))
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features_g, support, adj_m2g, adj_d2g, adj_d2m, lm, ld, labels, positive_train_label_row, neg_train_label_row, placeholders):
    """Construct feed dictionary. features, support, y_train, train_mask, placeholders"""
    feed_dict = dict()
    feed_dict.update({placeholders['adj_m2g']: adj_m2g})
    feed_dict.update({placeholders['adj_d2g']: adj_d2g})
    feed_dict.update({placeholders['adj_d2m']: adj_d2m})
    feed_dict.update({placeholders['features_g']: np.array(features_g)})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features_g[1].shape})
    feed_dict.update({placeholders['mirna_length']: lm})
    feed_dict.update({placeholders['diease_length']: ld})
    feed_dict.update({placeholders['labels_row']:labels})
    feed_dict.update({placeholders['positive_train_label_row']:positive_train_label_row})
    feed_dict.update({placeholders['neg_train_label_row']: neg_train_label_row})
    # feed_dict.update({placeholders['num_features_diease_nonzero']: features_diease[1].shape})
    # feed_dict.update({placeholders['features_diease']: np.array(features_diease)})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

def generate_node2vec_embeddings(A, emd_size=128, negative_injection=False):     #, train_neg=None
    # if negative_injection:
    #     row, col = train_neg
    #     A = A.copy()
    #     A[row, col] = 1  # inject negative train
    #     A[col, row] = 1  # inject negative train
    nx_G = nx.from_scipy_sparse_matrix(A)
    G = node2vec.Graph(nx_G, is_directed=False, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=emd_size, window=10, min_count=0, sg=1, workers=8, iter=1)
    wv = model.wv
    embeddings = np.zeros([A.shape[0], emd_size], dtype='float32')
    sum_embeddings = 0
    empty_list = []
    for i in range(A.shape[0]):
        if str(i) in wv:
            embeddings[i] = wv.word_vec(str(i))
            sum_embeddings += embeddings[i]
        else:
            empty_list.append(i)
    mean_embedding = sum_embeddings / (A.shape[0] - len(empty_list))
    embeddings[empty_list] = mean_embedding
    return embeddings

def bio_y_label(train_pos,train_neg):
    edge_lable = []
    y_label = []
    edge = []
    for i in range(len(train_pos[0])):
        edge_lable.append([train_pos[0][i], train_pos[1][i], 1])
    for i in range(len(train_neg[0])):
        edge_lable.append([train_neg[0][i], train_neg[1][i], 0])
    random.shuffle(edge_lable)
    for i in range(len(edge_lable)):
        edge.append([edge_lable[i][0],edge_lable[i][1]])
        y_label.append(edge_lable[i][2])

    label = np.array(y_label)
    classes = max(label) + 1
    one_hot_label = np.zeros(shape=(label.shape[0], classes))
    one_hot_label[np.arange(0, label.shape[0]), label] = 1
    return edge,one_hot_label

def bio_mask(y_label):
    train_mask = np.zeros(len(y_label))
    for i in range(len(y_label)):
        train_mask[i] = y_label[i][1]
    return np.array(train_mask, dtype=np.bool)

def load_ppi(fname='data/bio-ppi.csv'):
    fin = open(fname)
    print 'Reading: %s' % fname
    fin.readline()
    edges = []
    for line in fin:
        gene_id1, gene_id2= line.strip().split(',')
        edges += [[gene_id1,gene_id2]]

    net = nx.Graph()
    net.add_edges_from(edges)
    net.remove_nodes_from(nx.isolates(net))
    net.remove_edges_from(net.selfloop_edges())
    print 'Gene edges: %d' % len(net.edges)
    print 'Gene nodes: %d' % len(net.nodes)
    gene2idx = {node: i for i, node in enumerate(net.nodes())}
    return net, gene2idx, list(net.nodes())

def load_mirna_gene(gnet, gnodes, fname='data/bio-mirna-gene.csv'):
    m2g_net = gnet
    idx = len(gnodes)
    mirna2gene = defaultdict(set)
    fin = open(fname)
    print('Reading: %s' % fname)
    fin.readline()
    edges = []
    mirnas = []
    gene_nodes = gnodes

    num_gene = []
    num_gene_ingenenode = []
    for line in fin:
        mirna, gene = line.strip().split(',')
        num_gene.append(gene)
        if gene in gene_nodes:
            edges += [[mirna, gene]]
            mirna2gene[mirna].add(gene)
            mirnas.append(mirna)
            num_gene_ingenenode.append(gene)

    num_gene = set(num_gene)
    num_gene_ingenenode = set(num_gene_ingenenode)

    mirnas = set(mirnas)
    mirnas_list = list(mirnas)
    m2g_net.add_edges_from(edges)
    mirna2idx = {mirnaid: i+idx for i, mirnaid in enumerate(mirnas)}

    dict = sorted(mirna2idx.items(), key=lambda d: d[1])
    return mirna2gene, mirna2idx, mirnas_list, m2g_net

def load_diease_gene(gnet, gnodes, fname='data/bio-diease-gene.csv'):
    d2g_net = gnet
    idx = len(gnodes)
    diease2gene = defaultdict(set)
    fin = open(fname)
    print('Reading: %s' % fname)
    fin.readline()
    edges = []
    dieases = []
    gene_nodes = gnodes

    num_gene = []
    num_gene_ingenenode = []
    for line in fin:
        diease, gene = line.strip().split(',')
        num_gene.append(gene)
        if gene in gene_nodes:
            edges += [[diease, gene]]
            diease2gene[diease].add(gene)
            dieases.append(diease)
            num_gene_ingenenode.append(gene)

    num_gene = set(num_gene)
    num_gene_ingenenode = set(num_gene_ingenenode)

    dieases = set(dieases)
    dieases_list = list(dieases)
    d2g_net.add_edges_from(edges)
    diease2idx = {dieaseid: i+idx for i, dieaseid in enumerate(dieases)}
    return diease2gene, diease2idx, dieases_list, d2g_net

def load_diease_mirna(gnodes, mlist, dlist, fname='data/bio-diease-mirna.csv'):
    diease2mirna = defaultdict(set)
    fin = open(fname)
    print('Reading: %s' % fname)
    fin.readline()
    edges = []
    rna_list = []
    gene_nodes = gnodes

    for line in fin:
        diease_id, rna = line.strip().split(',')
        if diease_id in dlist and rna in mlist:
            edges += [[diease_id, rna]]
            diease2mirna[diease_id].add(rna)
            rna_list.append(rna)

    # diease2mirna_l = list(diease2mirna)
    # out = open('data/d2m-total.csv', 'w')
    # csv_w = csv.writer(out, dialect='excel')
    # for t in range(len(diease2mirna)):
    #     tt = list(diease2mirna[diease2mirna_l[t]])
    #     for h in range(len(tt)):
    #         csv_w.writerow( [diease2mirna_l[t],tt[h]] )

    rna_list = set(rna_list)
    return diease2mirna

def return_acc(preds, labels):
    idx_label = 0
    top1 = 0
    top2 = 0
    num_recall = []
    num_precision = []
    num_accuracy = []
    for i in range(len(preds)):
        idx = 0
        g = np.where(labels[i] == 1)

        print("i = ", '%04d' % i)
        for r in range(preds.shape[1]):
            print(str(preds[i][r]) + ',')
            # print('i = ' + str(i) + ',')    print(str(a[0])+',')

        # print(g)
        # print(preds[i][g])
        # print(sorted(preds[i][g],reverse=False))
        ##############################
        preds_label = np.zeros_like(preds[i])
        if len(g[0])>0 :
            k_max_idx = np.argsort(preds[i])[::-1][:len(g[0])]
            preds_label[k_max_idx]=1
            if (preds_label==labels[i]).all():
                idx_label = idx_label + 1
            else:
                sort_pred = sorted(preds[i][g], reverse=False)
                mean_pred = np.mean(preds[i][g])
                min = sort_pred[0]
                for j in range(preds.shape[1]):
                    if j not in g[0]:
                        if min < preds[i][j]:
                            idx = idx + 1
                if idx == 0:
                    idx_label = idx_label + 1

            top1_idx = np.where(preds[i] == np.max(preds[i]))
            if top1_idx[0] in g[0]:
                top1 = top1+1

        else:
            idx_label = idx_label
        ##############################

        # sort_pred = sorted(preds[i][g], reverse=False)
        # mean_pred = np.mean(preds[i][g])
        # if len(sort_pred) > 0:
        #     min = sort_pred[0]
        #     # min = mean_pred
        #     for j in range(preds.shape[1]):
        #         if j not in g[0]:
        #             test = (preds[i][j])
        #             if min < preds[i][j]:
        #                 idx = idx + 1
        # if idx == 0:
        #     idx_label = idx_label + 1

        r_insec = np.sum(np.multiply(preds_label, labels[i]))
        r_union = (preds_label+labels[i])
        r_union = np.where(r_union>=1, 1, 0)
        r_union_total = np.sum(r_union)
        r_preds_label = np.sum(preds_label)
        r_label = np.sum(labels[i])

        num_recall.append(r_insec/r_label)
        num_precision.append(r_insec/r_preds_label)
        num_accuracy.append(r_insec/r_union_total)


    r_acc = idx_label / preds.shape[0]
    top1_acc = top1 / preds.shape[0]
    recall = np.sum(num_recall) / len(preds)
    precision = np.sum(num_precision) / len(preds)
    jaccard = np.sum(num_accuracy) / len(preds)
    return r_acc, recall, jaccard, top1_acc

def get_acc_test(preds, labels):
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))
    preds = sigmoid(preds)

    idx_label = 0
    top1 = 0
    top2 = 0
    num_recall = []
    num_precision = []
    num_accuracy = []
    # for i in range(len(preds)):
    #     print(preds[i])
    r_labels = np.where(np.greater(preds,0.5), np.ones_like(labels), np.zeros_like(labels))
    correct_prediction = np.equal(r_labels, labels)
    accuracy_all = np.array(correct_prediction).astype(int)

    g_labels = labels.reshape((-1,1))
    g_preds = preds.reshape((-1,1))
    roc_sc = metrics.roc_auc_score(g_labels, g_preds)
    aupr_sc = metrics.average_precision_score(g_labels, g_preds)
    precision_20, recall_20, pr_thresholds_20 = precision_recall_curve(g_labels[0:40], g_preds[0:40])
    precision_40, recall_40, pr_thresholds_40 = precision_recall_curve(g_labels[0:80], g_preds[0:80])
    precision_60, recall_60, pr_thresholds_60 = precision_recall_curve(g_labels[0:120], g_preds[0:120])

    fpr_hgcn, tpr_hgcn, thresholds_hgcn = roc_curve(g_labels, g_preds, drop_intermediate=False)
    pic_curve_only(fpr_hgcn, tpr_hgcn)

    precision_20 = list(precision_20)
    out = open('data/lung/top20/bp20.csv', 'w')
    csv_w = csv.writer(out, dialect='excel')
    for t in range(len(precision_20)):
        csv_w.writerow( [precision_20[t]] )

    recall_20 = list(recall_20)
    out = open('data/lung/top20/br20.csv', 'w')
    csv_w = csv.writer(out, dialect='excel')
    for t in range(len(recall_20)):
        csv_w.writerow([recall_20[t]])
    aps_20 = metrics.average_precision_score(g_labels[0:40], g_preds[0:40])

    plt.figure()
    plt.step(recall_20, precision_20, color='g', alpha=0.2, where='post')
    plt.fill_between(recall_20, precision_20, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Top-20 Precision-Recall curve: AP={0:0.4f}'.format(aps_20))
    plt.savefig('data/lung/top20/precision_recall_20.png')

    #
    precision_40 = list(precision_40)
    out = open('data/lung/top40/bp40.csv', 'w')
    csv_w = csv.writer(out, dialect='excel')
    for t in range(len(precision_40)):
        csv_w.writerow([precision_40[t]])

    recall_40 = list(recall_40)
    out = open('data/lung/top40/br40.csv', 'w')
    csv_w = csv.writer(out, dialect='excel')
    for t in range(len(recall_40)):
        csv_w.writerow([recall_40[t]])
    aps_40 = metrics.average_precision_score(g_labels[0:80], g_preds[0:80])

    plt.figure()
    plt.step(recall_40, precision_40, color='b', alpha=0.2, where='post')
    plt.fill_between(recall_40, precision_40, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Top-40 Precision-Recall curve: AP={0:0.4f}'.format(aps_40))
    plt.savefig('data/lung/top40/precision_recall_40.png')

    #
    precision_60 = list(precision_60)
    out = open('data/lung/top60/bp60.csv', 'w')
    csv_w = csv.writer(out, dialect='excel')
    for t in range(len(precision_60)):
        csv_w.writerow([precision_60[t]])

    recall_60 = list(recall_60)
    out = open('data/lung/top60/br60.csv', 'w')
    csv_w = csv.writer(out, dialect='excel')
    for t in range(len(recall_60)):
        csv_w.writerow([recall_60[t]])
    aps_60 = metrics.average_precision_score(g_labels[0:120], g_preds[0:120])

    plt.figure()
    plt.step(recall_60, precision_60, color='r', alpha=0.2, where='post')
    plt.fill_between(recall_60, precision_60, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Top-60 Precision-Recall curve: AP={0:0.4f}'.format(aps_60))
    plt.savefig('data/lung/top60/precision_recall_60.png')

    #

    return np.mean(accuracy_all), roc_sc, aupr_sc

def get_acc_train(preds, labels):
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))
    preds = sigmoid(preds)

    idx_label = 0
    top1 = 0
    top2 = 0
    num_recall = []
    num_precision = []
    num_accuracy = []
    # for i in range(len(preds)):
    #     print(preds[i])
    r_labels = np.where(np.greater(preds,0.5), np.ones_like(labels), np.zeros_like(labels))
    correct_prediction = np.equal(r_labels, labels)
    accuracy_all = np.array(correct_prediction).astype(int)

    g_labels = labels.reshape((-1,1))
    g_preds = preds.reshape((-1,1))
    roc_sc = metrics.roc_auc_score(g_labels, g_preds)
    aupr_sc = metrics.average_precision_score(g_labels, g_preds)
    return np.mean(accuracy_all), roc_sc, aupr_sc


import xlrd
from sklearn.metrics import roc_curve, auc
def load_imcmda(fname_true='data/imcmda/interaction.xlsx', fname_score='data/imcmda/score.xlsx'):
    excelfile = xlrd.open_workbook(fname_true)
    sheet = excelfile.sheet_by_index(0)
    value_true_list = []
    for i in range(sheet.nrows):
            value_true_list.append(sheet.row_values(i))
    value_true = np.array(value_true_list).reshape(-1,1)

    excelfile_score = xlrd.open_workbook(fname_score)
    sheet_score = excelfile_score.sheet_by_index(0)
    score_list = []
    for i in range(sheet_score.nrows):
        score_list.append(sheet_score.row_values(i))
    score = np.array(score_list).reshape(-1, 1)

    fpr_imc, tpr_imc, thresholds_imc = roc_curve(value_true, score, drop_intermediate=False)
    return fpr_imc, tpr_imc, thresholds_imc

def load_glnmda(fname_fpr='data/glnmda/FPR.xlsx', fname_tpr='data/glnmda/TPR.xlsx'):
    fprfile = xlrd.open_workbook(fname_fpr)
    sheet = fprfile.sheet_by_index(0)
    fpr_list = []
    for i in range(sheet.nrows):
        fpr_list.append(sheet.row_values(i))
    fpr = np.array(fpr_list).reshape(-1,1)

    tprfile = xlrd.open_workbook(fname_tpr)
    sheet_tpr = tprfile.sheet_by_index(0)
    tpr_list = []
    for i in range(sheet_tpr.nrows):
        tpr_list.append(sheet_tpr.row_values(i))
    tpr = np.array(tpr_list).reshape(-1,1)

    return fpr,tpr

def load_spm(fname_fpr='data/spm/FPR.xlsx', fname_tpr='data/spm/TPR.xlsx'):
    fprfile = xlrd.open_workbook(fname_fpr)
    sheet = fprfile.sheet_by_index(0)
    fpr_list = []
    for i in range(sheet.nrows):
        fpr_list.append(sheet.row_values(i))
    fpr = np.array(fpr_list).reshape(-1, 1)

    tprfile = xlrd.open_workbook(fname_tpr)
    sheet_tpr = tprfile.sheet_by_index(0)
    tpr_list = []
    for i in range(sheet_tpr.nrows):
        tpr_list.append(sheet_tpr.row_values(i))
    tpr = np.array(tpr_list).reshape(-1, 1)

    return fpr, tpr

import matplotlib.pyplot as plt
def pic_curve(fpr_imcmda,tpr_imcmda, fpr_glnmda,tpr_glnmda, fpr_spm,tpr_spm, fpr_hgcn,tpr_hgcn):
    roc_auc_imc = auc(fpr_imcmda, tpr_imcmda)
    roc_auc_gln = auc(fpr_glnmda,tpr_glnmda)
    roc_auc_spm = auc(fpr_spm,tpr_spm)
    roc_auc_hgcn = auc(fpr_hgcn, tpr_hgcn)

    plt.figure()
    lw = 2  # 2
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

    plt.plot(fpr_imcmda, tpr_imcmda, color='blue', lw=lw, linestyle='-', label='IMCMDA (area = %0.4f)' % roc_auc_imc)
    plt.plot(fpr_spm, tpr_spm, color='black', lw=lw, linestyle='-', label='SPM (area = %0.4f)' % roc_auc_spm)
    plt.plot(fpr_glnmda, tpr_glnmda, color='pink', lw=lw, marker=',', label='GLNMDA (area = %0.4f)' % roc_auc_gln)
    plt.plot(fpr_hgcn, tpr_hgcn, color='red', lw=lw, marker=',', label='HGCNMDA (area = %0.4f)' % roc_auc_hgcn)
    plt.legend(loc="lower right")
    plt.savefig('data/pic/roc_total.png')

def pic_curve_only(fpr, tpr):
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

    plt.plot(fpr, tpr, color='red', lw=lw, marker=',', label='AUC area = %0.4f)' % roc_auc)
    plt.legend(loc="lower right")
    plt.savefig('data/only/roc.png')
    #plt.savefig('data/lung/roc.png')


def draw_pre_recall():
    fname_lung_pre_20 = open('data/breast/top20/bp20.csv')
    fname_lung_rec_20 = open('data/breast/top20/br20.csv')

    fname_lung_pre_40 = open('data/breast/top40/bp40.csv')
    fname_lung_rec_40 = open('data/breast/top40/br40.csv')

    fname_lung_pre_60 = open('data/breast/top60/bp60.csv')
    fname_lung_rec_60 = open('data/breast/top60/br60.csv')

    fname_lung_pre_20_list = []
    fname_lung_rec_20_list = []
    for line in fname_lung_pre_20:
        line = line.strip()
        fname_lung_pre_20_list.append(float(line))
    for line in fname_lung_rec_20:
        line = line.strip()
        fname_lung_rec_20_list.append(float(line))

    fname_lung_pre_40_list = []
    fname_lung_rec_40_list = []
    for line in fname_lung_pre_40:
        line = line.strip()
        fname_lung_pre_40_list.append(float(line))
    for line in fname_lung_rec_40:
        line = line.strip()
        fname_lung_rec_40_list.append(float(line))

    fname_lung_pre_60_list = []
    fname_lung_rec_60_list = []
    for line in fname_lung_pre_60:
        line = line.strip()
        fname_lung_pre_60_list.append(float(line))
    for line in fname_lung_rec_60:
        line = line.strip()
        fname_lung_rec_60_list.append(float(line))

    plt.figure()
    # plt.step(fname_lung_rec_20_list, fname_lung_pre_20_list, color='g', alpha=0.2, where='post')
    # plt.fill_between(fname_lung_rec_20_list, fname_lung_pre_20_list, step='post', alpha=0.2, color='w')
    # plt.step(fname_lung_rec_40_list, fname_lung_pre_40_list, color='r', alpha=0.2, where='post')
    # plt.fill_between(fname_lung_rec_40_list, fname_lung_pre_40_list, step='post', alpha=0.2, color='w')
    # plt.step(fname_lung_rec_60_list, fname_lung_pre_60_list, color='b', alpha=0.2, where='post')
    # plt.fill_between(fname_lung_rec_60_list, fname_lung_pre_60_list, step='post', alpha=0.2, color='w')

    plt.plot(fname_lung_rec_20_list, fname_lung_pre_20_list, lw=2, color='blue', linestyle='-', label='top-20')
    plt.plot(fname_lung_rec_40_list, fname_lung_pre_40_list, lw=2, color='black', linestyle='-', label='top-40')
    plt.plot(fname_lung_rec_60_list, fname_lung_pre_60_list, lw=2, color='pink', marker=',', label='top-60')
    #plt.plot(fpr_hgcn, tpr_hgcn, color='red', marker=',', label='HGCNMDA (area = %0.4f)')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.4, 1.0])
    plt.title('Breast neoplasm Precision-Recall curve among top-20, top-40, top-60')
    plt.legend(loc="lower right")
    plt.savefig('data/breast/precision_recall.png')

def save_train_result(preds, labels):
    preds = list(preds)
    labels = list(labels)
    out_pred = open('data/train_result/train_preds.csv', 'w')
    out_label = open('data/train_result/train_labels.csv', 'w')
    csv_pred = csv.writer(out_pred, dialect='excel')
    csv_label = csv.writer(out_label, dialect='excel')
    for t in range(len(preds)):
        csv_pred.writerow([preds[t]])
    for t in range(len(labels)):
        csv_label.writerow([labels[t]])

def save_test_result(preds, labels):
    preds = list(preds)
    labels = list(labels)
    out_pred = open('data/train_result/test_preds.csv', 'w')
    out_label = open('data/train_result/test_labels.csv', 'w')
    csv_pred = csv.writer(out_pred, dialect='excel')
    csv_label = csv.writer(out_label, dialect='excel')
    for t in range(len(preds)):
        csv_pred.writerow([preds[t]])
    for t in range(len(labels)):
        csv_label.writerow([labels[t]])

def draw_voilin():
    train_preds = open('data/train_result/train_preds.csv')
    train_labels = open('data/train_result/train_labels.csv')
    test_preds = open('data/train_result/test_preds.csv')
    test_labels = open('data/train_result/test_labels.csv')
    train_preds_list = []
    train_labels_list = []
    test_preds_list = []
    test_labels_list = []
    for line in train_preds:
        line = line.strip()
        train_preds_list.append(float(line[1:-1]))
    for line in train_labels:
        line = line.strip()
        train_labels_list.append(float(line[1:-1]))
    for line in test_preds:
        line = line.strip()
        test_preds_list.append(float(line[1:-1]))
    for line in test_labels:
        line = line.strip()
        test_labels_list.append(float(line[1:-1]))

    train_preds_array = np.array(train_preds_list)
    train_labels_array = np.array(train_labels_list)
    test_preds_array = np.array(test_preds_list)
    test_labels_array = np.array(test_labels_list)
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    train_preds_array = sigmoid(train_preds_array)
    test_preds_array = sigmoid(test_preds_array)

    # # # out_r = open('data/train_result/violin.csv', 'w')
    # # # csv_r = csv.writer(out_r, dialect='excel')
    # with open("data/train_result/violin.csv", "w") as f:
    #     csv_r = csv.writer(f)
    #     for t in range(len(train_preds_array)):
    #         if float(train_preds_array[t])>=0.5:
    #             csv_r.writerow([train_preds_array[t],1])
    #         else:
    #             csv_r.writerow([train_preds_array[t],0])
    #
    #     for t in range(len(test_preds_array)):
    #         if float(test_preds_array[t])>=0.5:
    #             csv_r.writerow([test_preds_array[t],1])
    #         else:
    #             csv_r.writerow([test_preds_array[t],0])

    # r_labels = np.where(np.greater(train_preds_array, 0.5), np.ones_like(train_labels_array), np.zeros_like(train_labels_array))
    # correct_prediction = np.equal(r_labels, train_labels_array)
    # accuracy_all = np.array(correct_prediction).astype(int)
    # a = np.mean(accuracy_all)

    tips = pd.read_csv("data/train_result/violin_posi.csv")
    #sns.violinplot(x="Kinds", y="Negative Scores", data=tips, inner=None)
    sns.violinplot(x="Sampling Category", y="Positive Sample Scores", data=tips, inner='box')
    sns.swarmplot(x="Sampling Category", y="Positive Sample Scores", data=tips)

    plt.savefig('violin_posi.png')

draw_voilin()

