from __future__ import division
from __future__ import print_function

import time
import random
import tensorflow as tf
import scipy.io as sio
from utils import *
from models import GCN, MLP
import scipy.sparse as sp
import objgraph
import gc
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
# tf.reset_default_graph()
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
gene_net, gene2idx, gnodes= load_ppi(fname='data/bio-ppi.csv')
mirna2gene, mirna2idx, mlist, m2g_net = load_mirna_gene(gnet=gene_net.copy(), gnodes=gnodes, fname='data/bio-mirna-gene.csv')
diease2gene, diease2idx, dlist, d2g_net = load_diease_gene(gnet=gene_net.copy(), gnodes=gnodes, fname='data/bio-diease-gene.csv')
diease2mirna = load_diease_mirna(gnodes=gnodes, mlist=mlist, dlist=dlist, fname='data/bio-diease-mirna.csv')


l=len(gnodes)
diease2mirna_list = list(diease2mirna)
diease2mirna_id_list = []
for i in range(len(diease2mirna_list)):
    id=diease2idx[diease2mirna_list[i]]
    tt=list(diease2mirna[diease2mirna_list[i]])
    for j in range(len(tt)):
        g = mirna2idx[tt[j]]
        diease2mirna_id_list.append([id-l,g-l])
adj_d2m = np.zeros((len(dlist),len(mlist)))
for j in range(len(diease2mirna_id_list)):
    adj_d2m[diease2mirna_id_list[j][0]][diease2mirna_id_list[j][1]]=1

labels_row_total = []
positive_label = []
neg_label = []
for t in range(len(adj_d2m)):
    if np.sum(adj_d2m[t])!=0:
        g = np.where(adj_d2m[t] == 1)
        labels_row_total.append(t)
        for lb in range(len(g[0])):
            positive_label.append((t,g[0][lb]))

for nn in range(len(adj_d2m)):
    for ne in range(adj_d2m.shape[1]):
        if adj_d2m[nn][ne] == 0:
            neg_label.append((nn,ne))

labels_row_total = list(set(labels_row_total))

# labels_row = labels_row_total[0:100]
# labels_row_test = labels_row_total[100:len(labels_row_total)]
random.shuffle(labels_row_total)

labels_row = labels_row_total[0:310]
labels_row_test = labels_row_total[310:len(labels_row_total)]

positive_train_label_row = []
for trainl in range(len(labels_row)):
    for lbe in range(len(positive_label)):
        if labels_row[trainl] == (list(positive_label[lbe]))[0]:
            positive_train_label_row.append(positive_label[lbe])

positive_test_label_row = []
for trainl in range(len(labels_row_test)):
    for lbe in range(len(positive_label)):
        if labels_row_test[trainl] == (list(positive_label[lbe]))[0]:
            positive_test_label_row.append(positive_label[lbe])


neg_train_label_row = []
for trainl in range(len(labels_row)):
    for lbe in range(len(neg_label)):
        if labels_row[trainl] == (list(neg_label[lbe]))[0]:
            neg_train_label_row.append(neg_label[lbe])

neg_test_label_row = []
for trainl in range(len(labels_row_test)):
    for lbe in range(len(neg_label)):
        if labels_row_test[trainl] == (list(neg_label[lbe]))[0]:
            neg_test_label_row.append(neg_label[lbe])


mapping = dict( gene2idx.items() )
gene_net = nx.relabel_nodes(gene_net, mapping)
adj_gene = nx.adjacency_matrix(gene_net)
gene_degrees = np.array(adj_gene.sum(axis=0)).squeeze()

mapping = dict( gene2idx.items() + mirna2idx.items() )
m2g_net = nx.relabel_nodes(m2g_net, mapping)
adj_m2g = nx.adjacency_matrix(m2g_net)
adj_m2g = adj_m2g.todense()
adj_m2g = np.array(adj_m2g)
adj_m2g = adj_m2g[len(gnodes):len(m2g_net.nodes),0:len(gnodes)]    #len(gnodes):len(m2g_net.nodes),0:len(gnodes)
# adj_m2g = sp.csc_matrix(adj_m2g, dtype='float32')
print(len(gnodes))
mapping = dict( gene2idx.items() + diease2idx.items() )
d2g_net = nx.relabel_nodes(d2g_net, mapping)
adj_d2g = nx.adjacency_matrix(d2g_net)
adj_d2g = adj_d2g.todense()
adj_d2g = np.array(adj_d2g)
adj_d2g = adj_d2g[len(gnodes):len(d2g_net.nodes),0:len(gnodes)]    #len(gnodes):len(d2g_net.nodes),0:len(gnodes)   19081:19475,0:19081
# adj_d2g = sp.csc_matrix(adj_d2g, dtype='float32')

features_g = generate_node2vec_embeddings(adj_gene, 128, False)
# features_g = sp.identity(len(gnodes))
features_g = sp.csr_matrix(features_g).tolil()
features_g = preprocess_features(features_g)

A = adj_gene.copy()  # the observed network
# A[test_pos[0], test_pos[1]] = 0  # mask test links
# A[test_pos[1], test_pos[0]] = 0  # mask test links
A = sp.csr_matrix(A)

if FLAGS.model == 'gcn':
    support = [preprocess_adj(A)]
    num_supports = 1  #A.shape[0]
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(A, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(A)]  # Not used
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

# Define model evaluation function
def evaluate(features_g, support, labels_row_test, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features_g, support, adj_m2g, adj_d2g, adj_d2m, len(mlist), len(dlist), labels_row_test, positive_test_label_row, neg_test_label_row, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.predict(), model.loss_pres, model.loss_labs], feed_dict=feed_dict_val)
    r_predict = outs_val[2][labels_row_test,:]
    r_lab = adj_d2m[labels_row_test,:]

    y_true = outs_val[4]
    y_score = outs_val[3]
    save_test_result(y_score,y_true)

    fpr_hgcn, tpr_hgcn, thresholds_hgcn = roc_curve(y_true, y_score, drop_intermediate=False)
    d = {'Threshold': thresholds_hgcn, 'True Positive Rate': tpr_hgcn, 'False Positive Rate': fpr_hgcn}
    metric_table = pd.DataFrame(d, columns=['Threshold', 'True Positive Rate', 'False Positive Rate'])

    fpr_imcmda, tpr_imcmda, thresholds_imcmda = load_imcmda(fname_true='data/imcmda/interaction.xlsx', fname_score='data/imcmda/score.xlsx')
    fpr_glnmda, tpr_glnmda = load_glnmda(fname_fpr='data/glnmda/FPR.xlsx', fname_tpr='data/glnmda/TPR.xlsx')
    fpr_spm, tpr_spm = load_spm(fname_fpr='data/spm/FPR.xlsx', fname_tpr='data/spm/TPR.xlsx')

    pic_curve(fpr_imcmda, tpr_imcmda, fpr_glnmda, tpr_glnmda, fpr_spm, tpr_spm, fpr_hgcn, tpr_hgcn)
    # print(metric_table)

    # acc = get_acc_test(outs_val[3],outs_val[4])
    acc = get_acc_train(outs_val[3], outs_val[4])

    return outs_val[0], outs_val[1], acc, (time.time() - t_test)

# patch=19
# kfold_num=int(len(labels_row_total)/patch)
# kfold_test_auc = []
# for split in range(kfold_num):
#     print("kfold num:",'%04d' % split)
#     random.shuffle(labels_row_total)
#     labels_row = labels_row_total[0:patch*(kfold_num-1)]
#     labels_row_test = labels_row_total[patch*(kfold_num-1):len(labels_row_total)]

tf.reset_default_graph()
# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features_g': tf.sparse_placeholder(tf.float32, shape=tf.constant(features_g[2], dtype=tf.int64)),
    'adj_m2g': tf.placeholder(tf.float32, shape=(None, adj_m2g.shape[1])),
    'adj_d2g': tf.placeholder(tf.float32, shape=(None, adj_d2g.shape[1])),
    'adj_d2m': tf.placeholder(tf.float32, shape=(None, adj_d2m.shape[1])),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'num_features_diease_nonzero': tf.placeholder(tf.int32),  # helper variable for sparse dropout
    'mirna_length': tf.placeholder(tf.int32),
    'diease_length': tf.placeholder(tf.int32),
    'labels_row': tf.placeholder(tf.int32),
    'positive_train_label_row': tf.placeholder(tf.int32, shape=(None,2)),
    'neg_train_label_row': tf.placeholder(tf.int32, shape=(None,2))
}

# Create model
model = model_func(placeholders, input_dim=features_g[2][1], logging=True)

# Initialize session
sess = tf.Session()
feed_dict = construct_feed_dict(features_g, support, adj_m2g, adj_d2g, adj_d2m, len(mlist), len(dlist), labels_row, positive_train_label_row, neg_train_label_row, placeholders)
feed_dict.update({placeholders['dropout']: FLAGS.dropout})
sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)

# Train model
for epoch in range(FLAGS.epochs):
    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features_g, support, adj_m2g, adj_d2g, adj_d2m, len(mlist), len(dlist), labels_row, positive_train_label_row, neg_train_label_row, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.predict(), model.placeholders['adj_d2m'], model.test, model.loss_pres,model.loss_labs, model.output_d2g, model.output_m2g, model.negative_sample], feed_dict=feed_dict)
    preds = outs[6]
    labels = outs[7]
    acc = get_acc_train(preds,labels)

    dai = FLAGS.epochs-1
    if epoch == dai:
        save_train_result(preds,labels)

    # d2g = outs[8] #(394,32)
    # m2g = outs[9] #(569,32)
    #
    # negative_sample = outs[10]
    #
    # train_adj_d2m = adj_d2m[labels_row,:]
    # ttt = np.sum(train_adj_d2m)
    # op1 = 0.0
    # op2 = 0.0
    # op3 = np.zeros((32,))
    # for dis in range(len(positive_train_label_row)):
    #     edge = list(positive_train_label_row[dis])
    #     op3 = op3 +(d2g[edge[0]] - m2g[edge[1]])
    #     op1 = op1+np.linalg.norm((d2g[edge[0]] - m2g[edge[1]]))
    #     op2 = op2+np.sqrt(np.sum(np.square(d2g[edge[0]] - m2g[edge[1]])))
    # average_positive_train_dis = op1/len(positive_train_label_row)
    # average_positive_train_vector = op3/len(positive_train_label_row)
    #
    # neg_dis_total = []
    # neg_dis = []
    # neg_dis_edge = []
    # for neg in range(len(neg_train_label_row)):
    #     n_edge = list(neg_train_label_row[neg])
    #     if (np.linalg.norm(d2g[n_edge[0]]-m2g[n_edge[1]]) > average_positive_train_dis):
    #         neg_dis_total.append( (np.linalg.norm(d2g[n_edge[0]]-m2g[n_edge[1]]), (n_edge[0],n_edge[1])) )
    #         neg_dis.append( np.linalg.norm(d2g[n_edge[0]]-m2g[n_edge[1]]) )
    #         neg_dis_edge.append( (n_edge[0],n_edge[1]) )
    #
    # k_max_dis_idx = np.argsort(neg_dis)[::-1][:len(positive_train_label_row)]
    # k_max_dis = [neg_dis[j] for j in k_max_dis_idx]
    # k_max_edge = [neg_dis_edge[j] for j in k_max_dis_idx]
    #
    # test_array = np.array(neg_dis_edge)   #ndarray qiepian
    # kkv = test_array[k_max_dis_idx]
    # test_k_max_edge = kkv.tolist()
    #
    # test_adj_d2m = adj_d2m[labels_row_test, :]
    # ttt1 = np.sum(test_adj_d2m)
    # train_adj_d2m = sp.csr_matrix(train_adj_d2m)

    # Validation
    # cost, v_acc, auc, duration = evaluate(features_rna, support, test_diease_rna_adj, test_features_diease, placeholders)
    # if epoch % 20 == 0:
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                                                 "train_acc1=", "{:.5f}".format(outs[2]),
                                                 "train_acc2=", "{:.5f}".format(acc[0]),
                                                 "train_auc=", "{:.5f}".format(acc[1]),
                                                 "train_aps=", "{:.5f}".format(acc[2]),
                                                 "time=", "{:.5f}".format(time.time() - t))

    # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
    #     print("Early stopping...")
    #     break

print("Optimization Finished!")

# # Testing
test_cost, test_acc, acc, test_duration = evaluate(features_g, support, labels_row_test, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),"accuracy=", "{:.5f}".format(test_acc),
      "acc2=", "{:.5f}".format(acc[0]), "auc=", "{:.5f}".format(acc[1]),
                                                    "aps=", "{:.5f}".format(acc[2]), "time=", "{:.5f}".format(test_duration))
# kfold_test_auc.append(acc[1])
tf.get_default_graph().finalize()
objgraph.show_growth()
gc.collect()
sess.close()
# result_test_auc = np.sum(kfold_test_auc)/kfold_num
# print( "According ",'%02d' % kfold_num, "-fold Cross Validation, the result test auc= ","{:.5f}".format(result_test_auc) )
    