# coding=utf-8
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
import xlrd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def get_se_sp_curve():
    test_preds = open('data/train_result/test_preds.csv')
    test_labels = open('data/train_result/test_labels.csv')
    test_preds_list = []
    test_labels_list = []

    for line in test_preds:
        line = line.strip()
        test_preds_list.append(float(line[1:-1]))
    for line in test_labels:
        line = line.strip()
        test_labels_list.append(float(line[1:-1]))

    test_preds_array = np.array(test_preds_list)
    test_labels_array = np.array(test_labels_list)

    fpr_hgcn, tpr_hgcn, thresholds_hgcn = roc_curve(test_labels_array, test_preds_array, drop_intermediate=False)
    roc_auc_hgcn = auc(fpr_hgcn, tpr_hgcn)

    plt.figure()
    lw = 2
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.plot(fpr_hgcn, tpr_hgcn, color='red', lw=lw, marker=',', label='HGCNMDA (area = %0.4f)' % roc_auc_hgcn)
    plt.legend(loc="lower right")
    plt.savefig('data/pic/roc_hgcn.png')

get_se_sp_curve()