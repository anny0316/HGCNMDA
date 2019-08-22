# HGCNMDA</br>
Requirements   
Linux (We only tested on Ubuntu-16.04)   
networkx (version == 2.2)   
Python (version == 2.7)   
protobuf (version ==3.7.0)   
scikit-learn (version == 0.20.3)   
scipy (version == 1.2.1)   

data : 
   ---breast : Breast neoplasm to show the performance of top-20, top-40 and top-60, separately. 
      ---top20 : precision and recall data for top20. 
	  ---top40 : precision and recall data for top40. 
	  ---top60 : precision and recall data for top60. 
	  precision_recall.png 
	  roc.png 
   ---lung : Lung neoplasm to show the performance of top-20, top-40 and top-60, separately. 
      ---top20 : precision and recall data for top20. 
	  ---top40 : precision and recall data for top40. 
	  ---top60 : precision and recall data for top60. 
	  precision_recall.png 
	  roc.png 
   ---lymp : Lymphoma to show the performance of top-20, top-40 and top-60, separately. 
      ---top20 : precision and recall data for top20. 
	  ---top40 : precision and recall data for top40. 
	  ---top60 : precision and recall data for top60. 
	  precision_recall.png 
	  roc.png 
   ---glnmda : Existing Algorithms 
   ---imcmda : Existing Algorithms 
   ---spm : Existing Algorithms 
   ---only : ROC curve of single disease. 
   ---pic : various curves. 
   ---train_result : Including train result, test result and violin data. 
   bio-diease-gene.csv : Diease and gene network database. 
   bio-diease-mirna.csv : Diease and miRNA network database. 
   bio-mirna-gene.csv : miRNA and gene network database. 
   bio-ppi.csv : PPI network database from decagon, it is available through the website: http://snap.stanford.edu/decagon. 
   d2m.csv : Number of miRNAs for each disease. 
   d2m-total.csv : Name of miRNAs for each disease.

software : node2vec program. 
test.py : Some test codes in the process of program development, which is negligible. 
inits.py : Including the initialization of some TF variables.
metrics.py : Including some measure functions for TF, such as get_consine_simi, masked_softmax_cross_entropy, get_knn, masked_accuracy. 
se_sp_mcc.py : Draw roc curve according to test_preds and test_labels. 
layers.py : Hierarchical and Layered in Heterogeneous Networks. 
models.py : The model established in this paper. 
utils.py : Loading data, reading data, processing data, generating result files and various curves. 
train.py : The main program file is executed by following commands: python train.py . This program takes about two hours to run.
