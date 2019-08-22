# HGCNMDA</br>
Requirements   
Linux (We only tested on Ubuntu-16.04)   
networkx (version == 2.2)   
Python (version == 2.7)   
protobuf (version ==3.7.0)   
scikit-learn (version == 0.20.3)   
scipy (version == 1.2.1)   

data : </br>
&nbsp;&nbsp;&nbsp;---breast : Breast neoplasm to show the performance of top-20, top-40 and top-60, separately. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;---top20 : precision and recall data for top20. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;---top40 : precision and recall data for top40. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;---top60 : precision and recall data for top60. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;precision_recall.png </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;roc.png</br>
&nbsp;&nbsp;&nbsp;---lung : Lung neoplasm to show the performance of top-20, top-40 and top-60, separately. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;---top20 : precision and recall data for top20. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;---top40 : precision and recall data for top40. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;---top60 : precision and recall data for top60. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;precision_recall.png </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;roc.png </br>
&nbsp;&nbsp;&nbsp;lymp : Lymphoma to show the performance of top-20, top-40 and top-60, separately. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;---top20 : precision and recall data for top20. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;---top40 : precision and recall data for top40. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;---top60 : precision and recall data for top60. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;precision_recall.png </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;roc.png </br>
&nbsp;&nbsp;&nbsp;glnmda : Existing Algorithms </br>
&nbsp;&nbsp;&nbsp;imcmda : Existing Algorithms </br>
&nbsp;&nbsp;&nbsp;spm : Existing Algorithms </br>
&nbsp;&nbsp;&nbsp;only : ROC curve of single disease. </br>
&nbsp;&nbsp;&nbsp;pic : various curves. </br>
&nbsp;&nbsp;&nbsp;train_result : Including train result, test result and violin data. </br>
&nbsp;&nbsp;&nbsp;bio-diease-gene.csv : Diease and gene network database. </br>
&nbsp;&nbsp;&nbsp;bio-diease-mirna.csv : Diease and miRNA network database. </br>
&nbsp;&nbsp;&nbsp;bio-mirna-gene.csv : miRNA and gene network database. </br>
&nbsp;&nbsp;&nbsp;bio-ppi.csv : PPI network database from decagon, it is available through the website: http://snap.stanford.edu/decagon. </br>
&nbsp;&nbsp;&nbsp;d2m.csv : Number of miRNAs for each disease. </br>
&nbsp;&nbsp;&nbsp;d2m-total.csv : Name of miRNAs for each disease.</br>

software : node2vec program. </br>
test.py : Some test codes in the process of program development, which is negligible. </br>
inits.py : Including the initialization of some TF variables.</br>
metrics.py : Including some measure functions for TF, such as get_consine_simi, masked_softmax_cross_entropy, get_knn, masked_accuracy.</br> 
se_sp_mcc.py : Draw roc curve according to test_preds and test_labels. </br>
layers.py : Hierarchical and Layered in Heterogeneous Networks. </br>
models.py : The model established in this paper. </br>
utils.py : Loading data, reading data, processing data, generating result files and various curves. </br>
train.py : The main program file is executed by following commands: python train.py . This program takes about two hours to run.</br>
