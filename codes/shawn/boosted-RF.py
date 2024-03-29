 # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from collections import Counter 
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import matplotlib.pyplot as plt
from Bio.Seq import Seq
from itertools import combinations_with_replacement
from itertools import permutations
from sklearn.feature_selection import SelectFromModel

def get_kmers(kmer_length): 

	all_kmers = []
	
	kmers = list(combinations_with_replacement("ATCG", kmer_length))
	for index in range(len(kmers)):
		kmers[index] = "".join(kmers[index])
		
	for kmer in kmers:
		permut = list(permutations(kmer))
		permut = list(set(["".join(x) for x in permut]))
		all_kmers = all_kmers + permut

	return all_kmers

#function to calculate the GC content of a list of sequences 
def calc_gc(sequences):
    
    gc_list = []
    
    for sequence in sequences:
    
        num_gc = 0
        for base in sequence:
            if base == "G" or base == "C":
                num_gc += 1

        gc_content = num_gc/len(sequence)
        gc_list.append(gc_content)

    return gc_list 

#function to get the list of kmer features for each sequences 
def get_features(data, kmer_list):

	features = []

	seq_len = len(data[0])
	kmer_length = len(kmer_list[0])

	for entry in data:

		feature_count = [0] * len(kmer_list)
		start = 0
		end = kmer_length

		while end <= seq_len + 1:
			seq_slice = entry[start:end]
			start += 1
			end += 1

			if len(seq_slice) == kmer_length:
				#append to list of counts
				index = kmer_list.index(seq_slice)
				feature_count[index] += 1 

		#get the gc content and add this to the list of features for this sequence  
		#gc_content = calc_gc(entry)
        
        #feature_count.append(gc_content)
        #append to list of all features 
		features.append(feature_count)
		
	return features

#function to get the rev. complement of a list of sequences
def rev_comp(seqs):
    
    rev_seqs = []
    
    for seq in seqs:
        seq = Seq(seq)
        rev_seqs.append(seq.reverse_complement())
        
    return rev_seqs 

def combine_features(*args):
    
    combined_features = []
    
    num_features = len(args[0])
    
    for index in range(num_features):
        
        new_list = []
        
        for a in args:

            if isinstance(a[index], list):
                new_list = new_list + a[index]
            else:
                new_list = new_list + list(a[index])
        
        combined_features.append(new_list)
        
    return combined_features

def extract_FFT_component(data):
    component = []
    for entry in data:
        A_signal = entry[:14]
        C_signal = entry[14:28]
        G_signal = entry[28:42]
        T_signal = entry[42:56]
        A_ps = abs(np.fft.rfft(A_signal))
        C_ps = abs(np.fft.rfft(C_signal))
        G_ps = abs(np.fft.rfft(G_signal))
        T_ps = abs(np.fft.rfft(T_signal))
        component_i = list(A_ps)+list(C_ps)+list(G_ps)+list(T_ps)
        component.append(component_i)
    N_component = len(A_ps)
    features = ['1']*N_component*4
    base = 'ACGT'
    for i in range(N_component*4):
            features[i] = 'fft '+base[i//N_component] +' '+ str(i%N_component) + '-th mode'
    return features, component


#training_data = pd.read_csv('C:/Users/Chung/rule-based-learning/datasets/humanSP1/humanSP1_train.csv', sep= ',', header=None)
#test_data = pd.read_csv('C:/Users/Chung/rule-based-learning/datasets/humanSP1/humanSP1_test.csv', sep= ',', header=None)

training_data = pd.read_csv('C:/Users/Chung/rule-based-learning/datasets/ecoli_s70/sig70_train.csv', sep= ',', header=None)
test_data = pd.read_csv('C:/Users/Chung/rule-based-learning/datasets/ecoli_s70/sig70_test.csv', sep= ',', header=None)

X = training_data.values[:,0]
Y = training_data.values[:, 1:2]
X_real_test = test_data.values[:,0]
X = [i.upper() for i in X]
X_real_test = [i.upper() for i in X_real_test]

X_strand_corrected = []
y_strand_corrected = []
for i in range(len(X)):
    seq = Seq(str(X[i]))
    X_strand_corrected.append(str(seq))
    y_strand_corrected.append(Y[i])
    X_strand_corrected.append(str(seq.reverse_complement()))
    y_strand_corrected.append(Y[i])
X = X_strand_corrected
Y = y_strand_corrected



updated_X = []
for line in X:
    tmp= np.zeros((4, len(X[0])))
    for i in range(len(line)):
        if  line[i] == 'A':
            tmp[0][i] = 1
        elif line[i] == 'C':
            tmp[1][i] = 1
        elif line[i] == 'G':
            tmp[2][i] = 1
        elif line[i] == 'T':
            tmp[3][i] = 1
    tmp = tmp.flatten()
    updated_X.append(tmp)
#Y = Y=='binding site'
gc_content = calc_gc(X)
symbol_2mers = get_kmers(2)
symbol_3mers = get_kmers(3)
symbol_4mers = get_kmers(4)
symbol_5mers = get_kmers(5)
features_2mer = get_features(X, symbol_2mers)
features_3mer = get_features(X, symbol_3mers)
features_4mer = get_features(X, symbol_4mers)

X = combine_features(updated_X, features_2mer)
X = combine_features(X, features_3mer)
#X = combine_features(X, features_4mer)
fft_features_name, fft_component = extract_FFT_component(updated_X)
X = combine_features(X, fft_component)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train = y_train.T[0]
y_test = y_test.T[0]

#feature names
ATCG_identity = ['A','C','G','T']
features = ["1"]*4*len(X[0])
for i in range(len(features)):
    base = i//len(X[0])
    features[i] = 'if position ' + str(i%len(X[0])) +' is ' + ATCG_identity[base]
features = features + symbol_2mers+ symbol_3mers + fft_features_name
#dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)


updated_X = []
for line in X_real_test:
    tmp= np.zeros((4, len(X[0])))
    for i in range(len(line)):
        if  line[i] == 'A':
            tmp[0][i] = 1
        elif line[i] == 'C':
            tmp[1][i] = 1
        elif line[i] == 'G':
            tmp[2][i] = 1
        elif line[i] == 'T':
            tmp[3][i] = 1
    tmp = tmp.flatten()
    updated_X.append(tmp)

features_2mer = get_features(X_real_test, symbol_2mers)
features_3mer = get_features(X_real_test, symbol_3mers)
features_4mer = get_features(X_real_test, symbol_4mers)
X_real_test = combine_features(updated_X, features_2mer)
X_real_test = combine_features(X_real_test, features_3mer)
#X_real_test = combine_features(X_real_test, features_4mer)
fft_features_name, fft_component = extract_FFT_component(updated_X)
X_real_test = combine_features(X_real_test, fft_component)

#training xgboost model
tfbs_classifier = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 500)
tfbs_classifier.fit(X_train, y_train)
y_pred = tfbs_classifier.predict(X_test)
scores = f1_score(y_test, y_pred, average='macro')

#save results
df_X_real_test = pd.DataFrame(data=X_real_test, columns=features)
y_real_pred = tfbs_classifier.predict(df_X_real_test)
df = pd.DataFrame(y_real_pred)
#df.to_csv('../result/humanSP1/boosted-RF-3mer+fft.csv', header=0, index=0)
df.to_csv('../result/ecoli_s70/boosted-RF-3mer+fft.csv', header=0, index=0)

##plot importance
tfbs_classifier.get_booster().feature_names = features
xgb.plot_importance(tfbs_classifier)
plt.rcParams['figure.figsize'] = [100, 100]
plt.show()
#plt.savefig('../figure/humanSP1/feature_importance_boostedtree-4mer.png')
plt.savefig('../figure/ecoli_s70/feature_importance_boostedtree-4mer.png')
'''
#hyper-parameter tuning
for i in [100,500,1000]:
    tfbs_classifier = xgb.XGBClassifier(nthread=-1, objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 10, alpha = 10, n_estimators = i)
    scores = np.zeros(1)
    for i in range(1):
        tfbs_classifier.fit(X_train, y_train)
        y_pred = tfbs_classifier.predict(X_test)
        scores = f1_score(y_test, y_pred, average='macro')
    print('accuracy: ', np.mean(scores), '+-', np.std(scores))
#3mer best parameter: max_depth = 10, n_estimators = 100
#4mer best parameter: max_depth = 5, n_estimators = 500
    '''
    
# select features using threshold
thresholds = np.sort(tfbs_classifier.feature_importances_)
thresholds = thresholds[np.linspace(1,len(thresholds),len(thresholds))%20==0]
for thres in thresholds:
    selection = SelectFromModel(tfbs_classifier, threshold=thres, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 500)
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)
    y_pred = selection_model.predict(select_X_test)
    scores = f1_score(y_test, y_pred, average='macro')
    print('accuracy: ',scores, ' threshold: ', thres)
