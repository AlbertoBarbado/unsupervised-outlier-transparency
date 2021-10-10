# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 10:13:42 2019

@author: alber
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.tree import _tree

def train_one_class_svm(dataset_mat, numerical_cols, categorical_cols, dct_params):
    """
    # TODO
    """
    
    # Scaling numerical data
    sc = StandardScaler()
    
    if len(numerical_cols):
        X_train = dataset_mat[numerical_cols]
        X_train = sc.fit_transform(X_train)
    else:
        X_train = dataset_mat
    
    X_train_model = X_train
    
    for col in categorical_cols:
        X_train_model = np.insert(X_train_model, np.shape(X_train_model)[1], dataset_mat[col].values, axis=1)
    
    # Train OneClassSVM
    model = svm.OneClassSVM(**dct_params)
    model.fit(X_train_model)
    preds = pd.DataFrame({"predictions":list(model.predict(X_train_model))}) 
    preds["distances"] = model.decision_function(X_train_model)
    df_anomalies = pd.merge(dataset_mat, preds, left_index=True, right_index=True)
    
    return df_anomalies, model

def grid_search(dataset_mat, numerical_cols, categorical_cols, dct_joblib):
    """
    # TODO
    """
    features = list(dataset_mat.columns)
    
    # REAL
    def grid(arg):
        """
        # TODO
        """
        params = {}
        params['nu'] = arg['nu']
        params['kernel'] = arg['kernel']
        params['gamma'] = arg['gamma']
        
        # Train model
        df_anomalies_all, clf = train_one_class_svm(dataset_mat, numerical_cols, categorical_cols, params)
        # Distance of OO to decision funcion
        d_pi = np.abs(clf.decision_function(np.matrix([0]*(len(features))))[0]) # dimension = len_features + 1 
        # Standarize distances
        df_anomalies_all['distances'] = df_anomalies_all["distances"]/(1-d_pi)
        scoring = df_anomalies_all[df_anomalies_all['predictions']==1]['distances'].max() - np.abs(df_anomalies_all[df_anomalies_all['predictions']==-1]['distances']).max()
        return {'nu':arg['nu'], 'kernel':arg['kernel'], 'gamma':arg['gamma'], 'scoring':scoring}
        
    arg_instances=[{'nu':nu, 'kernel':'rbf', 'gamma':gamma} for nu in np.arange(0.1,1.0,0.1) for gamma in np.arange(0.1,1.0,0.1)]
    results = Parallel(**dct_joblib)(map(delayed(grid), arg_instances))
    
    # Choose the best result
    dct_best = {'nu':-1, 'kernel':"rbf", 'gamma':-1, 'scoring':-np.inf}
    for dct_results in results:
        if dct_results['scoring'] > dct_best['scoring']:
            dct_best = dct_results
            
    return dct_best

def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print ("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print ("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print ("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print ("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)