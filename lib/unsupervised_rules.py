#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 18:41:43 2018

@author: Alberto Barbado González
"""

# Libraries
import numpy as np
import pandas as pd
from sklearn import svm
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler


def obtain_centroid(X_train, sc, n_clusters, categorical_cols, numerical_cols):
    """
    Function to obtain the centroid of a group of data points. It uses K-Prototypes algorithm so
    it can work with both numerical and categorical (non ordinal) data.
    
    It returns the centroid/prototype point for that data.
    
    """
    kproto = KPrototypes(n_clusters=n_clusters, init='Cao', verbose=2)
    kproto.fit_predict(X_train, categorical=list(range(np.shape(X_train)[1]-len(categorical_cols), np.shape(X_train)[1])))
    centroid = kproto.cluster_centroids_[0]
    centroid = sc.inverse_transform(centroid)
    if len(categorical_cols) > 0 and len(numerical_cols) > 0:
        for i in range(len(categorical_cols)):
            centroid = np.insert(centroid, np.shape(centroid)[1], kproto.cluster_centroids_[1][0][i], axis=1)
            
    return pd.DataFrame(centroid)


def ocsvm_rule_extractor(dataset_mat, numerical_cols, categorical_cols, dct_params):
    
    """
    Function to extract rules that justify in a comprehensive way why some data points
    are identified as outliers. The function returns a dataframe with the boundaries that
    define those rules according to the different features used as well as the model trained.
    
    """
    
    # Check data quantity
    n_vertex = 2**(len(numerical_cols) + len(categorical_cols))
    
    # Check that there's enough data points to obtain the vertex of the hypercube
    if n_vertex > len(dataset_mat):
        raise ValueError("ERROR! Insufficient data points") 
    
    # Scaling numerical data
    sc = StandardScaler()
    X_train = dataset_mat[numerical_cols]
    X_train = sc.fit_transform(X_train)
    
    for col in categorical_cols:
        X_train = np.insert(X_train, np.shape(X_train)[1], dataset_mat[col].values, axis=1)
    
    # Train OneClassSVM
    model = svm.OneClassSVM(**dct_params)
    model.fit(X_train)
    preds = pd.DataFrame({"predictions":list(model.predict(X_train))}) 
    preds["distances"] = model.decision_function(X_train)
    df_anomalies = pd.merge(dataset_mat, preds, left_index=True, right_index=True)
    
    df_anomalies_yes = df_anomalies[df_anomalies['predictions']==-1].sort_values(by="distances", ascending=True)
    X_train_yes = X_train[df_anomalies_yes.index]
    df_yes = df_anomalies_yes.copy().drop(columns={'predictions', 'distances'})
    
    # Obtain hypercube for outliers
    n = 1
    centroid_yes = obtain_centroid(X_train_yes, sc, n, categorical_cols, numerical_cols)
    df_yes_aux = df_yes - centroid_yes.iloc[0].values.squeeze()
    df_yes_aux = df_yes_aux.apply(lambda x: x.sort_values().values)
    list_index = list(df_yes_aux.head(int(n_vertex/2)).append(df_yes_aux.tail(int(n_vertex/2))).index)
    vectors_bound = df_anomalies_yes[df_anomalies_yes.index.isin(list_index)].drop(columns={'predictions', 'distances'})  
    df_bounds_max = vectors_bound.groupby(categorical_cols)[numerical_cols].max().reset_index() # df with the max variable values on the hyperplanes
    df_bounds_min = vectors_bound.groupby(categorical_cols)[numerical_cols].min().reset_index() # df with the min variable values on the hyperplanes
    df_bounds = pd.merge(df_bounds_max, df_bounds_min, left_on=categorical_cols, right_on=categorical_cols, suffixes=('_max', '_min'))
    
    # Extract rules
    number = 0
    list_rules = []
    for i, row in df_bounds.iterrows():
        number += 1
        s = 'Rule Nº {number}: IF '.format(number=number)
        for col, value in zip(row.index, row.values):
            if col in categorical_cols:
                s += '{col} = {value} AND '.format(col=col, value=value)
            elif col[-4:]=='_max':
                s += '{col} <= {value} AND '.format(col=col, value=value)
            else:
                s += '{col} >= {value} AND '.format(col=col, value=value)
        
        s = s[:-4] # delete last AND
        s = s.replace('_max', '').replace('_min', '')
        print(s)   
        list_rules.append(s)
        
    
    return model, df_bounds