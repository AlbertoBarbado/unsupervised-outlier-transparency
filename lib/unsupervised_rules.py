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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def obtain_centroid(X_train, sc, n_clusters):
    """
    Function to obtain the centroid of a group of data points. It uses K-Prototypes algorithm so
    it can work with both numerical and categorical (non ordinal) data.
    
    It returns the centroid/prototype point for that data.
    
    """
    kmeans = KMeans(n_clusters = n_clusters, init= 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    
    kmeans.fit_predict(X_train)
    centroid = kmeans.cluster_centers_[0]
    centroid = sc.inverse_transform(centroid)
    
    return pd.DataFrame(centroid)


def obtain_vertices(df_anomalies_no_sub, X_train, sc, n_vertex, numerical_cols):
    """
    Function to obtain the vertices from the hypercube of data points contained
    in a dataframe
    
    """

    # Obtain centroid to calculate the vertices in case there're enough points
    if len(df_anomalies_no_sub) > n_vertex:
        X_train_no = X_train[df_anomalies_no_sub.index] # Numerical scaled values
    
        # Obtain centroid for that sub-hypercube
        n = 1
        centroid_no = obtain_centroid(X_train_no, sc, n)
        
        # Obtain vertices
        df_no = df_anomalies_no_sub.copy()
        df_no_aux = df_no[numerical_cols] - centroid_no.iloc[0].values.squeeze()
        df_no_aux = df_no_aux.sort_values(by=numerical_cols, ascending=True)
 
        list_index = list(df_no_aux.head(int(n_vertex/2)).append(df_no_aux.tail(int(n_vertex/2))).index)
        vectors_bound = df_anomalies_no_sub[df_anomalies_no_sub.index.isin(list_index)]
    
    # In case there's less data in the sub-hypercube than the number of required vertices, all points are selected
    # no need to obtain the centroid
    else:
        print("Few data points in this iteration {0}; using all of them as rules.".format(len(df_anomalies_no_sub)))
        vectors_bound = df_anomalies_no_sub
        
    return vectors_bound

    


def ocsvm_rule_extractor(dataset_mat, numerical_cols, categorical_cols, dct_params):
    
    """
    Function to extract rules that justify in a comprehensive way why some data points
    are identified as outliers. The function returns a dataframe with the boundaries that
    define those rules according to the different features used as well as the model trained.
    
    """
    
    # Check data quantity
    n_vertex = 2**(len(numerical_cols) + len(categorical_cols))
    n_vertex_numerical = 2**len(numerical_cols)
    
    # Check that there's enough data points to obtain the vertex of the hypercube
    if n_vertex > len(dataset_mat):
        raise ValueError("ERROR! Insufficient data points") 
    
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
    preds = pd.DataFrame({"predictions":list(model.predict(X_train))}) 
    preds["distances"] = model.decision_function(X_train)
    df_anomalies = pd.merge(dataset_mat, preds, left_index=True, right_index=True)
    
    df_anomalies_no = df_anomalies[df_anomalies['predictions']==1].sort_values(by="distances", ascending=True).drop(columns={'predictions', 'distances'})
    
    
    # Case 1: Only numerical variables
    if len(categorical_cols) == 0:
        # Obtain vertices
        vectors_bound_all = obtain_vertices(df_anomalies_no, X_train, sc, n_vertex_numerical, numerical_cols)
        
        # Obtain limits
        df_bounds_max = vectors_bound_all.max().reset_index().rename(columns={'index':'cat'}).transpose() # df with the max variable values on the hyperplanes
        df_bounds_max.columns = df_bounds_max.loc['cat']
        df_bounds_max = df_bounds_max.reindex(df_bounds_max.index.drop('cat'))
        aux = [col + '_max' for col in df_bounds_max.columns]
        df_bounds_max.columns = aux
                
        
        df_bounds_min = vectors_bound_all.min().reset_index().rename(columns={'index':'cat'}).transpose() # df with the min variable values on the hyperplanes
        df_bounds_min.columns = df_bounds_min.loc['cat']
        df_bounds_min = df_bounds_min.reindex(df_bounds_min.index.drop('cat'))
        aux = [col + '_min' for col in df_bounds_min.columns]
        df_bounds_min.columns = aux
        
        df_bounds = df_bounds_max.join(df_bounds_min, how='inner')
        
    
    # Case 2: Only categorical variables
    elif len(numerical_cols) == 0:
        df_bounds = df_anomalies_no[categorical_cols].drop_duplicates().reset_index(drop=True)
        
        
    # Case 3: Numerical + Categorical
    else:
        df_cat = df_anomalies_no[categorical_cols]
        df_cat_unique = df_cat.drop_duplicates()
        vectors_bound_all = pd.DataFrame()
        
        for i, row in df_cat_unique.iterrows():
            
            # Obtain sub-hypercube
            list_index = df_cat[df_cat[row.index]==row.values].dropna().index # index for that sub-hypercube
            df_anomalies_no_sub = df_anomalies_no[(df_anomalies_no.index.isin(list_index))].copy() # sub-hypercube
              
            # Obtain vertices for this iteration
            vectors_bound = obtain_vertices(df_anomalies_no_sub, X_train, sc, n_vertex_numerical, numerical_cols)
                
            # Append results for this iteration
            if vectors_bound_all.empty:
                vectors_bound_all = vectors_bound
            else:
                vectors_bound_all = vectors_bound_all.append(vectors_bound)
            
            
        df_bounds_max = vectors_bound_all.groupby(categorical_cols)[numerical_cols].max().reset_index() # df with the max variable values on the hyperplanes
        df_bounds_min = vectors_bound_all.groupby(categorical_cols)[numerical_cols].min().reset_index() # df with the min variable values on the hyperplanes
        df_bounds = pd.merge(df_bounds_max, df_bounds_min, left_on=categorical_cols, right_on=categorical_cols, suffixes=('_max', '_min'))
        
        
        
    # Extract rules
    print("NOT anomaly...")
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
