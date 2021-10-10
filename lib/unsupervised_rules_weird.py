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
    
    It returns the centroid/prototype point for that data as well as the assigned clusters for each datapoint.
    
    """
    kmeans = KMeans(n_clusters = n_clusters, init= 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    
    labels = kmeans.fit_predict(X_train)
    centroid = kmeans.cluster_centers_
    centroid = sc.inverse_transform(centroid)
    
    return pd.DataFrame({'labels':labels}), pd.DataFrame(centroid)


def obtain_vertices(df_anomalies_no_sub, X_train, sc, n_vertex, numerical_cols, n_clusters):
    """
    Function to obtain the vertices from the hypercube of data points contained
    in a dataframe
    
    """
    # Drop duplicate rows
    df_anomalies_no_sub = df_anomalies_no_sub.copy().drop_duplicates()
    rule_subgroups = {}

    # Obtain centroid to calculate the vertices in case there're enough points
    df_train = pd.DataFrame()
    for col,j in zip(numerical_cols, range(X_train.shape[1])):
        df_train[col] = X_train[:,j]
    
    df_train_no = df_train.copy().loc[list(df_anomalies_no_sub.index)].reset_index(drop=True)
    df_anomalies_no_sub = df_anomalies_no_sub.reset_index(drop=True)

    # Obtain centroid for that sub-hypercube
    n = n_clusters
    labels, centroid_no = obtain_centroid(df_train_no, sc, n)
    df_anomalies_no_sub['cluster_label'] = labels
    
    for i in range(n):
        df_no = df_anomalies_no_sub[df_anomalies_no_sub['cluster_label']==i].copy().drop(['cluster_label'],axis=1).reset_index(drop=True)
         # Need at least datapoints for vertices for a number of hypercubes equal to the number of clusters
        if len(df_no) > n_vertex:
            
            # Obtain vertices
            # Euclidean distance of each datapoint to centroid
            df_no['distances'] = np.linalg.norm(df_no[numerical_cols].sub(np.array(centroid_no.loc[i].values.squeeze())), axis=1)
            df_no = df_no.sort_values(by=['distances'], ascending=True)
     
            list_index = list(df_no.head(int(n_vertex/2)).append(df_no.tail(int(n_vertex/2))).index)
            vectors_bound = df_no[df_no.index.isin(list_index)]
            
            centroid = centroid_no.loc[i].values.squeeze()

        # In case there's less data in the sub-hypercube than the number of required vertices, all points are selected
        # no need to obtain the centroid
        else:
            print("Datapoints = {0} | Few data points in this iteration; using all of them as rules.".format(len(df_no)))
            vectors_bound = df_no.copy()
            centroid = None
            vectors_bound['distances'] = None
        
        rule_subgroups[i] = [vectors_bound, centroid]
            
    return rule_subgroups


def obtain_limits(df):
    """
    # TODO
    """
    
    # Obtain limits
    vectors_bound_all = df.copy()
    vectors_bound_all.drop(["distances"], axis=1, inplace=True)
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
    
    return df_bounds


def function_check(x,limits,numerical_cols):
    """
    True: outside hypercube
    False: not outside hypercube
    """
    
    result = False
    for col in numerical_cols:
        l_max = limits[col+'_max'][0]
        l_min = limits[col+'_min'][0]
        
        # If its outside from some of the limits, then its outside the hypercube
        if (x[col] > l_max) or (x[col] < l_min):
            result = True
            
    return result

def obtain_rules(df_anomalies_no, df_anomalies_yes, X_train, sc, n_vertex_numerical, numerical_cols, numerical_cols_dont_scale):
    """
    # TODO
    
    """
    
    # Obtain vertices
    n = 0
    check = True
    
    # Drop duplicates
    df_anomalies_no.drop_duplicates(inplace=True)
    df_anomalies_yes.drop_duplicates(inplace=True)
    
    while check:
        n += 1
        # Rules
        df_bounds = []
        
        print("Iteration {0} | nº clusters used {0}".format(n))
        dict_vectors_bound_all = obtain_vertices(df_anomalies_no, X_train, sc, n_vertex_numerical, numerical_cols, n_clusters=n)
        
        for key, value in dict_vectors_bound_all.items():
            vectors_bound_all = value[0].copy()
            #centroid = value[1]
            
            ### Check if a datapoint anomalous would be inside the not anomalous hypercube
            
            # Case where all the points are used as vertices
            if vectors_bound_all['distances'].iloc[0] == None:
                check = False
                df_bounds.append(obtain_limits(vectors_bound_all))
            
            # Case where only some points are used as vertices
            else:
                limits = obtain_limits(vectors_bound_all)
                df_bounds.append(limits)
                
                df_anomalies_yes["check"] = df_anomalies_yes.apply(lambda x:function_check(x,limits,numerical_cols), axis=1)
                list_check = list(df_anomalies_yes["check"].unique())
                
                # If at least one anomalous point is inside the hypercube, repeat again for ALL points with one more cluster
                if False in list_check:
                    check = True
                    break # Return to 'while' loop
                else:
                    check = False
                    
                    
    return df_bounds


def print_rules(df_bounds, categorical_cols):
    """
    # TODO
    
    """
    
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
        
        
    return list_rules
    


def ocsvm_rule_extractor(dataset_mat, numerical_cols_to_scale, categorical_cols, dct_params, numerical_cols_dont_scale):
    
    """
    Function to extract rules that justify in a comprehensive way why some data points
    are identified as outliers. The function returns a dataframe with the boundaries that
    define those rules according to the different features used as well as the model trained.
    
    """
    
    # Check data quantity
    numerical_cols = numerical_cols_to_scale + numerical_cols_dont_scale
    n_vertex = 2**(len(numerical_cols) + len(categorical_cols))
    n_vertex_numerical = 2**len(numerical_cols)
    
    # Check that there's enough data points to obtain the vertex of the hypercube
    if n_vertex > len(dataset_mat):
        raise ValueError("ERROR! Insufficient data points") 
    
    # Scaling numerical data
    sc = StandardScaler()
    
    if len(numerical_cols_to_scale):
        X_train = dataset_mat[numerical_cols_to_scale]
        X_train = sc.fit_transform(X_train)
    else:
        X_train = dataset_mat
    
    X_train_model = X_train
    
    for col in numerical_cols_dont_scale:
        X_train_model = np.insert(X_train_model, np.shape(X_train_model)[1], dataset_mat[col].values, axis=1)
    
    X_train_all = X_train_model # X_train with all the numerical columns
    
    for col in categorical_cols+numerical_cols_dont_scale:
        X_train_model = np.insert(X_train_model, np.shape(X_train_model)[1], dataset_mat[col].values, axis=1)
    
    # Train OneClassSVM
    model = svm.OneClassSVM(**dct_params)
    model.fit(X_train_model)
    preds = pd.DataFrame({"predictions":list(model.predict(X_train_model))}) 
    preds["distances"] = model.decision_function(X_train_model)
    df_anomalies = pd.merge(dataset_mat, preds, left_index=True, right_index=True)
    
    df_anomalies_no = df_anomalies[df_anomalies['predictions']==1].sort_values(by="distances", ascending=True).drop(['predictions', 'distances'], axis=1)
    df_anomalies_yes = df_anomalies[df_anomalies['predictions']==-1].sort_values(by="distances", ascending=True).drop(['predictions', 'distances'], axis=1)
    
    
    # Case 1: Only numerical variables
    if len(categorical_cols) == 0:
        
        # Obtain rules
        df_bounds = obtain_rules(df_anomalies_no, df_anomalies_yes, X_train, sc, n_vertex_numerical, numerical_cols, numerical_cols_dont_scale)
        
        # Extract rules
        list_rules_total = []
        print("NOT anomaly...")
        j = 0
        for df in df_bounds:
            j += 1
            print("----- Subgroup {0} ------".format(j))
            list_rules = print_rules(df, categorical_cols)
            list_rules_total.append(list_rules)
            print("")
        
    
    # Case 2: Only categorical variables
    elif len(numerical_cols) == 0:
        df_bounds = df_anomalies_no[categorical_cols].drop_duplicates().reset_index(drop=True)
        
        # Extract rules
        print("NOT anomaly...")
        list_rules = print_rules(df_bounds, categorical_cols)
        
    # Case 3: Numerical + Categorical
    else:
        df_cat = df_anomalies_no[categorical_cols]
        df_cat_unique = df_cat.drop_duplicates()
        df_cat_yes = df_anomalies_yes[categorical_cols]
        df_bounds_all = []
        
        j=0
        for i, row in df_cat_unique.iterrows():
            j += 1
            print("Category {0}".format(j))
            
            # Obtain sub-hypercube (not outliers)
            list_index = df_cat[df_cat[row.index]==row.values].dropna().index # index for that sub-hypercube
            df_anomalies_no_sub = df_anomalies_no[(df_anomalies_no.index.isin(list_index))].copy() # sub-hypercube
            
            # Outliers for this iteration
            list_index_yes = df_cat_yes[df_cat_yes[row.index]==row.values].dropna().index
            df_anomalies_yes_sub = df_anomalies_yes[(df_anomalies_yes.index.isin(list_index_yes))].copy() # outliers for this iteration
              
            # Obtain vertices for this iteration
            df_bounds = obtain_rules(df_anomalies_no_sub[numerical_cols].copy(), df_anomalies_yes_sub[numerical_cols].copy(), X_train, sc, n_vertex_numerical, numerical_cols)

            for col in categorical_cols:
                for df in df_bounds:
                    df[col] = row[col]
            
            df_bounds_all.append(df_bounds)
            print("")
            
        df_bounds = df_bounds_all
        
        # Extract rules
        i = 0
        list_rules_total = []
        print("NOT anomaly...")
        for rules_cat in df_bounds:
            i += 1
            print("*"*75)
            print("Combination of categorical variables Nº {0} ".format(i))
            j = 0
            for df in rules_cat:
                j += 1
                print("----- Subgroup {0} ------".format(j))
                list_rules = print_rules(df, categorical_cols)
                list_rules_total.append(list_rules)
                print("")
            print("*"*75)
    
    return model, df_bounds, df_anomalies
