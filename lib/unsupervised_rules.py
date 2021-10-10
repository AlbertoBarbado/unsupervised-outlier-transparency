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
from kmodes.kprototypes import KPrototypes
from joblib import Parallel, delayed

import pickle

MAX_ITERS = 100

def obtain_centroid(X_train, clustering_algorithm, categorical_cols, sc, n_clusters):
    """
    Function to obtain the centroid of a group of data points. 
    It uses K-Prototypes or with K-Modes algorithm.
    
    It returns the centroid/prototype point
    for that data as well as the assigned clusters for each datapoint.
    
    """
    if clustering_algorithm == "kprototypes":
        
        if n_clusters >= len(X_train):
            print("Warning - more points than clusters! Using one cluster less")
            n_clusters = n_clusters - 1
        
        kp = KPrototypes(n_clusters=n_clusters,
                         init='Huang',
                         max_iter=5,
                         n_init=5,
                         verbose=0)
        
        # Columns with categorical features.
        idx = 0
        list_cat = []
        for feature in list(X_train.columns):
            if feature in categorical_cols:
                list_cat.append(idx)
            idx += 1
        
        labels = kp.fit_predict(X_train.values, categorical=list_cat)
        centroid = kp.cluster_centroids_
        
        if False:
            centroid[0] = sc.inverse_transform(centroid[0])
            list_aux = list(range(len(X_train.columns)))
            aux = pd.DataFrame(centroid[1]).rename(columns={i:j for i,j in zip(list_aux[:len(list_cat)], list_cat)})
            aux = pd.DataFrame(centroid[0]).join(aux)
            centroid = aux.values
            centroid = pd.DataFrame(centroid)
        else:
            aux = centroid
            # centroid = pd.concat([pd.DataFrame(x) for x in centroid]).T
            centroid = pd.DataFrame(centroid)
            centroid = centroid.drop(columns=list_cat, errors="ignore")
            centroid = pd.DataFrame(sc.inverse_transform(centroid))
            # aux = pd.concat([pd.DataFrame(x) for x in aux]).T
            aux = pd.DataFrame(aux)
            centroid[list_cat] = pd.DataFrame(aux)[list_cat]
        
    elif clustering_algorithm == "kmeans":
        if n_clusters >= len(X_train):
            n_clusters = len(X_train)
        kmeans = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            max_iter=100,
            n_init=10,
            random_state=0)
        labels = kmeans.fit_predict(X_train)
        centroid = kmeans.cluster_centers_
        
        # Inverse transform cluster value
        centroid = sc.inverse_transform(centroid)
        centroid = pd.DataFrame(centroid)
        
    else:
        raise ValueError("Cluster method {0} not recognised; Use 'kmeans' or 'kprototypes' instead".format(clustering_algorithm))
    
    return pd.DataFrame({'labels': labels}), centroid


def obtain_vertices(df_anomalies_no_sub, X_train, sc, n_vertex, numerical_cols,
                    categorical_cols, clustering_algorithm, n_clusters):
    """
    Function to obtain the vertices from the hypercube of data points contained
    in a dataframe
    
    """
    # Drop duplicate rows
    rule_subgroups = {}

    # Obtain centroid to calculate the vertices in case there're enough points
    df_train = pd.DataFrame()
    for col, j in zip(numerical_cols, range(X_train.shape[1])):
        df_train[col] = X_train[:, j]

    df_train_no = df_train.copy().loc[list(
        df_anomalies_no_sub.index)]
    
    if clustering_algorithm == "kprototypes":
        feature_cols = numerical_cols + categorical_cols
        # If kprotoypes, add categorical columns
        df_train_no = df_train_no.join(df_anomalies_no_sub[categorical_cols])
    else:
        feature_cols = numerical_cols

    # Obtain centroid for that sub-hypercube
    n = n_clusters
    labels, centroid_no = obtain_centroid(df_train_no, clustering_algorithm, 
                                          categorical_cols, sc, n)
    df_anomalies_no_sub['cluster_label'] = list(labels['labels'].values)

    for i in range(n):
        df_no = df_anomalies_no_sub[df_anomalies_no_sub[
            'cluster_label'] == i].copy().drop(
                ['cluster_label'], axis=1)
        idx = list(df_no.index)
        # Need at least datapoints for vertices for a number
        # of hypercubes equal to the number of clusters
        if len(df_no) >= n_vertex:

            # Obtain vertices
            # Euclidean distance of each datapoint to centroid
            df_no['distances'] = np.linalg.norm(
                df_no[feature_cols].sub(
                    np.array(centroid_no.loc[i].values.squeeze())),
                axis=1)
            df_no = df_no.sort_values(
                by=numerical_cols + ['distances'], ascending=True)
            # For each numerical col choose a datapoint with max/min value;
            # if there's a tie choose considering distances and choosing
            # the furthest ones
            list_index = []
            for col in numerical_cols:
                max_col = df_no[col].max()
                min_col = df_no[col].min()
                df_sub = df_no[(df_no[col] == max_col)
                               | (df_no[col] == min_col)].sort_values(
                                   by=[col] + ['distances'], ascending=True)
                list_index.append(list(df_sub.head(1).index)[0])
                list_index.append(list(df_sub.tail(1).index)[0])

            vectors_bound = df_no[df_no.index.isin(list_index)]
            centroid = centroid_no.loc[i].values.squeeze()

        # In case there's less data in the sub-hypercube 
        # than the number of required vertices, all points are selected
        # no need to obtain the centroid
        else:
            #            print("Datapoints = {0} | Few data points in this iteration; using all of them as rules.".format(len(df_no)))
            vectors_bound = df_no.copy()
            centroid = None
            vectors_bound['distances'] = None
        
        rule_subgroups[i] = [vectors_bound, centroid, df_no]

    return rule_subgroups


def obtain_limits(df):
    """
    # TODO
    """

    # Obtain limits
    vectors_bound_all = df.copy()
    vectors_bound_all.drop(["distances"], axis=1, inplace=True)
    df_bounds_max = vectors_bound_all.max().reset_index().rename(
        columns={
            'index': 'cat'
        }).transpose()  # df with the max variable values on the hyperplanes
    df_bounds_max.columns = df_bounds_max.loc['cat']
    df_bounds_max = df_bounds_max.reindex(df_bounds_max.index.drop('cat'))
    aux = [col + '_max' for col in df_bounds_max.columns]
    df_bounds_max.columns = aux

    df_bounds_min = vectors_bound_all.min().reset_index().rename(
        columns={
            'index': 'cat'
        }).transpose()  # df with the min variable values on the hyperplanes
    df_bounds_min.columns = df_bounds_min.loc['cat']
    df_bounds_min = df_bounds_min.reindex(df_bounds_min.index.drop('cat'))
    aux = [col + '_min' for col in df_bounds_min.columns]
    df_bounds_min.columns = aux

    df_bounds = df_bounds_max.join(df_bounds_min, how='inner')

    return df_bounds


def function_check(x, limits, numerical_cols):
    """
    True: outside hypercube
    False: not outside hypercube
    """

    result = False
    for col in numerical_cols:
        l_max = limits[col + '_max'][0]
        l_min = limits[col + '_min'][0]

        # If its outside from some of the limits, then its outside the hypercube
        if (x[col] > l_max) or (x[col] < l_min):
            result = True

    return result


def obtain_rules_discard(df_anomalies_no_sub, df_anomalies_yes_sub, X_train, sc,
                         n_vertex_numerical, numerical_cols, categorical_cols,
                         clustering_algorithm, use_inverse):
    """
    # TODO
    Function that instead of using more clusters until a number is reached that
    fits all non anomalous data without the anomalies, it fits all the not
    anomalous datapoints into two clusters, and then, if they contain anomalies, 
    they keep fitting each subspace into another two clusters;
    when a fitted subspace does not have anomalies, it is removed
    from the used space. It keeps doing this recursively until all
    non anomalous datapoints are fitted.
    """

    def hyper_limits(vectors_bound_all, df_anomalies_yes_sub, numerical_cols):
        limits = obtain_limits(vectors_bound_all)
        df_anomalies_yes_sub["outside_hcube"] = df_anomalies_yes_sub.apply(
            lambda x: function_check(x, limits, numerical_cols), axis=1)
        return df_anomalies_yes_sub, limits

    if clustering_algorithm == "kprototypes":
        feature_cols = numerical_cols + categorical_cols
    else:
        feature_cols = numerical_cols

    # Tolerance param
    max_iters = MAX_ITERS

    # Obtain vertices
    n = 0
    check = True

    # Drop duplicates
    df_anomalies_no_sub.drop_duplicates(inplace=True)
    df_anomalies_yes_sub.drop_duplicates(inplace=True)
    df_final = []
    
    # Ñapa: duplicate datapoints if below 2
    if len(df_anomalies_no_sub)<2:
        df_anomalies_no_sub = df_anomalies_no_sub.append(df_anomalies_no_sub)
        df_anomalies_no_sub = df_anomalies_no_sub.reset_index(drop=True)
    
    if len(df_anomalies_yes_sub)<2:
        df_anomalies_yes_sub = df_anomalies_yes_sub.append(df_anomalies_no_sub)
        df_anomalies_yes_sub = df_anomalies_yes_sub.reset_index(drop=True)
    
    # Data used -- start using all and 1 cluster
    dct_subdata = {"data": df_anomalies_no_sub, "n_clusters": 1}
    list_subdata = [dct_subdata]

    # Check until all non anomalous data is used for rule inferring
    j = 0
    while check:
        # When there is no data to infer rules, finish
        if len(list_subdata) == 0:
            break
        list_original = list_subdata.copy()
        list_subdata = []  # Reset list
        # For each subdata space, use two clusters to try and infer rules
        for dct_subdata in list_original:
            # Load data
            df_anomaly_no = dct_subdata['data']
            n = dct_subdata['n_clusters']
            j += 1

            # Check tolerance
            if j >= max_iters:
                check=False
                break
            # If there is only one point left, skip it
            elif n > len(df_anomaly_no):
                continue

            # Rules
            print("Iteration {0} | nº clusters used {1}".format(j, n))
            # Returns n_vertex_numerical datapoints
            # if n_vertex_numerical > len(df_anomalies_no) for each cluster;
            # else returns df_anomalies_no
            dict_vectors_bound_all = obtain_vertices(
                df_anomaly_no,
                X_train,
                sc,
                n_vertex_numerical,
                numerical_cols,
                categorical_cols,
                clustering_algorithm,
                n_clusters=n)

            # For each cluster in that subdata
            for key, value in dict_vectors_bound_all.items():
                vectors_bound_all = value[0].copy()
                df_anomalies_yes_sub, limits = hyper_limits(
                    vectors_bound_all, df_anomalies_yes_sub, feature_cols)
                list_check = list(
                    df_anomalies_yes_sub["outside_hcube"].unique())

                # Recover original indexes
                df_anomaly_iter = value[2]
                df_aux = df_anomaly_no.copy().reset_index()
                cols_merge = [
                    column for column in list(df_anomaly_iter.columns)
                    if column != "distances"
                ]
                df_anomaly_iter = df_anomaly_iter[cols_merge]
                df_anomaly_iter = df_anomaly_iter.merge(
                    df_aux,
                    how="left",
                    left_on=cols_merge,
                    right_on=cols_merge)
                df_anomaly_iter.index = df_anomaly_iter['index']
                del df_anomaly_iter['index']

                # If there are points that belong to the other class,
                # retrain with one more cluster
                if False in list_check:
                    dct_subdata = {'data': df_anomaly_iter, 'n_clusters': 2}
                    list_subdata.append(dct_subdata)
                # When there are no points from the other class,
                # turn into rules (and do not use those points again)
                elif len(df_anomaly_no)==1.:
                    df_final.append(limits)
                else:
                    df_final.append(limits)

    return df_final


def obtain_rules_keep(df_anomalies_no_sub, df_anomalies_yes_sub, X_train, sc,
                      n_vertex_numerical, numerical_cols, categorical_cols,
                      clustering_algorithm, use_inverse):
    """
    # TODO
    
    This function keeps increasing the number of clusters until there are
    no outliers inside the rules or the tolerance criterion is reached.
    
    """

    # Obtain vertices
    n = 0
    check = True
    epsilon = 0.1
    max_iter = MAX_ITERS
    #    e = int(np.round(epsilon*len(df_anomalies_no_sub)))
    hard_exit = False

    # Drop duplicates
    df_anomalies_no_sub.drop_duplicates(inplace=True)
    df_anomalies_yes_sub.drop_duplicates(inplace=True)
    
    # Ñapa: duplicate datapoints if below 2
    if len(df_anomalies_no_sub)<2:
        df_anomalies_no_sub = df_anomalies_no_sub.append(df_anomalies_no_sub)
        df_anomalies_no_sub = df_anomalies_no_sub.reset_index(drop=True)
    
    if len(df_anomalies_yes_sub)<2:
        df_anomalies_yes_sub = df_anomalies_yes_sub.append(df_anomalies_no_sub)
        df_anomalies_yes_sub = df_anomalies_yes_sub.reset_index(drop=True)

    while check:
        # Rules
        df_final = []
        n += 1
        print("")
        print("*" * 100)
        print("Iteration {0} | nº clusters used {0}".format(n))
        # Returns n_vertex_numerical datapoints
        # if n_vertex_numerical > len(df_anomalies_no) for each cluster;
        # else returns df_anomalies_no
        try:
            dict_vectors_bound_all = obtain_vertices(
                df_anomalies_no_sub,
                X_train,
                sc,
                n_vertex_numerical,
                numerical_cols,
                categorical_cols,
                clustering_algorithm,
                n_clusters=n)
        except ValueError:
            print("Too many clusters for the remaining datapoints -- ending process")
            # Use previous iteration
            dict_vectors_bound_all = obtain_vertices(
                df_anomalies_no_sub,
                X_train,
                sc,
                n_vertex_numerical,
                numerical_cols,
                categorical_cols,
                clustering_algorithm,
                n_clusters=n-2)
            
            # And specify that this is the last iter
            hard_exit = True

        for key, value in dict_vectors_bound_all.items():
            vectors_bound_all = value[0].copy()
            n_points_cluster = len(value[2])

            ### Check if cluster is empty - if it's empty, ignore it
            if vectors_bound_all.empty:
                check = False
                continue

            ### Check if a datapoint anomalous would be inside the not anomalous hypercube
            # Case where all the points are used as vertices
            if vectors_bound_all['distances'].iloc[0] == None:
                # Try to create fake hypercube and see if no anomalies fall within it
                fake_hyper_limits = obtain_limits(vectors_bound_all)
                df_anomalies_yes_sub["outside_hcube"] = df_anomalies_yes_sub.apply(
                    lambda x: function_check(x, fake_hyper_limits, numerical_cols),
                    axis=1)
                list_check = list(
                    df_anomalies_yes_sub["outside_hcube"].unique())
                n_inside = len(df_anomalies_yes_sub[df_anomalies_yes_sub[
                    "outside_hcube"] == False])

                # If outliers inside the rules and not reaching the iter limit
                # If hard condition is reached
                if hard_exit:
                   check = False
                   df_final.append(fake_hyper_limits)
                
                elif False in list_check and n < max_iter:
                    print("Less points than vertices...")
                    print("Points cluster: {0} | Nº Vertices: {1}".format(
                        n_points_cluster, n_vertex_numerical))
                    
                    # Convergence Criteria 1 - Small clusters
                    threshold = epsilon * n_vertex_numerical
                    threshold = threshold if threshold > 1 else 1

                    if n_points_cluster <= threshold:
                        check = True
                        break
                    # Convergence Criteria 2 - More datapoints than clusters
                    elif n > len(df_anomalies_no_sub) -1:
                        check = True
                        break
                    # If nothing happens, try again
                    else:
                        df_final.append(fake_hyper_limits)
                        check = False
                        
                # If the iter limit is reached
                else:
                    check = False
                    df_final.append(fake_hyper_limits)

            # Case where only some points are used as vertices
            else:
                limits = obtain_limits(vectors_bound_all)
                df_anomalies_yes_sub[
                    "outside_hcube"] = df_anomalies_yes_sub.apply(
                        lambda x: function_check(x, limits, numerical_cols),
                        axis=1)
                n_inside = len(df_anomalies_yes_sub[df_anomalies_yes_sub[
                    "outside_hcube"] == False])
                list_check = list(
                    df_anomalies_yes_sub["outside_hcube"].unique())
                
                # If hard condition, end
                if hard_exit:
                    check = False
                    df_final.append(limits)

                # If at least one anomalous point is inside the hypercube,
                # repeat again for ALL points with one more cluster
                elif False in list_check and n < max_iter:
                    print("Anomalies inside hypercube!. Key: ", key, "|list_check: ", list_check,
                          "|n_anomalies: ", n_inside)
                    check = True
                    break
                else:
                    check = False
                    df_final.append(limits)

    return df_final


def obtain_rules_keep_reset(df_anomalies_no_sub, df_anomalies_yes_sub, X_train,
                            sc, n_vertex_numerical, numerical_cols,
                            categorical_cols, clustering_algorithm, use_inverse):
    """
    # TODO
    
    Just like obtain_rules_keep() but after each iteration the inliers
    inside rules without outliers are removed from the dataset and the system
    begins again for the remaining inliers.
    
    In this case, this function will keep iterating till all inliers are within 
    a rule.
    
    """
    # Obtain vertices
    n = 0
    # epsilon = 0.1
    #    e = int(np.round(epsilon*len(df_anomalies_no_sub)))
    # Tolerance param
    max_iters = MAX_ITERS

    # Drop duplicates
    df_anomalies_no_sub.drop_duplicates(inplace=True)
    df_anomalies_yes_sub.drop_duplicates(inplace=True)
    
    # Ñapa: duplicate datapoints if below 2
    if len(df_anomalies_no_sub)<2:
        df_anomalies_no_sub = df_anomalies_no_sub.append(df_anomalies_no_sub)
        df_anomalies_no_sub = df_anomalies_no_sub.reset_index(drop=True)
    
    if len(df_anomalies_yes_sub)<2:
        df_anomalies_yes_sub = df_anomalies_yes_sub.append(df_anomalies_yes_sub)
        df_anomalies_yes_sub = df_anomalies_yes_sub.reset_index(drop=True)
    
    df_final = []
    df_datapoints_used = df_anomalies_no_sub.copy()
    j = 0
    while len(df_datapoints_used) != 0:
        n += 1
        j += 1
        
        # Check limit
        if j > max_iters:
            print("Max iters {max_iters} reached! Finishing process...".format(max_iters=max_iters))
            break
        # Rules
        df_bounds = []
        print("")
        print("*" * 100)
        print("Iteration {0} | nº clusters used {1}".format(j, n))
        print("Remaining datapoints {0}/{1}".format(len(df_datapoints_used), len(df_anomalies_no_sub)))
        # Returns n_vertex_numerical datapoints
        # if n_vertex_numerical > len(df_anomalies_no) for each cluster;
        # else returns df_anomalies_no
        try:
            dict_vectors_bound_all = obtain_vertices(
                df_datapoints_used,
                X_train,
                sc,
                n_vertex_numerical,
                numerical_cols,
                categorical_cols,
                clustering_algorithm,
                n_clusters=n)
        except ValueError:
            print("Too many clusters for the remaining datapoints -- ending process")
            print(n, len(df_datapoints_used))
            dict_vectors_bound_all = obtain_vertices(
                df_datapoints_used,
                X_train,
                sc,
                n_vertex_numerical,
                numerical_cols,
                categorical_cols,
                clustering_algorithm,
                n_clusters=n-3)
            
            df_datapoints_used = pd.DataFrame()

        for key, value in dict_vectors_bound_all.items():
            vectors_bound_all = value[0].copy()
            n_points_cluster = len(value[2])

            ### Check if cluster is empty - if it's empty, ignore it
            if vectors_bound_all.empty:
                continue

            ### Check if a datapoint anomalous would be inside the not anomalous hypercube
            # Case where all the points are used as vertices
            if vectors_bound_all['distances'].iloc[0] == None:
                # Try to create fake hypercube and see if no anomalies fall within it
                fake_hyper_limits = obtain_limits(vectors_bound_all)
                df_anomalies_yes_sub["outside_hcube"] = df_anomalies_yes_sub.apply(
                    lambda x: function_check(x, fake_hyper_limits, numerical_cols),
                    axis=1)
                list_check = list(
                    df_anomalies_yes_sub["outside_hcube"].unique())
                n_inside = len(df_anomalies_yes_sub[df_anomalies_yes_sub[
                    "outside_hcube"] == False])

                # Append rules
                df_bounds.append(fake_hyper_limits)
                if False in list_check:
                    print("Less points than vertices...")
                    print("Points cluster: {0} | Nº Vertices: {1}".format(
                        n_points_cluster, n_vertex_numerical))
                    
                    # # Convergence Criteria 1 - Small clusters
                    # threshold = epsilon * n_vertex_numerical
                    # threshold = threshold if threshold > 1 else 1

                    # if n_points_cluster <= threshold:
                    #     df_datapoints_used = df_datapoints_used[~df_datapoints_used.index.isin(value[2].index)]
                    #     df_final.append(fake_hyper_limits)
                    #     n = 0
                    # Convergence Criteria 2 - As many points as clusters
                    if n == len(df_datapoints_used) -1:
                        df_datapoints_used = df_datapoints_used[~df_datapoints_used.index.isin(value[2].index)]
                        df_final.append(fake_hyper_limits)
                        n = 0
                        
                    # If nothing happens, try again
                    else:
                        continue
                else:
                    df_datapoints_used = df_datapoints_used[~df_datapoints_used.index.isin(value[2].index)]
                    df_final.append(fake_hyper_limits)

            # Case where only some points are used as vertices
            else:
                limits = obtain_limits(vectors_bound_all)
                df_bounds.append(limits)

                df_anomalies_yes_sub[
                    "outside_hcube"] = df_anomalies_yes_sub.apply(
                        lambda x: function_check(x, limits, numerical_cols),
                        axis=1)
                n_inside = len(df_anomalies_yes_sub[df_anomalies_yes_sub[
                    "outside_hcube"] == False])
                list_check = list(
                    df_anomalies_yes_sub["outside_hcube"].unique())

                # If at least one anomalous point is inside the hypercube,
                # repeat again for ALL points with one more cluster
                if False in list_check:
                    print("Anomalies inside hypercube!. Key: ", key, "|list_check: ", list_check,
                          "|n_anomalies: ", n_inside)
                    continue
                
                # Remove those datapoints from dataframe and append rules
                else:
                    df_datapoints_used = df_datapoints_used[~df_datapoints_used.index.isin(value[2].index)]
                    df_final.append(limits)
                    n = 0

    return df_final



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
            elif col[-4:] == '_max':
                s += '{col} <= {value} AND '.format(col=col, value=value)
            else:
                s += '{col} >= {value} AND '.format(col=col, value=value)

        s = s[:-4]  # delete last AND
        s = s.replace('_max', '').replace('_min', '')
        print(s)
        list_rules.append(s)

    return list_rules


def rule_prunning(list_cols, rule_check, rule_ref):
    """
    # TODO
    """
    list_check = []
    for col in list_cols:
        if "min" in col:
            if rule_check[col] >= rule_ref[col]:
                list_check.append(True)
            else:
                list_check.append(False)
        elif "max" in col:
            if rule_check[col] <= rule_ref[col]:
                list_check.append(True)
            else:
                list_check.append(False)

    if False in list_check:
        return rule_check
    else:
        return rule_ref


def simplify_rules_alt(categorical_cols, df_all):
    """
    Function to reduce the number of rules checking if they are contained inside
    another hypercube. 
    
    It can receive a list of categorical_cols in order to analyze the subsets
    belonging to the different categorical combinations independently; in order
    to analyze all the hyperspace together, that list should be empty.
    
    # TODO
    """

    def iter_prunning(categorical_cols, df_all):
        def choose_one(i, rule_check, cols, df_cat):
            print("Iter {0}/{1}".format(i, len(df_all)))
            df_new = pd.concat([
                pd.DataFrame(rule_prunning(cols, rule_check, rule_ref)).T
                for j, rule_ref in df_cat.iterrows() if i != j
            ])
            cols_max = [col for col in list(df_new.columns) if "max" in col]
            cols_min = [col for col in list(df_new.columns) if "min" in col]
            df_new = df_new.sort_values(
                by=cols_max + cols_min,
                ascending=[False] * len(cols_max) + [True] * len(cols_min))
            return df_new.head(1)

        cols = [x for x in list(df_all.columns) if x not in categorical_cols]

        if len(categorical_cols) > 0:
            df_end = pd.DataFrame()
            if len(categorical_cols) > 0:
                df_cat = df_all[categorical_cols]
                df_cat_unique = df_cat.drop_duplicates()
                for i, row in df_cat_unique.iterrows():
                    print("Iter {0}".format(i))
                    #                df_all_aux = df_all.copy()
                    #                for col in categorical_cols:
                    #                    df_all_aux = df_all_aux[df_all_aux[col]==row[col]]
                    list_index = df_cat[df_cat[row.index] ==
                                        row.values].dropna().index
                    df_all_aux = df_all[(df_all.index.isin(list_index))].copy()

                    if len(df_all_aux) > 1:
                        df_iter = pd.concat([
                            choose_one(i, rule_check, cols, df_all_aux)
                            for i, rule_check in df_all_aux.iterrows()
                        ])
                    else:
                        df_iter = df_all_aux

                    if df_end.empty:
                        df_end = df_iter
                    else:
                        df_end = df_end.append(df_iter)

            df_end = df_end.drop_duplicates()
        else:
            df_end = pd.concat([
                choose_one(i, rule_check, cols, df_all)
                for i, rule_check in df_all.iterrows()
            ])
        return df_end

    check = True
    while check:
        df_end = iter_prunning(categorical_cols, df_all)

        if len(df_end) == len(df_all):
            print("No more improvements... finishing up")
            check = False
        else:
            # New iter with pruned rules
            df_all = df_end.drop_duplicates().reset_index(drop=True)

    return df_end



def ocsvm_rule_extractor(dataset_mat, numerical_cols, categorical_cols,
                         clustering_algorithm, method, use_inverse,
                         dct_params, path_save_model="",
                         store_intermediate=False):
    """
    Function to extract rules that justify in a comprehensive way why some
    data points  are identified as outliers. 
    The function returns a dataframe with the boundaries that define those
    rules according to the different features used as well
    as the model trained.
    
    """
    
    PATH_SAVE_MODEL = path_save_model
    STORE_INTERMEDIATE = store_intermediate

    # Check data quantity
    n_vertex = (len(categorical_cols) + 1) * 2**(len(numerical_cols))
    n_vertex_numerical = 2**len(numerical_cols)

    # Scaling numerical data
    sc = StandardScaler()

    if len(numerical_cols):
        X_train = dataset_mat[numerical_cols]
        X_train = sc.fit_transform(X_train)
    else:
        X_train = dataset_mat

    X_train_model = X_train

    i = 0
    for col in categorical_cols:
        i += 1
        print(i, "/", len(categorical_cols))
        X_train_model = np.insert(
            X_train_model,
            np.shape(X_train_model)[1],
            dataset_mat[col].values,
            axis=1)
        
    pickle.dump(sc, open("{0}/sc.p".format(PATH_SAVE_MODEL), "wb"))
    
    # If there are no categorical cols, always use kmeans
    if len(categorical_cols) == 0 and clustering_algorithm == "kprototypes":
        print("No categorical cols, using K means instead")
        clustering_algorithm = "kmeans"

    # TODO: Save model
    if STORE_INTERMEDIATE:
        try:
            print("Loading model")
            model = pickle.load(open("{0}/backup.p".format(PATH_SAVE_MODEL), "rb"))
            print("Model loaded!")
        except:
            # Train OneClassSVM
            print("Fitting model!")
            model = svm.OneClassSVM(**dct_params)
            model.fit(X_train_model)
            print("Model fitted!")
            pickle.dump(model, open("{0}/backup.p".format(PATH_SAVE_MODEL), "wb"))
    else:
        # Train OneClassSVM
        print("Fitting model!")
        model = svm.OneClassSVM(**dct_params)
        model.fit(X_train_model)
        print("Model fitted!")
        
    print("Continuing process...")
    preds = pd.DataFrame({"predictions": list(model.predict(X_train_model))})
    preds["distances"] = model.decision_function(X_train_model)
    df_anomalies = pd.merge(
        dataset_mat.reset_index(drop=True), 
        preds.reset_index(drop=True), 
        left_index=True,
        right_index=True
        )

    # If True, obtain rules for Anomalous datapoints
    if use_inverse:
        df_anomalies['distances'] = -1 * df_anomalies['distances']
        df_anomalies['predictions'] = -1 * df_anomalies['predictions']

    # Split outliers/inliers
    df_anomalies_no = df_anomalies[df_anomalies['predictions'] ==
                                   1].sort_values(
                                       by="distances", ascending=True).drop(
                                           ['predictions', 'distances'],
                                           axis=1)
    df_anomalies_yes = df_anomalies[df_anomalies['predictions'] ==
                                    -1].sort_values(
                                        by="distances", ascending=True).drop(
                                            ['predictions', 'distances'],
                                            axis=1)

    # If there are categorical cols and user wants to use k-prototypes.
    if clustering_algorithm == "kprototypes" and len(categorical_cols) > 0:
        categorical_cols_train = [number for number in range(X_train.shape[1],
                                                             X_train_model.shape[1])]
        
        if method == "discard":
            # Obtain rules
            df_bounds = obtain_rules_discard(df_anomalies_no, df_anomalies_yes,
                                             X_train_model, sc, n_vertex_numerical,
                                             numerical_cols, categorical_cols,
                                             clustering_algorithm, use_inverse) 
        elif method == "keep":
            df_bounds = obtain_rules_keep(df_anomalies_no, df_anomalies_yes,
                                          X_train_model, sc, n_vertex_numerical,
                                          numerical_cols, categorical_cols,
                                          clustering_algorithm, use_inverse) 
        elif method == "keep_reset":
            df_bounds = obtain_rules_keep_reset(df_anomalies_no, df_anomalies_yes,
                                                X_train_model, sc, n_vertex_numerical,
                                                numerical_cols, categorical_cols,
                                                clustering_algorithm, use_inverse) 
        else:
            raise ValueError("Method {0} not implemented; Use 'keep' or 'discard' instead".format(method))
    
    # By default, use K-means++
    else:                                            
        # Case 1: Only numerical variables
        if len(categorical_cols) == 0:
          
            if method == "discard":
                # Obtain rules
                df_bounds = obtain_rules_discard(df_anomalies_no, df_anomalies_yes,
                                                 X_train, sc, n_vertex_numerical,
                                                 numerical_cols, categorical_cols,
                                                 clustering_algorithm, use_inverse) 
            elif method == "keep":
                df_bounds = obtain_rules_keep(df_anomalies_no, df_anomalies_yes,
                                              X_train, sc, n_vertex_numerical,
                                              numerical_cols, categorical_cols, 
                                              clustering_algorithm, use_inverse) 
            elif method == "keep_reset":
                df_bounds = obtain_rules_keep_reset(df_anomalies_no, df_anomalies_yes,
                                                    X_train, sc, n_vertex_numerical,
                                                    numerical_cols, categorical_cols, 
                                                    clustering_algorithm, use_inverse)
            else:
                raise ValueError("Method {0} not implemented; Use 'keep' or 'discard' instead".format(method))
    
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
            df_bounds = df_anomalies_no[
                categorical_cols].drop_duplicates().reset_index(drop=True)
    
            # Extract rules
            print("NOT anomaly...")
            list_rules = print_rules(df_bounds, categorical_cols)
    
        # Case 3: Numerical + Categorical
        else:
            df_cat = df_anomalies_no[categorical_cols]
            df_cat_unique = df_cat.drop_duplicates()
            df_cat_yes = df_anomalies_yes[categorical_cols]
            df_bounds_all = []
    
            j = 0
            for i, row in df_cat_unique.iterrows():
                j += 1
                print("Category {0}/{1}".format(j, len(df_cat_unique)))
    
                # Obtain sub-hypercube (not outliers)
                list_index = df_cat[df_cat[row.index] == row.values].dropna(
                ).index  # index for that sub-hypercube
                df_anomalies_no_sub = df_anomalies_no[(df_anomalies_no.index.isin(
                    list_index))].copy()  # sub-hypercube
    
                # Outliers for this iteration
                list_index_yes = df_cat_yes[df_cat_yes[row.index] ==
                                            row.values].dropna().index
                df_anomalies_yes_sub = df_anomalies_yes[(
                    df_anomalies_yes.index.isin(list_index_yes)
                )].copy()  # outliers for this iteration
    
                # If empty, skip
                if df_anomalies_yes_sub.empty or df_anomalies_no_sub.empty:
                    continue
                
                # Obtain vertices for this iteration
                if method == "discard":
                    # Obtain rules
                    df_bounds = obtain_rules_discard(df_anomalies_no_sub[numerical_cols].copy(),
                                                     df_anomalies_yes_sub[numerical_cols].copy(),
                                                     X_train, sc, n_vertex_numerical,
                                                     numerical_cols, categorical_cols,
                                                     clustering_algorithm, use_inverse)
                elif method == "keep":
                    df_bounds = obtain_rules_keep(df_anomalies_no_sub[numerical_cols].copy(),
                                                  df_anomalies_yes_sub[numerical_cols].copy(),
                                                  X_train, sc, n_vertex_numerical,
                                                  numerical_cols, categorical_cols,
                                                  clustering_algorithm, use_inverse) 
                elif method == "keep_reset":
                    df_bounds = obtain_rules_keep_reset(df_anomalies_no_sub[numerical_cols].copy(),
                                                        df_anomalies_yes_sub[numerical_cols].copy(),
                                                        X_train, sc, n_vertex_numerical,
                                                        numerical_cols, categorical_cols,
                                                        clustering_algorithm, use_inverse) 
                    
                else:
                    raise ValueError("Method {0} not implemented; Use 'keep', 'keep_reset' or 'discard' instead".format(method))
    
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
                print("*" * 75)
                print("Combination of categorical variables Nº {0} ".format(i))
                j = 0
                for df in rules_cat:
                    j += 1
                    print("----- Subgroup {0} ------".format(j))
                    list_rules = print_rules(df, categorical_cols)
                    list_rules_total.append(list_rules)
                    print("")
                print("*" * 75)
                
    
    ### Rule Pruning
    print("Starting pruning process")
    df_aux = []
    for df_category in df_bounds:
        if len(df_category) > 1:
            for df in df_category:
                df_aux.append(df)
        else:
            df_aux.append(df_category)
    
    df_aux = [df for df in df_aux if len(df)>0] # Drop empty lists
    df_aux = [df if type(df)!=list else df[0] for df in df_aux] # Extract dfs from lists
    df_all = pd.concat(df_aux)
    df_all = df_all.reset_index(drop=True)
    
    if len(df_all)>1:
        # Only reduce rules considering different subspaces if the clustert algorithm is kprototypes
        if clustering_algorithm == "kprototypes": 
            df_end = simplify_rules_alt([], df_all).drop_duplicates()
        else:
            df_end = simplify_rules_alt(categorical_cols, df_all).drop_duplicates()
        print("Pruned {0} to {1}".format(len(df_all), len(df_end)))
    else:
        df_end = df_all
    
    return model, sc, df_end, df_anomalies
