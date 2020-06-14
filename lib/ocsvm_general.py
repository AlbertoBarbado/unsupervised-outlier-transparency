# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:43:05 2019

@author: alber
"""
import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ruleset
import arff
import pickle

from lib.unsupervised_rules import ocsvm_rule_extractor
from joblib import Parallel, delayed
from sklearn import tree
from sklearn.tree.export import export_text
from alibi.explainers import AnchorTabular
from rulefit import RuleFit
from skrules import SkopeRules
from lib.unsupervised_rules import simplify_rules_alt
from aix360.algorithms.rbm import (BooleanRuleCG, FeatureBinarizer,
                                   LogisticRuleRegression, GLRMExplainer)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from itertools import combinations, permutations, product
from shapely.geometry import Polygon
from aix360.algorithms.protodash import ProtodashExplainer
from interpret.glassbox import DecisionListClassifier
from lib.config import N_JOBS
from lib.common import (train_one_class_svm, grid_search, tree_to_code, save_df_as_arff, 
                        check_datapoint_inside, check_datapoint_inside_only, turn_rules_to_df,
                        file_naming_ocsvm, plot_2D)
from lib.xai_metrics import (rule_overlapping_score, check_stability)

def ocsvm_rules_completion(df_anomalies, df_rules, numerical_cols, 
                           categorical_cols, inliers_used=True,
                           clustering_algorithm="kmeans", path="",  
                           file_name="default_name"):
    """
    Function to complete the information of the rules extracted; obtaining how many
    datapoints are inside the hypercubes.
    
    Rules have fixed size; for kmeans clustering, the input cols could contain 
    both categorical and numerical (regarding how the rules are checked).
    When the algorithm is kprototypes, all the columns are included within "numerical"
    since all the features will be treated the same way.

    Parameters
    ----------
    df_anomalies : TYPE
        DESCRIPTION.
    df_rules : TYPE
        DESCRIPTION.
    numerical_cols : TYPE
        DESCRIPTION.
    categorical_cols : TYPE
        DESCRIPTION.
    inliers_used : TYPE, optional
        DESCRIPTION. The default is True.
    clustering_algorithm : TYPE, optional
        DESCRIPTION. The default is "kmeans".
    path : TYPE, optional
        DESCRIPTION. The default is "".
    file_name : TYPE, optional
        DESCRIPTION. The default is "".

    Returns
    -------
    df_rules : TYPE
        DESCRIPTION.

    """
    
    if clustering_algorithm == "kprototypes":
        numerical_cols = numerical_cols + categorical_cols
        categorical_cols = []
    
    df_rules['n_inliers_included'] = 0
    df_rules['n_outliers_included'] = 0
    n_inliers = len(df_anomalies[df_anomalies['predictions']==1])
    n_outliers = len(df_anomalies[df_anomalies['predictions']==-1])
    n_vertex = (len(categorical_cols) + 1)*2**(len(numerical_cols))
    
    print("Checking inliers inside rules...")
    df_check = Parallel(n_jobs=N_JOBS)(delayed(check_datapoint_inside_only)(data_point,df_rules,numerical_cols,categorical_cols) for i, data_point in df_anomalies[df_anomalies['predictions']==1].iterrows())
    df_check = pd.concat([x[x['check']>0] for x in df_check])
    df_check = pd.DataFrame(df_check.groupby(df_check.index).sum()).reset_index()
    df_temp =  df_rules[['n_inliers_included']].reset_index()
    df_check = df_temp.merge(df_check, how="outer")[['check']].fillna(0)
    df_rules['n_inliers_included'] = df_check

    print("Checking outliers inside rules...")
    df_check = Parallel(n_jobs=N_JOBS)(delayed(check_datapoint_inside_only)(data_point,df_rules,numerical_cols,categorical_cols) for i, data_point in df_anomalies[df_anomalies['predictions']==-1].iterrows())
    df_check = pd.concat([x[x['check']>0] for x in df_check])
    df_check = pd.DataFrame(df_check.groupby(df_check.index).sum()).reset_index()
    df_temp =  df_rules[['n_inliers_included']].reset_index()
    df_check = df_temp.merge(df_check, how="outer")[['check']].fillna(0)
    df_rules['n_outliers_included'] = df_check
    
    # Check how many datapoints are included with the rules with Precision=1
    print("Checking inliers/outliers inside hypercubes with Precision=1...")
    n_inliers_p1 = 0
    n_inliers_p0 = 0
    n_outliers_p1 = 0
    n_outliers_p0 = 0
    n_inliers = len(df_anomalies[df_anomalies['predictions']==1])
    n_outliers = len(df_anomalies[df_anomalies['predictions']==-1])

    def wrapper_precision_check(data_point):
        df_rules['check'] = check_datapoint_inside(data_point,
                                                   df_rules,
                                                   numerical_cols,
                                                   categorical_cols)['check']
        n_inliers_p1 = 0
        n_inliers_p0 = 0
        n_outliers_p1 = 0
        n_outliers_p0 = 0
    
        if inliers_used:
            # If inlier
            if data_point['predictions']==1:
                # Rules with any P and that include this datapoint
                df_aux = df_rules[(df_rules['check']==1)] 
                if len(df_aux) > 0:
                    n_inliers_p0 += 1
                
                # Rules with P=1 and that include this datapoint
                df_aux = df_rules[(df_rules['n_outliers_included']==0)
                                  & (df_rules['check']==1)] 
                if len(df_aux) > 0:
                    n_inliers_p1 += 1
        else:
            # If outlier
            if data_point['predictions']==-1:
                # Rules with any P and that include this datapoint
                df_aux = df_rules[(df_rules['check']==1)] 
                if len(df_aux) > 0:
                    n_outliers_p0 += 1
                
                # Rules with P=1 and that include this datapoint
                df_aux = df_rules[(df_rules['n_inliers_included']==0)
                                  & (df_rules['check']==1)] 
                if len(df_aux) > 0:
                    n_outliers_p1 += 1
                    
        return {'n_inliers_p0':n_inliers_p0,
                'n_inliers_p1':n_inliers_p1,
                'n_outliers_p0':n_outliers_p0,
                'n_outliers_p1':n_outliers_p1}
                    
                    
    dct_out = Parallel(n_jobs=N_JOBS)(delayed(wrapper_precision_check)(data_point) for i, data_point in df_anomalies.iterrows())
    df_out = pd.DataFrame(dct_out).sum()
    
    for i, data_point in df_anomalies.iterrows():
        df_rules['check'] = check_datapoint_inside(data_point,
                                                   df_rules,
                                                   numerical_cols,
                                                   categorical_cols)['check']
        if inliers_used:
            # If inlier
            if data_point['predictions']==1:
                # Rules with any P and that include this datapoint
                df_aux = df_rules[(df_rules['check']==1)] 
                if len(df_aux) > 0:
                    n_inliers_p0 += 1
                
                # Rules with P=1 and that include this datapoint
                df_aux = df_rules[(df_rules['n_outliers_included']==0)
                                  & (df_rules['check']==1)] 
                if len(df_aux) > 0:
                    n_inliers_p1 += 1
        else:
            # If outlier
            if data_point['predictions']==-1:
                # Rules with any P and that include this datapoint
                df_aux = df_rules[(df_rules['check']==1)] 
                if len(df_aux) > 0:
                    n_outliers_p0 += 1
                
                # Rules with P=1 and that include this datapoint
                df_aux = df_rules[(df_rules['n_inliers_included']==0)
                                  & (df_rules['check']==1)] 
                if len(df_aux) > 0:
                    n_outliers_p1 += 1
    
    if inliers_used:
        df_rules['n_inliers'] = n_inliers
        df_rules['n_inliers_p0'] = df_out['n_inliers_p0']
        df_rules['n_inliers_p1'] = df_out['n_inliers_p1']
        try:
            del df_rules['check']
        except:
            pass
        path_aux = "inliers"
    else:
        df_rules['n_outliers_p1'] = df_out['n_outliers_p1']
        df_rules['n_outliers_p0'] = df_out['n_outliers_p0']
        df_rules['n_outliers'] = n_outliers
        try:
            del df_rules['check']
        except:
            pass
        path_aux = "outliers"
    
    # Save to CSV
    if len(path)>0:
        print("Saving results (all rules)...")
        df_rules.to_csv("{path}/{file_name}_rules_{type_r}_pruned_ocsvm.csv".format(path = path,
                                                                                    file_name = file_name,
                                                                                    type_r = path_aux),
                                    index=False)

    return df_rules



def ocsvm_rules(df_mat, numerical_cols, categorical_cols,
                cluster_algorithm, method, rules_used,
                dct_params, path, file_template,
                metrics=True,
                store_intermediate=False,
                plot_fig=False):
    """
    
    Parameters
    ----------
    df_mat : TYPE
        DESCRIPTION.
    numerical_cols : TYPE
        DESCRIPTION.
    categorical_cols : TYPE
        DESCRIPTION.
    cluster_algorithm : TYPE
        DESCRIPTION.
    method : TYPE
        DESCRIPTION.
    rules_used : TYPE
        DESCRIPTION.
    dct_params : TYPE
        DESCRIPTION.
    path_folder : TYPE
        DESCRIPTION.
    file_template : TYPE
        DESCRIPTION.
    plot_fig : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    
    print("Beginning process...")
    df_rules_inliers = pd.DataFrame()
    df_rules_outliers = pd.DataFrame()
    df_rules_inliers_p1 = pd.DataFrame()
    df_rules_outliers_p1 = pd.DataFrame()
    
    if rules_used == "all" or rules_used == "inliers":
        print("\n\n")
        print("*"*100)
        print("Obtaining Rules for Inliers...")
        print("*"*100)
        use_inverse = False
        file_name = file_naming_ocsvm(file_template=file_template,
                                      cluster_algorithm=cluster_algorithm,
                                      method=method,
                                      use_inverse=use_inverse)
    
        #### Obtain Rules [Inliers]
        if not store_intermediate:
            # Rules
            print("Fitting OCSVM model...")
            clf, sc, df_result, df_anomalies = ocsvm_rule_extractor(dataset_mat=df_mat,
                                                                    numerical_cols=numerical_cols,
                                                                    categorical_cols=categorical_cols,
                                                                    clustering_algorithm=cluster_algorithm,
                                                                    method=method,
                                                                    use_inverse=use_inverse,
                                                                    dct_params=dct_params,
                                                                    store_intermediate=store_intermediate,
                                                                    path_save_model=path)
            df_all = df_result
            
            df_no = df_anomalies[df_anomalies['predictions'] == 1]
            df_no = df_no.drop_duplicates()
            print(
                "Max different values (inliers) : {0} | Rules extracted {1}".format(
                    len(df_no), len(df_all)))
            if len(path)>0:
                print("Saving rules...")
                df_all.to_csv(path + '/df_rules_' + file_name + '.csv', index=False)
                df_anomalies.to_csv(path + '/df_anomalies_' + file_name + '.csv', index=False)
            
        else:
            try:
                df_all = pd.read_csv(path + '/df_rules_' + file_name + '.csv')
                df_anomalies = pd.read_csv(path + '/df_anomalies_' + file_name + '.csv')
                clf = pickle.load(open("{0}/backup.p".format(path), "rb"))
                sc = pickle.load(open("{0}/sc.p".format(path), "rb"))
            except:
                print("File not found! Fitting OCSVM model...")
                clf, sc, df_result, df_anomalies = ocsvm_rule_extractor(dataset_mat=df_mat,
                                                                        numerical_cols=numerical_cols,
                                                                        categorical_cols=categorical_cols,
                                                                        clustering_algorithm=cluster_algorithm,
                                                                        method=method,
                                                                        use_inverse=use_inverse,
                                                                        dct_params=dct_params,
                                                                        store_intermediate=store_intermediate,
                                                                        path_save_model=path)
                df_all = df_result
                
                df_no = df_anomalies[df_anomalies['predictions'] == 1]
                df_no = df_no.drop_duplicates()
                print(
                    "Max different values (inliers) : {0} | Rules extracted {1}".format(
                        len(df_no), len(df_all)))
                if len(path)>0:
                    print("Saving rules...")
                    df_all.to_csv(path + '/df_rules_' + file_name + '.csv', index=False)
                    df_anomalies.to_csv(path + '/df_anomalies_' + file_name + '.csv', index=False)
                
        
        # If kprototypes, do not consider "categorical cols" for the purpose of the rest of the code
        if cluster_algorithm == "kprototypes":
            feature_cols = list(set(numerical_cols + categorical_cols))
            cat_additional = []
        else:
            feature_cols = numerical_cols
            cat_additional = categorical_cols
        
        # Complete Rules
        print("Checking inliers inside hypercubes...")
        try:
            df_rules = pd.read_csv("{path}/{file_name}_rules_{type_r}_pruned_ocsvm.csv".format(path=path,
                                                                                               file_name=file_name,
                                                                                               type_r = "inliers"))
        except:
            df_rules = ocsvm_rules_completion(df_anomalies,
                                              df_all,
                                              feature_cols, 
                                              cat_additional,
                                              inliers_used=True,
                                              clustering_algorithm=cluster_algorithm,
                                              path=path,
                                              file_name=file_name)
        
        # Use only pure rules
        df_rules_inliers = df_rules
        df_rules = df_rules[df_rules["n_outliers_included"]==0]
        
        if metrics:
            print("Obtaining metrics...")
            df_rules = rule_overlapping_score(df_rules, df_anomalies,
                                              feature_cols, cat_additional)
            
            df_rules = check_stability(df_anomalies, df_rules, clf,
                                        feature_cols, cat_additional,
                                        using_inliers=True)
        
        
        # Saving rules obtained
        if len(path)>0:
            print("Saving rules...")
            df_rules.to_csv(path + '/df_rules_complete_' + file_name + '.csv', index=False)
    
        if plot_fig:
            #### Plot Rules [Inliers]
            print("Plotting rules for inliers...")
            df_rules = df_rules.copy()
            df_rules = df_rules.drop_duplicates().reset_index(drop=True)
            plot_2D(df_rules,
                    df_anomalies,
                    folder = path,
                    path_name=file_name)
            
        df_rules_inliers_p1 = df_rules
    
    if rules_used == "all" or rules_used == "outliers":
        print("\n\n")
        print("*"*100)
        print("Obtaining Rules for Outliers...")
        print("*"*100)
        
        #### Obtain Rules [Outliers]
        use_inverse = True
        file_name = file_naming_ocsvm(file_template=file_template,
                                      cluster_algorithm=cluster_algorithm,
                                      method=method,
                                      use_inverse=use_inverse) 
        
        if not store_intermediate:
            # Rules
            print("Fitting OCSVM model...")
            clf, sc, df_result, df_anomalies = ocsvm_rule_extractor(dataset_mat=df_mat,
                                                                    numerical_cols=numerical_cols,
                                                                    categorical_cols=categorical_cols,
                                                                    clustering_algorithm=cluster_algorithm,
                                                                    method=method,
                                                                    use_inverse=use_inverse,
                                                                    dct_params=dct_params,
                                                                    store_intermediate=False,
                                                                    path_save_model=path)
            df_all = df_result
            
            df_no = df_anomalies[df_anomalies['predictions'] == 1]
            df_no = df_no.drop_duplicates()
            print(
                "Max different values (outliers) : {0} | Rules extracted {1}".format(
                    len(df_no), len(df_all)))
            if len(path)>0:
                print("Saving rules...")
                df_all.to_csv(path + '/df_rules_' + file_name + '.csv', index=False)
                df_anomalies.to_csv(path + '/df_anomalies_' + file_name + '.csv', index=False)
            
        else:
            try:
                df_all = pd.read_csv(path + '/df_rules_' + file_name + '.csv')
                df_anomalies = pd.read_csv(path + '/df_anomalies_' + file_name + '.csv')
                clf = pickle.load(open("{0}/backup.p".format(path), "rb"))
                sc = pickle.load(open("{0}/sc.p".format(path), "rb"))
            except:
                print("File not found! Fitting OCSVM model...")
                clf, sc, df_result, df_anomalies = ocsvm_rule_extractor(dataset_mat=df_mat,
                                                                        numerical_cols=numerical_cols,
                                                                        categorical_cols=categorical_cols,
                                                                        clustering_algorithm=cluster_algorithm,
                                                                        method=method,
                                                                        use_inverse=use_inverse,
                                                                        dct_params=dct_params,
                                                                        store_intermediate=store_intermediate,
                                                                        path_save_model=path)
                df_all = df_result
                
                df_no = df_anomalies[df_anomalies['predictions'] == 1]
                df_no = df_no.drop_duplicates()
                print(
                    "Max different values (outliers) : {0} | Rules extracted {1}".format(
                        len(df_no), len(df_all)))
                if len(path)>0:
                    print("Saving rules...")
                    df_all.to_csv(path + '/df_rules_' + file_name + '.csv', index=False)
                    df_anomalies.to_csv(path + '/df_anomalies_' + file_name + '.csv', index=False)
            
        # If kprototypes, do not consider "categorical cols" for the purpose of the rest of the code
        if cluster_algorithm == "kprototypes":
            feature_cols = list(set(numerical_cols + categorical_cols))
            cat_additional = []
        else:
            feature_cols = numerical_cols
            cat_additional = categorical_cols
        
        # Complete Rules
        print("Checking outliers inside hypercubes...") 
        df_anomalies['predictions'] = df_anomalies['predictions']*-1
        df_anomalies['distances'] = df_anomalies['distances']*-1
        
        try:
            df_rules = pd.read_csv("{path}/{file_name}_rules_{type_r}_pruned_ocsvm.csv".format(path=path,
                                                                                               file_name=file_name,
                                                                                               type_r = "outliers"))
        except:
            df_rules = ocsvm_rules_completion(df_anomalies,
                                              df_all,
                                              feature_cols, 
                                              cat_additional,
                                              inliers_used=False,
                                              clustering_algorithm=cluster_algorithm,
                                              path=path,
                                              file_name=file_name)
        
        # Use only pure rules
        df_rules_outliers = df_rules 
        df_rules = df_rules[df_rules["n_inliers_included"]==0]
        
        if metrics:
            print("Obtaining metrics...")
            df_rules = rule_overlapping_score(df_rules, df_anomalies,
                                              feature_cols, cat_additional)
            
            df_rules = check_stability(df_anomalies, df_rules, clf,
                                        feature_cols, cat_additional,
                                        using_inliers=False)
            
        # Saving rules obtained
        if len(path)>0:
            print("Saving rules...")
            df_rules.to_csv(path + '/df_rules_complete_' + file_name + '.csv', index=False)
        
        if plot_fig:
            #### Plot Rules [Outliers]
            print("Plotting rules for outliers...")
            df_rules = df_rules.copy()
            df_rules = df_rules.drop_duplicates().reset_index(drop=True)
            plot_2D(df_rules,
                    df_anomalies,
                    folder = path,
                    path_name = file_name)
    
        df_rules_outliers_p1 = df_rules 
    
    return (df_rules_inliers, df_rules_outliers,
            df_rules_inliers_p1, df_rules_outliers_p1)




