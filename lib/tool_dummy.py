# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:43:05 2019

@author: alber
"""

import pandas as pd
import numpy as np
import pickle 

from lib.unsupervised_rules import ocsvm_rule_extractor
from lib.tools import (dt_rules, turn_rules_to_df, plot_2D, anchors_rules,
                       rulefit_rules, skoperules_rules, surrogate_dt_rules,
                       ocsvm_rules_completion, file_naming_ocsvm,
                       rule_overlapping_score, check_stability)

import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ruleset
import arff
import time
import pickle

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
from sklearn.preprocessing import StandardScaler
from aix360.algorithms.protodash import ProtodashExplainer
from interpret.glassbox import DecisionListClassifier

N_JOBS = 4

def check_datapoint_inside_only(data_point, df_rules, numerical_cols,
                                categorical_cols, check_opposite=True):
    """
    1 for the hypercubes where it's inside, 0 for when not. It checks differently
    whether its for scenarios where the rules are independent according to the 
    different combination of categorical variables or whether everything is analyzed 
    alltogether. 
        
    Parameters
    ----------
    data_point : TYPE
        DESCRIPTION.
    df_rules : TYPE
        DESCRIPTION.
    numerical_cols : TYPE
        DESCRIPTION.
    categorical_cols : TYPE
        DESCRIPTION.
    check_opposite : TYPE
        It indicates whether to consider datapoints with >=/<= or strict >/<. 
        Since we will see the rules in a counterfactual way (p.e what should
        happen for an outlier to be an inlier) we consider the datapoints of the
        target rules with >=/<=, and the ones from the other class as >/< (that
        means that we consider rules with P=1 even if they have points from the
        other class on the edges) [NOT USED]
    Returns
    -------
    df_plot : TYPE
        DESCRIPTION.

    """
    df_plot = df_rules.copy()
    
    if len(df_rules)==0:
        df_plot['check'] = 0
        return df_plot
    
    # Default value
    df_plot['check'] = 1
    
    # Check for categorical
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            value = data_point[col]
            df_plot['check'] = df_plot['check']*(df_plot.apply(lambda x: 1 if (x[col] == value) else 0, axis=1))
    # Check for numerical
    if len(numerical_cols) > 0:
        for col in numerical_cols:
            value = data_point[col]
            if check_opposite:
                df_plot['check'] = df_plot['check']*(df_plot.apply(lambda x: 1 if ((x[col + '_max'] >= value) & (value >= x[col + '_min'])) else 0, axis=1))
            else:
                df_plot['check'] = df_plot['check']*(df_plot.apply(lambda x: 1 if ((x[col + '_max'] > value) & (value > x[col + '_min'])) else 0, axis=1))

    return df_plot[['check']]

def check_datapoint_inside(data_point, df_rules, numerical_cols,
                           categorical_cols, check_opposite=True):
    """
    1 for the hypercubes where it's inside, 0 for when not. It checks differently
    whether its for scenarios where the rules are independent according to the 
    different combination of categorical variables or whether everything is analyzed 
    alltogether. 
        
    Parameters
    ----------
    data_point : TYPE
        DESCRIPTION.
    df_rules : TYPE
        DESCRIPTION.
    numerical_cols : TYPE
        DESCRIPTION.
    categorical_cols : TYPE
        DESCRIPTION.
    check_opposite : TYPE
        It indicates whether to consider datapoints with >=/<= or strict >/<. 
        Since we will see the rules in a counterfactual way (p.e what should
        happen for an outlier to be an inlier) we consider the datapoints of the
        target rules with >=/<=, and the ones from the other class as >/< (that
        means that we consider rules with P=1 even if they have points from the
        other class on the edges) [NOT USED]
    Returns
    -------
    df_plot : TYPE
        DESCRIPTION.

    """
    df_plot = df_rules.copy()
    
    if len(df_rules)==0:
        df_plot['check'] = 0
        return df_plot
    
    # Default value
    df_plot['check'] = 1
    
    # Check for categorical
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            value = data_point[col]
            df_plot['check'] = df_plot['check']*(df_plot.apply(lambda x: 1 if (x[col] == value) else 0, axis=1))
    # Check for numerical
    if len(numerical_cols) > 0:
        for col in numerical_cols:
            value = data_point[col]
            if check_opposite:
                df_plot['check'] = df_plot['check']*(df_plot.apply(lambda x: 1 if ((x[col + '_max'] >= value) & (value >= x[col + '_min'])) else 0, axis=1))
            else:
                df_plot['check'] = df_plot['check']*(df_plot.apply(lambda x: 1 if ((x[col + '_max'] > value) & (value > x[col + '_min'])) else 0, axis=1))

    return df_plot


def ocsvm_rules_experiments_pipeline(df_mat, numerical_cols, categorical_cols,
                                     cluster_algorithm, method, rules_used,
                                     dct_params, path_folder, file_template,
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
                                                                    path_save_model=path_folder)
            df_all = df_result
            
            df_no = df_anomalies[df_anomalies['predictions'] == 1]
            df_no = df_no.drop_duplicates()
            print(
                "Max different values (inliers) : {0} | Rules extracted {1}".format(
                    len(df_no), len(df_all)))
            print("Saving rules...")
            df_all.to_csv(path_folder + '/df_rules_' + file_name + '.csv', index=False)
            df_anomalies.to_csv(path_folder + '/df_anomalies_' + file_name + '.csv', index=False)
            
        else:
            try:
                df_all = pd.read_csv(path_folder + '/df_rules_' + file_name + '.csv')
                df_anomalies = pd.read_csv(path_folder + '/df_anomalies_' + file_name + '.csv')
                clf = pickle.load(open("{0}/backup.p".format(path_folder), "rb"))
                sc = pickle.load(open("{0}/sc.p".format(path_folder), "rb"))
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
                                                                        path_save_model=path_folder)
                df_all = df_result
                
                df_no = df_anomalies[df_anomalies['predictions'] == 1]
                df_no = df_no.drop_duplicates()
                print(
                    "Max different values (inliers) : {0} | Rules extracted {1}".format(
                        len(df_no), len(df_all)))
                print("Saving rules...")
                df_all.to_csv(path_folder + '/df_rules_' + file_name + '.csv', index=False)
                df_anomalies.to_csv(path_folder + '/df_anomalies_' + file_name + '.csv', index=False)
            
        
        # If kprototypes, do not consider "categorical cols" for the purpose of the rest of the code
        if cluster_algorithm == "kprototypes":
            feature_cols = list(set(numerical_cols + categorical_cols))
            cat_additional = []
        else:
            feature_cols = numerical_cols
            cat_additional = categorical_cols
                
        df_anomalies = df_anomalies
        df_rules = df_all
        inliers_used=True
        clustering_algorithm=cluster_algorithm
        path=path_folder
        file_name=file_name
        
        df_rules['n_inliers_included'] = 0
        df_rules['n_outliers_included'] = 0
        n_inliers = len(df_anomalies[df_anomalies['predictions']==1])
        n_outliers = len(df_anomalies[df_anomalies['predictions']==-1])
        n_vertex = (len(cat_additional) + 1)*2**(len(feature_cols))
        
        print("Checking inliers inside rules...")
        df_check = Parallel(n_jobs=N_JOBS)(delayed(check_datapoint_inside_only)(data_point,df_rules,feature_cols,cat_additional) for i, data_point in df_anomalies[df_anomalies['predictions']==1].iterrows())
        df_check = pd.concat([x[x['check']>0] for x in df_check])
        df_check = pd.DataFrame(df_check.groupby(df_check.index).sum()).reset_index()
        df_temp =  df_rules[['n_inliers_included']].reset_index()
        df_check = df_temp.merge(df_check, how="outer")[['check']].fillna(0)
        df_rules['n_inliers_included'] = df_check
        
        print("Checking outliers inside rules...")
        df_check = Parallel(n_jobs=N_JOBS)(delayed(check_datapoint_inside_only)(data_point,df_rules,feature_cols,cat_additional) for i, data_point in df_anomalies[df_anomalies['predictions']==-1].iterrows())
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
                                                       feature_cols,
                                                       cat_additional)['check']
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
                                                       feature_cols,
                                                       cat_additional)['check']
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
        df_rules.to_csv("{path}/{file_name}_rules_{type_r}_pruned_ocsvm.csv".format(path=path,
                                                                                    file_name=file_name,
                                                                                    type_r = path_aux),
                                    index=False)
        
        # Use only pure rules
        df_rules = df_rules[df_rules["n_outliers_included"]==0]
        
        print("Obtaining metrics...")
        df_rules = rule_overlapping_score(df_rules, df_anomalies,
                                          feature_cols, cat_additional)
        
        df_rules = check_stability(df_anomalies, df_rules, clf,
                                    feature_cols, cat_additional,
                                    using_inliers=True)
        
        # Saving rules obtained
        print("Saving rules...")
        df_rules.to_csv(path_folder + '/df_rules_complete_' + file_name + '.csv', index=False)
    
        if plot_fig:
            #### Plot Rules [Inliers]
            print("Plotting rules for inliers...")
            df_rules = df_rules.copy()
            df_rules = df_rules.drop_duplicates().reset_index(drop=True)
            plot_2D(df_rules,
                    df_anomalies,
                    folder = path_folder,
                    path_name=file_name)
    
    
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
                                                                    path_save_model=path_folder)
            df_all = df_result
            
            df_no = df_anomalies[df_anomalies['predictions'] == 1]
            df_no = df_no.drop_duplicates()
            print(
                "Max different values (outliers) : {0} | Rules extracted {1}".format(
                    len(df_no), len(df_all)))
            print("Saving rules...")
            df_all.to_csv(path_folder + '/df_rules_' + file_name + '.csv', index=False)
            df_anomalies.to_csv(path_folder + '/df_anomalies_' + file_name + '.csv', index=False)
            
        else:
            try:
                df_all = pd.read_csv(path_folder + '/df_rules_' + file_name + '.csv')
                df_anomalies = pd.read_csv(path_folder + '/df_anomalies_' + file_name + '.csv')
                clf = pickle.load(open("{0}/backup.p".format(path_folder), "rb"))
                sc = pickle.load(open("{0}/sc.p".format(path_folder), "rb"))
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
                                                                        path_save_model=path_folder)
                df_all = df_result
                
                df_no = df_anomalies[df_anomalies['predictions'] == 1]
                df_no = df_no.drop_duplicates()
                print(
                    "Max different values (outliers) : {0} | Rules extracted {1}".format(
                        len(df_no), len(df_all)))
                print("Saving rules...")
                df_all.to_csv(path_folder + '/df_rules_' + file_name + '.csv', index=False)
                df_anomalies.to_csv(path_folder + '/df_anomalies_' + file_name + '.csv', index=False)
        
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
        
        df_anomalies = df_anomalies
        df_rules = df_all
        inliers_used=False
        clustering_algorithm=cluster_algorithm
        path=path_folder
        file_name=file_name
        
        df_rules['n_inliers_included'] = 0
        df_rules['n_outliers_included'] = 0
        n_inliers = len(df_anomalies[df_anomalies['predictions']==1])
        n_outliers = len(df_anomalies[df_anomalies['predictions']==-1])
        n_vertex = (len(cat_additional) + 1)*2**(len(feature_cols))
        
        print("Checking inliers inside rules...")
        df_check = Parallel(n_jobs=N_JOBS)(delayed(check_datapoint_inside_only)(data_point,df_rules,feature_cols,cat_additional) for i, data_point in df_anomalies[df_anomalies['predictions']==1].iterrows())
        df_check = pd.concat([x[x['check']>0] for x in df_check])
        df_check = pd.DataFrame(df_check.groupby(df_check.index).sum()).reset_index()
        df_temp =  df_rules[['n_inliers_included']].reset_index()
        df_check = df_temp.merge(df_check, how="outer")[['check']].fillna(0)
        df_rules['n_inliers_included'] = df_check
        
        print("Checking outliers inside rules...")
        df_check = Parallel(n_jobs=N_JOBS)(delayed(check_datapoint_inside_only)(data_point,df_rules,feature_cols,cat_additional) for i, data_point in df_anomalies[df_anomalies['predictions']==-1].iterrows())
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
                                                       feature_cols,
                                                       cat_additional)['check']
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
                                                       feature_cols,
                                                       cat_additional)['check']
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
        df_rules.to_csv("{path}/{file_name}_rules_{type_r}_pruned_ocsvm.csv".format(path=path,
                                                                                    file_name=file_name,
                                                                                    type_r = path_aux),
                                    index=False)
        
            
        df_rules = df_rules[df_rules["n_inliers_included"]==0]
        
        print("Obtaining metrics...")
        df_rules = rule_overlapping_score(df_rules, df_anomalies,
                                          feature_cols, cat_additional)
        
        df_rules = check_stability(df_anomalies, df_rules, clf,
                                    feature_cols, cat_additional,
                                    using_inliers=False)
        
        # Saving rules obtained
        print("Saving rules...")
        df_rules.to_csv(path_folder + '/df_rules_complete_' + file_name + '.csv', index=False)
        
        if plot_fig:
            #### Plot Rules [Outliers]
            print("Plotting rules for outliers...")
            df_rules = df_rules.copy()
            df_rules = df_rules.drop_duplicates().reset_index(drop=True)
            plot_2D(df_rules,
                    df_anomalies,
                    folder = path_folder,
                    path_name = file_name)
            
    else:
        raise ValueError("Argument {0} not found -- use ['all', 'outliers' or 'inliers'] instead".format(rules_used) )



