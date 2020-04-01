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
        
        # Complete Rules
        print("Checking inliers inside hypercubes...")
        try:
            df_rules = pd.read_csv("{path}/{file_name}_rules_{type_r}_pruned_ocsvm.csv".format(path=path_folder,
                                                                                file_name=file_name,
                                                                                type_r = "inliers"))
        except:
            df_rules = ocsvm_rules_completion(df_anomalies,
                                              df_all,
                                              feature_cols, 
                                              cat_additional,
                                              inliers_used=True,
                                              clustering_algorithm=cluster_algorithm,
                                              path=path_folder,
                                              file_name=file_name)
        
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
        
        try:
            df_rules = pd.read_csv("{path}/{file_name}_rules_{type_r}_pruned_ocsvm.csv".format(path=path_folder,
                                                                                file_name=file_name,
                                                                                type_r = "outliers"))
        except:
            df_rules = ocsvm_rules_completion(df_anomalies,
                                              df_all,
                                              feature_cols, 
                                              cat_additional,
                                              inliers_used=False,
                                              clustering_algorithm=cluster_algorithm,
                                              path=path_folder,
                                              file_name=file_name)
            
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



