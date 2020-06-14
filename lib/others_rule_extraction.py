# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 08:44:31 2019

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
                    check_datapoint_inside, check_datapoint_inside_only, turn_rules_to_df)
from lib.xai_metrics import (rule_overlapping_score, check_stability)
    
def surrogate_dt_rules(df_anomalies, model, numerical_cols,
                       categorical_cols, metrics=True, path="",
                       file_name="default_name"):
    """
    TODO

    Parameters
    ----------
    df_anomalies : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    numerical_cols : TYPE
        DESCRIPTION.
    categorical_cols : TYPE
        DESCRIPTION.
    metrics : TYPE, optional
        DESCRIPTION. The default is True.
    path : TYPE, optional
        DESCRIPTION. The default is "".
    file_name : TYPE, optional
        DESCRIPTION. The default is "default_name".

    Returns
    -------
    df_rules_inliers : TYPE
        DESCRIPTION.
    df_rules_outliers : TYPE
        DESCRIPTION.
    df_rules_inliers_p1 : TYPE
        DESCRIPTION.
    df_rules_outliers_p1 : TYPE
        DESCRIPTION.

    """
    
    def dt_rules(clf, df_mat):
        """
        Function to transform the printed structure of a DT into the set of rules
        derived from the paths to the terminal nodes.
        It also includes the length of each of those rules,
        as well as the prediction associated with it (value of that terminal node).
    
        Parameters
        ----------
        clf : TYPE
            DESCRIPTION.
        df_mat : TYPE
            DESCRIPTION.
    
        Returns
        -------
        df_rules : TYPE
            DESCRIPTION.
    
        """
        r = export_text(clf, feature_names=list(df_mat.columns))
        
        list_splits = r.split("|---")
        list_splits = [x.replace("|", "") for x in list_splits]
        list_splits = [x.replace("class: -1", "") for x in list_splits]
        list_splits = [x.replace("class: 1", "") for x in list_splits]
        list_splits = [x.strip() for x in list_splits]
        df_splits = pd.DataFrame({"levels":list_splits})
        df_splits = df_splits[df_splits['levels'] != ""].reset_index(drop=True).reset_index()
        df_splits['index'] += 1
        
        
        df_rules = pd.DataFrame()
        
        for i, point in df_mat.iterrows():
            node_indices = clf.decision_path(point.values.reshape(1, -1))
            rule = ""
            node_indices = pd.DataFrame(node_indices.toarray().T).reset_index()
            node_indices = node_indices.merge(df_splits)
            node_indices = node_indices[node_indices[0] == 1]
            for i in list(node_indices['levels']):
                if rule == "":
                    rule = i
                else:
                    rule = rule + " & " + i 
            dct_aux = {'rule':rule,
                       'prediction':clf.predict(point.values.reshape(1, -1)),
                       'len_rule':len(node_indices)}
            
            df_rules = df_rules.append(pd.DataFrame(dct_aux, index=[0]))
        
        df_rules = df_rules.drop_duplicates()
        
        return df_rules


    # Init
    y_real = df_anomalies['predictions']
    feature_cols = numerical_cols + categorical_cols
    df_mat = df_anomalies[feature_cols]
    
    # Fit model (overfitted)
    clf = tree.DecisionTreeClassifier(random_state=42)
    clf = clf.fit(df_anomalies[feature_cols], y_real)
    rules_tree = clf.tree_.value
    print("Depth tree: ", clf.tree_.max_depth)
    print("Nodes tree: ", clf.tree_.node_count)
    
    leaf_nodes = clf.tree_.value
    leaf_nodes_anomalies = [
        clf.classes_[np.argmax(x)] for x in leaf_nodes
        if clf.classes_[np.argmax(x)] == 1
    ]
    print("leaf nodes not anomalies: ", len(leaf_nodes_anomalies))
    
    # Obtain Rules
    r = export_text(clf, feature_names=list(df_mat.columns))
    r_s = r.split("class: ")
    leaf_yes = [x for x in r_s if x[0:2] == "-1"]
    leaf_no = [x for x in r_s if x[0] == "1"]
    print("tree rules not anomalies: ", len(leaf_yes))
    print("tree rules anomalies: ", len(leaf_no))
    
    df_rules_original = dt_rules(clf, df_mat[feature_cols])
    df_rules_original = df_rules_original.rename(columns={'len_rule':'size_rules'})
    
    ### Inliers
    list_rules_no = list(df_rules_original[df_rules_original['prediction']==1]['rule'])
    df_rules_inliers = turn_rules_to_df(df_anomalies, list_rules_no, feature_cols)
    df_rules_inliers['size_rules'] = df_rules_original[df_rules_original['prediction']==1]['size_rules']
    
    ### Outliers
    list_rules_yes = list(df_rules_original[df_rules_original['prediction']==-1]['rule'])
    df_rules_outliers = turn_rules_to_df(df_anomalies, list_rules_yes, feature_cols)
    df_rules_outliers['size_rules'] = df_rules_original[df_rules_original['prediction']==-1]['size_rules']
    
    ### All these rules have a P=1; Only need to check how many datapoints are included in them
    n_inliers = len(df_anomalies[df_anomalies['predictions']==1])
    n_outliers = len(df_anomalies[df_anomalies['predictions']==-1])
    
    print("Checking inliers inside rules for inliers/outliers...")
    df_rules_inliers['n_inliers_included'] = 0
    df_rules_outliers['n_inliers_included'] = 0
    for i, data_point in df_anomalies[df_anomalies['predictions']==1].iterrows():
        df_rules_inliers['n_inliers_included'] += check_datapoint_inside(data_point,
                                                                         df_rules_inliers,
                                                                         feature_cols,
                                                                         [])['check']
        
        df_rules_outliers['n_inliers_included'] += check_datapoint_inside(data_point,
                                                                          df_rules_outliers,
                                                                          feature_cols,
                                                                          [])['check']
        
    print("Checking outliers inside rules for inliers/outliers...")
    df_rules_inliers['n_outliers_included'] = 0
    df_rules_outliers['n_outliers_included'] = 0
    for i, data_point in df_anomalies[df_anomalies['predictions']==-1].iterrows():
        df_rules_inliers['n_outliers_included'] += check_datapoint_inside(data_point,
                                                                          df_rules_inliers,
                                                                          feature_cols,
                                                                          [])['check']
        
        df_rules_outliers['n_outliers_included'] += check_datapoint_inside(data_point,
                                                                           df_rules_outliers,
                                                                           feature_cols,
                                                                           [])['check']
    
    # Check how many datapoints are included with the rules with Precision=1
    print("Checking inliers/outliers inside hypercubes with Precision=1...")
    n_inliers_p1 = 0
    n_inliers_p0 = 0
    n_outliers_p1 = 0
    n_outliers_p0 = 0
    n_inliers = len(df_anomalies[df_anomalies['predictions']==1])
    n_outliers = len(df_anomalies[df_anomalies['predictions']==-1])
        
    for i, data_point in df_anomalies.iterrows():
        df_rules_inliers['check'] = check_datapoint_inside(data_point,
                                                           df_rules_inliers,
                                                           feature_cols,
                                                           [])['check']
        
        df_rules_outliers['check'] = check_datapoint_inside(data_point,
                                                           df_rules_outliers,
                                                           feature_cols,
                                                           [])['check']
        # If inlier
        if data_point['predictions']==1:
            # Rules with any P and that include this datapoint
            df_aux = df_rules_inliers[(df_rules_inliers['check']==1)] 
            if len(df_aux) > 0:
                n_inliers_p0 += 1
            
            # Rules with P=1 and that include this datapoint
            df_aux = df_rules_inliers[(df_rules_inliers['n_outliers_included']==0)
                                       & (df_rules_inliers['check']==1)] 
            if len(df_aux) > 0:
                n_inliers_p1 += 1
            
        # If outlier
        elif data_point['predictions']==-1:
            # Rules with any P and that include this datapoint
            df_aux = df_rules_outliers[(df_rules_outliers['check']==1)] 
            if len(df_aux) > 0:
                n_outliers_p0 += 1
            
            # Rules with P=1 and that include this datapoint
            df_aux = df_rules_outliers[(df_rules_outliers['n_inliers_included']==0)
                                       & (df_rules_outliers['check']==1)] 
            if len(df_aux) > 0:
                n_outliers_p1 += 1
    
    df_rules_inliers['n_inliers'] = n_inliers
    df_rules_inliers['n_inliers_p0'] = n_inliers_p0
    df_rules_inliers['n_inliers_p1'] = n_inliers_p1
    df_rules_outliers['n_outliers_p1'] = n_outliers_p1
    df_rules_outliers['n_outliers_p0'] = n_outliers_p0
    df_rules_outliers['n_outliers'] = n_outliers
    del df_rules_inliers['check'], df_rules_outliers['check']
       
    # Save to CSV
    if len(path)>0:
        print("Saving results (all rules)...")
        df_rules_outliers.to_csv("{path}/{file_name}_rules_outliers_DT.csv".format(path=path, file_name=file_name), index=False)
        df_rules_inliers.to_csv("{path}/{file_name}_rules_inliers_DT.csv".format(path=path, file_name=file_name), index=False)
          
    # Prune rules
    df_yes_pruned = df_rules_outliers[(df_rules_outliers['n_inliers_included'] == 0) &
                                      (df_rules_outliers['n_outliers_included'] > 0)]
    df_yes_pruned = df_yes_pruned.reset_index(drop=True)
    
    if len(df_yes_pruned) > 1:
        df_yes_pruned = simplify_rules_alt([], df_yes_pruned).drop_duplicates()
        
    df_no_pruned = df_rules_inliers[(df_rules_inliers['n_inliers_included'] > 0) &
                                    (df_rules_inliers['n_outliers_included'] == 0)]
    df_no_pruned = df_no_pruned.reset_index(drop=True)
    
    if len(df_no_pruned) > 1:
        df_no_pruned = simplify_rules_alt([], df_no_pruned).drop_duplicates()
        
    # Obtain additional metrics
    if metrics:
        print("Obtaining additional metrics...")
        
        ### Overlapping
        coeff = 1000
        df_rules = df_rules_inliers.append(df_rules_outliers)
        df_dummy = df_rules.copy().replace(np.inf, 0).replace(-np.inf, 0) # discard infinities
        max_dummy = df_dummy[[col for col in list(df_rules.columns) if '_max' in col]].max()
        max_dummy = [x for x in list(max_dummy.values) if x != np.inf][0]*coeff # arbitrary large value
        min_dummy = df_dummy[[col for col in list(df_rules.columns) if '_min' in col]].min() # arbitrary low value
        min_dummy = [x for x in list(min_dummy.values) if x != -np.inf][0]
        min_dummy = min_dummy*coeff if min_dummy < 0 else min_dummy/coeff
        
        df_yes_pruned = df_yes_pruned.drop_duplicates().reset_index(drop=True)
        df_yes_pruned = df_yes_pruned.replace(np.inf, max_dummy)
        df_yes_pruned = df_yes_pruned.replace(-np.inf, min_dummy)
        df_yes_pruned = rule_overlapping_score(df_yes_pruned, df_anomalies,
                                               feature_cols,
                                               [])
        
        df_no_pruned = df_no_pruned.drop_duplicates().reset_index(drop=True)
        df_no_pruned = df_no_pruned.replace(np.inf, max_dummy)
        df_no_pruned = df_no_pruned.replace(-np.inf, min_dummy)
        df_no_pruned = rule_overlapping_score(df_no_pruned, df_anomalies,
                                               feature_cols,
                                               [])
        ### Stability
        df_yes_pruned = check_stability(df_anomalies, df_yes_pruned, model,
                                        feature_cols, [],
                                        using_inliers = False)
        df_no_pruned = check_stability(df_anomalies, df_no_pruned, model,
                                        feature_cols, [],
                                        using_inliers = True)
        
        # Replace with original limits
        df_yes_pruned = df_yes_pruned.replace(max_dummy, np.inf)
        df_yes_pruned = df_yes_pruned.replace(min_dummy, -np.inf)
        df_no_pruned = df_no_pruned.replace(max_dummy, np.inf)
        df_no_pruned = df_no_pruned.replace(min_dummy, -np.inf)
    
    # Save to CSV
    if len(path)>0:
        print("Saving results (P@1 rules)...")
        df_yes_pruned.to_csv("{path}/{file_name}_rules_outliers_pruned_DT.csv".format(path=path, file_name=file_name),
                                    index=False)
        df_no_pruned.to_csv("{path}/{file_name}_rules_inliers_pruned_DT.csv".format(path=path, file_name=file_name),
                                   index=False)
    
    df_rules_inliers_p1 = df_no_pruned
    df_rules_outliers_p1 = df_yes_pruned
    
    return (df_rules_inliers, df_rules_outliers,
            df_rules_inliers_p1, df_rules_outliers_p1)
    
    
def anchors_rules(df_anomalies, numerical_cols, categorical_cols,
                  model, scaler, metrics=True, path="",  file_name="default_name"):
    """
    TODO

    Parameters
    ----------
    df_anomalies : TYPE
        DESCRIPTION.
    numerical_cols : TYPE
        DESCRIPTION.
    categorical_cols : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    scaler : TYPE
        DESCRIPTION.
    metrics : TYPE, optional
        DESCRIPTION. The default is True.
    path : TYPE, optional
        DESCRIPTION. The default is "".
    file_name : TYPE, optional
        DESCRIPTION. The default is "default_name".

    Returns
    -------
    TYPE
        DESCRIPTION.

    """


    
    def _transform_rules(list_rules, list_cols):
        """
        Similar to turn_rules_to_df() but considering feature unscaling.

        Parameters
        ----------
        list_rules : TYPE
            DESCRIPTION.
        list_cols : TYPE
            DESCRIPTION.

        Returns
        -------
        df_rules_transformed : TYPE
            DESCRIPTION.
        list_rules_transformed : TYPE
            DESCRIPTION.

        """
        dct_features_pos = {}
        i = 0
        for col in list_cols:
            dct_features_pos[col] = i
            i += 1
        
        df_rules_transformed = pd.DataFrame()
        list_rules_transformed = []
               
        # Iter for each rule
        for rule in list_rules:
            dct_aux = {}
            list_iter = []
            # Default values
            for col in list_cols:
                dct_aux[col + '_max'] = np.inf
                dct_aux[col + '_min'] = -np.inf
            
            list_subrules = rule
            
            # Iter for each component of the rule and obtain the limits
            for subrule in list_subrules:
                ref_col = ""
                list_symbols = [">=", "<=", ">", "<"]
                for symbol in list_symbols:
                    split_rule = subrule.split(symbol)
                    if len(split_rule)==2:
                        ref_col = split_rule[0]
                    elif len(split_rule)==3:
                        ref_col = split_rule[1]
                
                ref_col = ref_col.lstrip().rstrip()
                col = ref_col
                str_aux = ""
                
                # Check if structure is XXX < Z < XXXX
                aux = subrule.split("<")
                aux = [x.strip("=") for x in aux]
                if len(aux) > 2:
                    # Lower limit
                    if col in numerical_cols:
                        vect_aux = [0]*len(numerical_cols)
                        vect_aux[dct_features_pos[col]] = float(aux[0])
                        vect_aux = sc.inverse_transform(vect_aux)
                        str_aux += str(np.round(vect_aux[dct_features_pos[col]], 2)) + ' <'
                        if np.float(vect_aux[dct_features_pos[col]]) >= dct_aux[col + '_min']:
                            dct_aux[col + '_min'] = np.float(vect_aux[dct_features_pos[col]])
                    else:
                        str_aux += str(np.round(np.float(aux[0]), 2)) + ' <'
                        if np.float(aux[0]) >= dct_aux[col + '_min']:
                            dct_aux[col + '_min'] = np.float(aux[0])

                    str_aux += aux[1]
                    
                    # Upper limit
                    if col in numerical_cols:
                        vect_aux[dct_features_pos[col]] = float(aux[2])
                        vect_aux = sc.inverse_transform(vect_aux)
                        str_aux += '< ' + str(np.round(vect_aux[dct_features_pos[col]], 2))
                        if np.float(vect_aux[dct_features_pos[col]]) <= dct_aux[col + '_max']:
                            dct_aux[col + '_max'] = np.float(vect_aux[dct_features_pos[col]])
                    else:
                        str_aux += '< ' + str(np.round(np.float(aux[2]), 2))
                        if np.float(aux[2]) <= dct_aux[col + '_max']:
                            dct_aux[col + '_max'] = np.float(aux[2])
                        
                # General structure
                else:
                    if ">=" in subrule:
                        aux = subrule.split(">=")
                        
                        if col in numerical_cols:
                            vect_aux = [0]*len(numerical_cols)
                            vect_aux[dct_features_pos[col]] = float(aux[1])
                            vect_aux = sc.inverse_transform(vect_aux)
                            str_aux += aux[0] + '>= ' + str(np.round(vect_aux[dct_features_pos[col]], 2))
                            
                            if np.float(vect_aux[dct_features_pos[col]]) >= dct_aux[col + '_min']:
                                dct_aux[col + '_min'] = np.float(vect_aux[dct_features_pos[col]])
                            
                        else:
                            str_aux += aux[0] + '> ' + str(np.round(float(aux[1]), 2))
                            
                            if np.float(aux[1]) >= dct_aux[col + '_min']:
                                dct_aux[col + '_min'] = np.float(aux[1])
                        
 
                    elif ">" in subrule:
                        aux = subrule.split(">")
                        
                        if col in numerical_cols:
                            vect_aux = [0]*len(numerical_cols)
                            vect_aux[dct_features_pos[col]] = float(aux[1])
                            vect_aux = sc.inverse_transform(vect_aux)
                            str_aux += aux[0] + '> ' + str(np.round(vect_aux[dct_features_pos[col]], 2))
                            
                            if np.float(vect_aux[dct_features_pos[col]]) >= dct_aux[col + '_min']:
                                dct_aux[col + '_min'] = np.float(vect_aux[dct_features_pos[col]])
                            
                        else:
                            str_aux += aux[0] + '> ' + str(np.round(float(aux[1]), 2))
                        
                            if np.float(aux[1]) >= dct_aux[col + '_min']:
                                dct_aux[col + '_min'] = np.float(aux[1])
                        

                    if "<=" in subrule:
                        aux = subrule.split("<=")
                        
                        if col in numerical_cols:
                            vect_aux = [0]*len(numerical_cols)
                            vect_aux[dct_features_pos[col]] = float(aux[1])
                            vect_aux = sc.inverse_transform(vect_aux)
                            str_aux += aux[0] + '<= ' + str(np.round(vect_aux[dct_features_pos[col]], 2))
                            
                            if np.float(vect_aux[dct_features_pos[col]]) <= dct_aux[col + '_max']:
                                dct_aux[col + '_max'] = np.float(vect_aux[dct_features_pos[col]])
                        else:
                            str_aux += aux[0] + '<= ' + str(np.round(np.float(aux[1]), 2))
                            
                            if np.float(aux[1]) <= dct_aux[col + '_max']:
                                dct_aux[col + '_max'] = np.float(aux[1])
                        
 
                    elif "<" in subrule:
                        aux = subrule.split("<")
                        
                        if col in numerical_cols:
                            vect_aux = [0]*len(numerical_cols)
                            vect_aux[dct_features_pos[col]] = float(aux[1])
                            vect_aux = sc.inverse_transform(vect_aux)
                            str_aux += aux[0] + '< ' + str(np.round(vect_aux[dct_features_pos[col]], 2))
                        
                            if np.float(vect_aux[dct_features_pos[col]]) <= dct_aux[col + '_max']:
                                dct_aux[col + '_max'] = np.float(vect_aux[dct_features_pos[col]])
                        else:
                            vect_aux = [0]*len(numerical_cols)
                            vect_aux[dct_features_pos[col]] = float(aux[1])
                            vect_aux = sc.inverse_transform(vect_aux)
                            str_aux += aux[0] + '< ' + str(np.round(np.float(aux[1]), 2))
                        
                            if np.float(aux[1]) <= dct_aux[col + '_max']:
                                dct_aux[col + '_max'] = np.float(aux[1])
            
                if str_aux != "":
                    list_iter.append(str_aux)
                        
            list_rules_transformed.append(list_iter)

            df_rules_transformed = df_rules_transformed.append(pd.DataFrame(dct_aux, index=[0]))
            
        return df_rules_transformed, list_rules_transformed
    
    
    df_scaled = df_anomalies.copy()
    clf = model
    sc = scaler
    feature_cols = numerical_cols + categorical_cols
    
    if sc != None:
        df_scaled[numerical_cols] = sc.transform(df_scaled[numerical_cols])
        data = df_scaled[feature_cols].values
        labels = df_scaled["predictions"].values
        
    # Fit Anchors model
    predict_fn = lambda x: clf.predict(x)
    explainer = AnchorTabular(predict_fn, feature_cols)
    explainer.fit(data)
    
    # Obtain rules for each datapoint (anomalous)
    df_scaled_yes = df_scaled[df_scaled['predictions']==-1][feature_cols]
    
    if len(df_scaled_yes)>10000:
        n_samples = 1000
        explainer_proto = ProtodashExplainer()
        list_cols = numerical_cols + categorical_cols
        df_data = df_scaled_yes
        (W, S, _) = explainer_proto.explain(df_data[list_cols].values,
                                      df_data[list_cols].values,
                                      m=n_samples,
                                      kernelType='Gaussian',
                                      sigma=2)
        df_prototypes = df_scaled[df_scaled.index.isin(list(S))][list_cols].reset_index(drop=True)
        df_scaled_yes = df_prototypes
        
    list_rules_anchors_yes = [explainer.explain(j.values)['names'] 
                              for _,j in df_scaled_yes[feature_cols].iterrows()]
    
    if len(path)>0:
        print("Backing up files...")
        pickle.dump(list_rules_anchors_yes, open("{0}/list_rules_anchors_yes.p".format(path), "wb"))
    
    # Keep unique ones
    list_aux = []
    for col in list_rules_anchors_yes:
        if col not in list_aux:
            list_aux.append(col)
        else:
            pass
    list_rules_anchors_yes = list_aux
    df_rules_anchors_yes, list_rules_transformed_yes = _transform_rules(list_rules = list_rules_anchors_yes,
                                                                        list_cols = feature_cols)
    print("Calculating the size of the rules (outliers)...")
    list_aux = []
    df_aux = pd.DataFrame(list_rules_transformed_yes)
    for i, row in df_aux.iterrows():
        size_rule = 0
        for value in row.values:
            if value != None:
                if ">" in value: size_rule += 1
                if "<" in value: size_rule += 1
        list_aux.append(size_rule)
    df_rules_anchors_yes['size_rules'] = list_aux
    
    # Obtain rules (inliers)
    df_scaled_no = df_scaled[df_scaled['predictions']==1][feature_cols]
    if len(df_scaled_no)>10000:
        n_samples = 1000
        explainer_proto = ProtodashExplainer()
        list_cols = numerical_cols + categorical_cols
        df_data = df_scaled_no
        (W, S, _) = explainer_proto.explain(df_data[list_cols].values,
                                      df_data[list_cols].values,
                                      m=n_samples,
                                      kernelType='Gaussian',
                                      sigma=2)
        df_prototypes = df_scaled[df_scaled.index.isin(list(S))][list_cols].reset_index(drop=True)
        df_scaled_no = df_prototypes
    
    list_rules_anchors_no = [explainer.explain(j.values)['names'] 
                              for _,j in df_scaled_no[feature_cols].iterrows()]
    
    if len("path")>0:
        print("Backing up files...")
        pickle.dump(list_rules_anchors_no, open("{0}/list_rules_anchors_no.p".format(path), "wb"))
    
    list_aux = []
    for col in list_rules_anchors_no:
        if col not in list_aux:
            list_aux.append(col)
        else:
            pass
    list_rules_anchors_no = list_aux
    
    df_rules_anchors_no, list_rules_transformed_no = _transform_rules(list_rules = list_rules_anchors_no,
                                                                      list_cols = feature_cols)
    
    print("Calculating the size of the rules (inliers)...")
    list_aux = []
    df_aux = pd.DataFrame(list_rules_transformed_no)
    for i, row in df_aux.iterrows():
        size_rule = 0
        for value in row.values:
            if value != None:
                if ">" in value: size_rule += 1
                if "<" in value: size_rule += 1
        list_aux.append(size_rule)
    df_rules_anchors_no['size_rules'] = list_aux
    
    print("Checking inliers inside rules for inliers/outliers...")
    df_rules_anchors_no['n_inliers_included'] = 0
    df_rules_anchors_yes['n_inliers_included'] = 0
    for i, data_point in df_anomalies[df_anomalies['predictions']==1].iterrows():
        df_rules_anchors_no['n_inliers_included'] += check_datapoint_inside(data_point,
                                                                 df_rules_anchors_no,
                                                                 feature_cols,
                                                                 [])['check']
        
        df_rules_anchors_yes['n_inliers_included'] += check_datapoint_inside(data_point,
                                                                 df_rules_anchors_yes,
                                                                 feature_cols,
                                                                 [])['check']
        
    print("Checking outliers inside rules for inliers/outliers...")
    df_rules_anchors_no['n_outliers_included'] = 0
    df_rules_anchors_yes['n_outliers_included'] = 0
    for i, data_point in df_anomalies[df_anomalies['predictions']==-1].iterrows():
        df_rules_anchors_no['n_outliers_included'] += check_datapoint_inside(data_point,
                                                                 df_rules_anchors_no,
                                                                 feature_cols,
                                                                 [])['check']
        
        df_rules_anchors_yes['n_outliers_included'] += check_datapoint_inside(data_point,
                                                                 df_rules_anchors_yes,
                                                                 feature_cols,
                                                                 [])['check']
        
    # Check how many datapoints are included with the rules with Precision=1 and in general
    print("Checking inliers/outliers inside hypercubes with Precision=1...")
    n_inliers_p1 = 0
    n_inliers_p0 = 0
    n_outliers_p1 = 0
    n_outliers_p0 = 0
    n_inliers = len(df_anomalies[df_anomalies['predictions']==1])
    n_outliers = len(df_anomalies[df_anomalies['predictions']==-1])
    
    for i, data_point in df_anomalies.iterrows():
        df_rules_anchors_no['check'] = check_datapoint_inside(data_point,
                                                              df_rules_anchors_no,
                                                              feature_cols,
                                                              [])['check']
        df_rules_anchors_yes['check'] = check_datapoint_inside(data_point,
                                                               df_rules_anchors_yes,
                                                               feature_cols,
                                                               [])['check']
        # If inlier
        if data_point['predictions']==1:
            # Rules with any P and that include this datapoint
            df_aux = df_rules_anchors_no[(df_rules_anchors_no['check']==1)] 
            if len(df_aux) > 0:
                n_inliers_p0 += 1
            
            # Rules with P=1 and that include this datapoint
            df_aux = df_rules_anchors_no[(df_rules_anchors_no['n_outliers_included']==0) & (df_rules_anchors_no['check']==1)] 
            if len(df_aux) > 0:
                n_inliers_p1 += 1
            
        # If outlier
        elif data_point['predictions']==-1:
            # Rules with any P and that include this datapoint
            df_aux = df_rules_anchors_yes[(df_rules_anchors_yes['check']==1)] 
            if len(df_aux) > 0:
                n_outliers_p0 += 1
            
            # Rules with P=1 and that include this datapoint
            df_aux = df_rules_anchors_yes[(df_rules_anchors_yes['n_inliers_included']==0) & (df_rules_anchors_yes['check']==1)] 
            if len(df_aux) > 0:
                n_outliers_p1 += 1
    
    df_rules_anchors_no['n_inliers'] = n_inliers
    df_rules_anchors_no['n_inliers_p0'] = n_inliers_p0
    df_rules_anchors_no['n_inliers_p1'] = n_inliers_p1
    df_rules_anchors_yes['n_outliers'] = n_outliers
    df_rules_anchors_yes['n_outliers_p1'] = n_outliers_p1
    df_rules_anchors_yes['n_outliers_p0'] = n_outliers_p0
    
    del df_rules_anchors_yes['check'], df_rules_anchors_no['check']
    
    # Save to CSV
    if len(path)>0:
        print("Saving results (all rules)... ")
        df_rules_anchors_yes.to_csv("{path}/{file_name}_rules_outliers_Anchors.csv".format(path=path, file_name=file_name), index=False)
        df_rules_anchors_no.to_csv("{path}/{file_name}_rules_inliers_Anchors.csv".format(path=path, file_name=file_name), index=False)
        
    # Prune rules
    df_yes_pruned = df_rules_anchors_yes[(df_rules_anchors_yes['n_inliers_included'] == 0) &
                                         (df_rules_anchors_yes['n_outliers_included'] > 0)]
    df_yes_pruned = df_yes_pruned.reset_index(drop=True)
    
    if len(df_yes_pruned) > 1:
        df_yes_pruned = simplify_rules_alt([], df_yes_pruned).drop_duplicates()
        
    df_no_pruned = df_rules_anchors_no[(df_rules_anchors_no['n_inliers_included'] > 0) &
                                       (df_rules_anchors_no['n_outliers_included'] == 0)]
    df_no_pruned = df_no_pruned.reset_index(drop=True)
    
    if len(df_no_pruned) > 1:
        df_no_pruned = simplify_rules_alt([], df_no_pruned).drop_duplicates()
        
    # Obtain additional metrics
    if metrics:
        print("Obtaining additional metrics...")
        coeff = 1000
        
        ### Overlapping
        df_rules = df_no_pruned.append(df_yes_pruned)
        df_dummy = df_rules.copy().replace(np.inf, 0).replace(-np.inf, 0) # discard infinities
        max_dummy = df_dummy[[col for col in list(df_rules.columns) if '_max' in col]].max()
        max_dummy = [x for x in list(max_dummy.values) if x != np.inf][0]*coeff # arbitrary large value
        min_dummy = df_dummy[[col for col in list(df_rules.columns) if '_min' in col]].min() # arbitrary low value
        min_dummy = [x for x in list(min_dummy.values) if x != -np.inf][0]
        min_dummy = min_dummy*coeff if min_dummy < 0 else min_dummy/coeff
          
        df_yes_pruned = df_yes_pruned.drop_duplicates().reset_index(drop=True)
        df_yes_pruned = df_yes_pruned.replace(np.inf, max_dummy)
        df_yes_pruned = df_yes_pruned.replace(-np.inf, min_dummy)
        df_yes_pruned = rule_overlapping_score(df_yes_pruned, df_anomalies,
                                               feature_cols, [])
        
        df_no_pruned = df_no_pruned.drop_duplicates().reset_index(drop=True)
        df_no_pruned = df_no_pruned.replace(np.inf, max_dummy)
        df_no_pruned = df_no_pruned.replace(-np.inf, min_dummy)
        df_no_pruned = rule_overlapping_score(df_no_pruned, df_anomalies,
                                               feature_cols,
                                               [])
        
        ### Stability
        df_yes_pruned = check_stability(df_anomalies, df_yes_pruned, model,
                                        feature_cols, [],
                                        using_inliers = False)
        df_no_pruned = check_stability(df_anomalies, df_no_pruned, model,
                                        feature_cols, [],
                                        using_inliers = True)
        
        # Replace with original limits
        df_yes_pruned = df_yes_pruned.replace(max_dummy, np.inf)
        df_yes_pruned = df_yes_pruned.replace(min_dummy, -np.inf)
        df_no_pruned = df_no_pruned.replace(max_dummy, np.inf)
        df_no_pruned = df_no_pruned.replace(min_dummy, -np.inf)
    
    # Save to CSV
    if len("path")>0:
        print("Saving results (P@1 rules)...")
        df_yes_pruned.to_csv("{path}/{file_name}_rules_outliers_pruned_Anchors.csv".format(path=path, file_name=file_name),
                                    index=False)
        df_no_pruned.to_csv("{path}/{file_name}_rules_inliers_pruned_Anchors.csv".format(path=path, file_name=file_name),
                                   index=False)
    
    print("Process succsesfully finished!")
    
    df_rules_inliers = df_rules_anchors_no
    df_rules_outliers = df_rules_anchors_yes
    df_rules_inliers_p1 = df_no_pruned
    df_rules_outliers_p1 = df_yes_pruned
    
    return (df_rules_inliers, df_rules_outliers,
            df_rules_inliers_p1, df_rules_outliers_p1)
    


def rulefit_rules(df_anomalies, model, numerical_cols, categorical_cols, metrics=True,
                  path="",  file_name="default_name"):
    """
    TODO

    Parameters
    ----------
    df_anomalies : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    numerical_cols : TYPE
        DESCRIPTION.
    categorical_cols : TYPE
        DESCRIPTION.
    metrics : TYPE, optional
        DESCRIPTION. The default is True.
    path : TYPE, optional
        DESCRIPTION. The default is "".
    file_name : TYPE, optional
        DESCRIPTION. The default is "default_name".

    Returns
    -------
    df_check : TYPE
        DESCRIPTION.
    df_rules_inliers_p1 : TYPE
        DESCRIPTION.
    df_rules_outliers_p1 : TYPE
        DESCRIPTION.

    """

    # Prepare Data
    feature_cols = numerical_cols + categorical_cols
    
    ### All Rules
    df_aux = df_anomalies.copy()
    
    # Prepare data
    X_train = df_aux[feature_cols]
    y_train = df_aux['predictions']
    
    # Fit model
    rf = RuleFit(tree_size=len(feature_cols)*2, rfmode='classify')
    rf.fit(X_train.values, y_train.values, feature_names=feature_cols) 
    
    # Get rules
    print("Obtaining Rules using RuleFit...")
    rules_all = rf.get_rules()
    rules_all.to_csv("{path}/{file_name}_RuleFit.csv".format(path=path, file_name=file_name), index=False)
    rules_all = rules_all[rules_all.coef != 0]
    rules_all = rules_all[rules_all.importance > 0].sort_values("support", ascending=False)
    rules_all = rules_all[rules_all['type']=="rule"]
    rules_all['size_rules'] = rules_all.apply(lambda x: len(x['rule'].split("&")), axis=1)
    
    # Obtain rules in df format
    print("Turning Rules to hypercubes...")
    df_rules_all = turn_rules_to_df(df_anomalies, list(rules_all['rule']),
                                  feature_cols)
    df_rules_all['size_rules'] = rules_all['size_rules'].values
    
    list_cols_limits = [col for col in list(df_rules_all.columns) if col != "size_rules"]
    df_plot = df_rules_all.copy()
    df_plot = df_plot.drop_duplicates(subset=list_cols_limits).reset_index(drop=True)
    df_check = df_plot.copy()
    
    # Check datapoints inside hypercubes (inliers)
    print("Checking inliers inside hypercubes...")
    df_check['n_inliers_included'] = 0
    for i, data_point in df_anomalies[df_anomalies['predictions']==1].iterrows():
        df_check['n_inliers_included'] += check_datapoint_inside(data_point,
                                                                 df_plot,
                                                                 feature_cols,
                                                                 [])['check']
    
    # Check datapoints inside hypercube (outliers)
    print("Checking outliers inside hypercubes...")
    df_check['n_outliers_included'] = 0
    for i, data_point in df_anomalies[df_anomalies['predictions']==-1].iterrows():
        df_check['n_outliers_included'] += check_datapoint_inside(data_point,
                                                                  df_plot,
                                                                  feature_cols,
                                                                  [])['check']
        
    # Check how many datapoints are included with the rules with Precision=1 and in general
    print("Checking inliers/outliers inside hypercubes with Precision=1...")
    n_inliers_p1 = 0
    n_inliers_p0 = 0
    n_outliers_p1 = 0
    n_outliers_p0 = 0
    n_inliers = len(df_anomalies[df_anomalies['predictions']==1])
    n_outliers = len(df_anomalies[df_anomalies['predictions']==-1])
    
    for i, data_point in df_anomalies.iterrows():
        df_check['check'] = check_datapoint_inside(data_point,
                                                   df_plot,
                                                   feature_cols,
                                                   [])['check']
        
        # If inlier
        if data_point['predictions']==1:
            # Rules with any P and that include this datapoint
            df_aux = df_check[(df_check['check']==1)] 
            if len(df_aux) > 0:
                n_inliers_p0 += 1
            
            # Rules with P=1 and that include this datapoint
            df_aux = df_check[(df_check['n_outliers_included']==0) & (df_check['check']==1)] 
            if len(df_aux) > 0:
                n_inliers_p1 += 1
            
        # If outlier
        elif data_point['predictions']==-1:
            # Rules with any P and that include this datapoint
            df_aux = df_check[(df_check['check']==1)] 
            if len(df_aux) > 0:
                n_outliers_p0 += 1
            
            # Rules with P=1 and that include this datapoint
            df_aux = df_check[(df_check['n_inliers_included']==0) & (df_check['check']==1)] 
            if len(df_aux) > 0:
                n_outliers_p1 += 1
    
    df_check['n_inliers'] = n_inliers
    df_check['n_inliers_p0'] = n_inliers_p0
    df_check['n_inliers_p1'] = n_inliers_p1
    df_check['n_outliers_p1'] = n_outliers_p1
    df_check['n_outliers_p0'] = n_outliers_p0
    df_check['n_outliers'] = n_outliers
    del df_check['check']
    
    # Save Results
    if len(path)>0:
        print("Saving results (RuleFit all rules)...")
        df_check.to_csv("{path}/{file_name}_rules_RuleFit.csv".format(path=path, file_name=file_name), index=False)
    
    # Prune rules
    df_rules_inliers = df_check[(df_check['n_inliers_included'] > 0) &
                                (df_check['n_outliers_included']==0)]
    df_rules_inliers = df_rules_inliers.reset_index(drop=True)
    if len(df_rules_inliers) > 1:
        df_rules_inliers = simplify_rules_alt([], df_rules_inliers).drop_duplicates()
    df_rules_outliers = df_check[(df_check['n_outliers_included'] > 0) &
                                 (df_check['n_inliers_included']==0)]
    df_rules_outliers = df_rules_outliers.reset_index(drop=True)
    if len(df_rules_outliers) > 1:
        df_rules_outliers = simplify_rules_alt([], df_rules_outliers).drop_duplicates()
    
    # Obtain additional metrics
    if metrics:
        print("Obtaining additional metrics...")
        coeff = 1000
        
        ### Overlapping
        df_rules = df_rules_inliers.append(df_rules_outliers)
        df_dummy = df_rules.copy().replace(np.inf, 0).replace(-np.inf, 0) # discard infinities
        max_dummy = df_dummy[[col for col in list(df_rules.columns) if '_max' in col]].max()
        max_dummy = [x for x in list(max_dummy.values) if x != np.inf][0]*coeff # arbitrary large value
        min_dummy = df_dummy[[col for col in list(df_rules.columns) if '_min' in col]].min() # arbitrary low value
        min_dummy = [x for x in list(min_dummy.values) if x != -np.inf][0]
        min_dummy = min_dummy*coeff if min_dummy < 0 else min_dummy/coeff
          
        df_rules_inliers = df_rules_inliers.drop_duplicates().reset_index(drop=True)
        df_rules_inliers = df_rules_inliers.replace(np.inf, max_dummy)
        df_rules_inliers = df_rules_inliers.replace(-np.inf, min_dummy)
        df_rules_inliers = rule_overlapping_score(df_rules_inliers, df_anomalies,
                                                  feature_cols,
                                                  [])
        
        df_rules_outliers = df_rules_outliers.drop_duplicates().reset_index(drop=True)
        df_rules_outliers = df_rules_outliers.replace(np.inf, max_dummy)
        df_rules_outliers = df_rules_outliers.replace(-np.inf, min_dummy)
        df_rules_outliers = rule_overlapping_score(df_rules_outliers, df_anomalies,
                                                   feature_cols,
                                                   [])
        
        ### Stability
        df_rules_outliers = check_stability(df_anomalies, df_rules_outliers, model,
                                            feature_cols, [],
                                            using_inliers = False)
        df_rules_inliers = check_stability(df_anomalies, df_rules_inliers, model,
                                           feature_cols, [],
                                           using_inliers = True)
        
        # Replace with original limits
        df_rules_inliers = df_rules_inliers.replace(max_dummy, np.inf)
        df_rules_inliers = df_rules_inliers.replace(min_dummy, -np.inf)
        df_rules_outliers = df_rules_outliers.replace(max_dummy, np.inf)
        df_rules_outliers = df_rules_outliers.replace(min_dummy, -np.inf)
    
    # Save to CSV
    if len(path)>0:
        print("Saving results (P@1 rules)...")
        df_rules_outliers.to_csv("{path}/{file_name}_rules_outliers_pruned_RuleFit.csv".format(path=path, file_name=file_name), index=False)
        df_rules_inliers.to_csv("{path}/{file_name}_rules_inliers_pruned_RuleFit.csv".format(path=path, file_name=file_name), index=False)
    
    print("Process succsesfully finished!")
    df_rules_inliers_p1 = df_rules_inliers
    df_rules_outliers_p1 = df_rules_outliers
    
    return df_check, df_rules_inliers_p1, df_rules_outliers_p1
 
    
def skoperules_rules(df_anomalies, model, numerical_cols, categorical_cols,
                     metrics=True, path="",  file_name="default_name"):
    """
    TODO

    Parameters
    ----------
    df_anomalies : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    numerical_cols : TYPE
        DESCRIPTION.
    categorical_cols : TYPE
        DESCRIPTION.
    metrics : TYPE, optional
        DESCRIPTION. The default is True.
    path : TYPE, optional
        DESCRIPTION. The default is "".
    file_name : TYPE, optional
        DESCRIPTION. The default is "default_name".

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """


    ### Inliers
    # Prepare Data
    feature_cols = numerical_cols + categorical_cols
    df_aux = df_anomalies.copy()
    df_aux['predictions'] = df_aux.apply(lambda x: 0 if x['predictions'] < 0 else 1, axis=1)
    
    X_train = df_aux[feature_cols]
    y_train = df_aux['predictions']
    
    # Rules
    print("Obtaining Rules using SkopeRules...")
    rng = np.random.RandomState(42)
    clf = SkopeRules(random_state=rng,
                     precision_min=1.0,
                     recall_min=0.0,
                     feature_names=feature_cols)
    clf.fit(X_train, y_train)
    rules = clf.rules_
    
    if len(rules)==0:
        print("No rules found for this dataset")
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    
    df_rules_info_inliers = pd.DataFrame({"rule":[v[0].replace(" and ", " & ") for v in rules],
                                          "precision":[v[1][0] for v in rules],
                                          "recall":[v[1][1] for v in rules],
                                          "n_points_correct":[v[1][2] for v in rules]})
    df_rules_info_inliers['size_rules'] = df_rules_info_inliers.apply(lambda x: len(x['rule'].split("&")), axis=1)
    
    rules = [v[0].replace(" and ", " & ") for v in rules]
    
    # Obtain rules in df format
    print("Turning Rules to hypercubes...")
    df_rules_inliers = turn_rules_to_df(df_anomalies,
                                        rules,
                                        feature_cols)
    df_rules_inliers = df_rules_inliers.reset_index(drop=True)
    df_rules_inliers['size_rules'] = df_rules_info_inliers['size_rules']
    
    
    ### Outliers
    # Prepare Data
    feature_cols = numerical_cols + categorical_cols
    df_aux = df_anomalies.copy()
    df_aux['predictions'] = df_aux.apply(lambda x: 1 if x['predictions'] < 0 else 0, axis=1)
    
    X_train = df_aux[feature_cols]
    y_train = df_aux['predictions']
    
    # Rules
    print("Obtaining Rules using SkopeRules...")
    rng = np.random.RandomState(42)
    clf = SkopeRules(random_state=rng,
                     precision_min=1.0,
                     recall_min=0.0,
                     feature_names=feature_cols)
    clf.fit(X_train, y_train)
    rules = clf.rules_
    
    df_rules_info_outliers = pd.DataFrame({"rule":[v[0].replace(" and ", " & ") for v in rules],
                                  "precision":[v[1][0] for v in rules],
                                  "recall":[v[1][1] for v in rules],
                                  "n_points_correct":[v[1][2] for v in rules]})
    df_rules_info_outliers['size_rules'] = df_rules_info_outliers.apply(lambda x: len(x['rule'].split("&")), axis=1)
    
    rules = [v[0].replace(" and ", " & ") for v in rules]
    
    if len(rules)==0:
        print("No rules found for this dataset")
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    
    # Obtain rules in df format
    print("Turning Rules to hypercubes...")
    df_rules_outliers = turn_rules_to_df(df_anomalies,
                                        rules,
                                        feature_cols)
    df_rules_outliers = df_rules_outliers.reset_index(drop=True)
    df_rules_outliers['size_rules'] = df_rules_info_outliers['size_rules']
    
    ### Check datapoints inside rules
    # Check datapoints inside hypercubes (inliers)
    print("Checking inliers inside hypercubes...")
    df_rules_inliers['n_inliers_included'] = 0
    df_rules_outliers['n_inliers_included'] = 0
    
    for i, data_point in df_anomalies[df_anomalies['predictions']==1].iterrows():
        df_rules_inliers['n_inliers_included'] += check_datapoint_inside(data_point,
                                                                         df_rules_inliers,
                                                                         feature_cols,
                                                                         [])['check']
        
        df_rules_outliers['n_inliers_included'] += check_datapoint_inside(data_point,
                                                                         df_rules_outliers,
                                                                         feature_cols,
                                                                         [])['check']
        
    
    ### Check datapoints inside hypercube (outliers)
    print("Checking outliers inside hypercubes...")
    df_rules_inliers['n_outliers_included'] = 0
    df_rules_outliers['n_outliers_included'] = 0
    
    for i, data_point in df_anomalies[df_anomalies['predictions']==-1].iterrows():   
        df_rules_inliers['n_outliers_included'] += check_datapoint_inside(data_point,
                                                                          df_rules_inliers,
                                                                          feature_cols,
                                                                          [])['check']
        
        df_rules_outliers['n_outliers_included'] += check_datapoint_inside(data_point,
                                                                           df_rules_outliers,
                                                                           feature_cols,
                                                                           [])['check']
        
    # Check how many datapoints are included with the rules with Precision=1
    print("Checking inliers/outliers inside hypercubes with Precision=1...")
    n_inliers_p1 = 0
    n_inliers_p0 = 0
    n_outliers_p1 = 0
    n_outliers_p0 = 0
    n_inliers = len(df_anomalies[df_anomalies['predictions']==1])
    n_outliers = len(df_anomalies[df_anomalies['predictions']==-1])
        
    for i, data_point in df_anomalies.iterrows():
        df_rules_inliers['check'] = check_datapoint_inside(data_point,
                                                           df_rules_inliers,
                                                           feature_cols,
                                                           [])['check']
        
        df_rules_outliers['check'] = check_datapoint_inside(data_point,
                                                           df_rules_outliers,
                                                           feature_cols,
                                                           [])['check']
        # If inlier
        if data_point['predictions']==1:
            # Rules with any P and that include this datapoint
            df_aux = df_rules_inliers[(df_rules_inliers['check']==1)] 
            if len(df_aux) > 0:
                n_inliers_p0 += 1
            
            # Rules with P=1 and that include this datapoint
            df_aux = df_rules_inliers[(df_rules_inliers['n_outliers_included']==0)
                                       & (df_rules_inliers['check']==1)] 
            if len(df_aux) > 0:
                n_inliers_p1 += 1
            
        # If outlier
        elif data_point['predictions']==-1:
            # Rules with any P and that include this datapoint
            df_aux = df_rules_outliers[(df_rules_outliers['check']==1)] 
            if len(df_aux) > 0:
                n_outliers_p0 += 1
            
            # Rules with P=1 and that include this datapoint
            df_aux = df_rules_outliers[(df_rules_outliers['n_inliers_included']==0)
                                       & (df_rules_outliers['check']==1)] 
            if len(df_aux) > 0:
                n_outliers_p1 += 1
    
    df_rules_inliers['n_inliers'] = n_inliers
    df_rules_inliers['n_inliers_p0'] = n_inliers_p0
    df_rules_inliers['n_inliers_p1'] = n_inliers_p1
    df_rules_outliers['n_outliers_p1'] = n_outliers_p1
    df_rules_outliers['n_outliers_p0'] = n_outliers_p0
    df_rules_outliers['n_outliers'] = n_outliers
    del df_rules_inliers['check'], df_rules_outliers['check']
    
    # Save to CSV
    if len(path)>0:
        print("Saving results (all rules)...")
        df_rules_outliers.to_csv("{path}/{file_name}_rules_outliers_SkopeRules.csv".format(path=path, file_name=file_name), index=False)
        df_rules_inliers.to_csv("{path}/{file_name}_rules_inliers_SkopeRules.csv".format(path=path, file_name=file_name), index=False)
      
    # Prune rules
    df_yes_pruned = df_rules_outliers[(df_rules_outliers['n_inliers_included'] == 0) &
                                      (df_rules_outliers['n_outliers_included'] > 0)]
    df_yes_pruned = df_yes_pruned.reset_index(drop=True)
    
    if len(df_yes_pruned) > 1:
        df_yes_pruned = simplify_rules_alt([], df_yes_pruned).drop_duplicates()
        
    df_no_pruned = df_rules_inliers[(df_rules_inliers['n_inliers_included'] > 0) &
                                    (df_rules_inliers['n_outliers_included'] == 0)]
    df_no_pruned = df_no_pruned.reset_index(drop=True)
    
    if len(df_no_pruned) > 1:
        df_no_pruned = simplify_rules_alt([], df_no_pruned).drop_duplicates()
        
    # Obtain additional metrics
    if metrics:
        print("Obtaining additional metrics...")
        coeff = 1000
        ### Overlapping
        df_rules = df_rules_inliers.append(df_rules_outliers)
        df_dummy = df_rules.copy().replace(np.inf, 0).replace(-np.inf, 0) # discard infinities
        max_dummy = df_dummy[[col for col in list(df_rules.columns) if '_max' in col]].max()
        max_dummy = [x for x in list(max_dummy.values) if x != np.inf][0]*coeff # arbitrary large value
        min_dummy = df_dummy[[col for col in list(df_rules.columns) if '_min' in col]].min() # arbitrary low value
        min_dummy = [x for x in list(min_dummy.values) if x != -np.inf][0]
        min_dummy = min_dummy*coeff if min_dummy < 0 else min_dummy/coeff
        
        df_yes_pruned = df_yes_pruned.drop_duplicates().reset_index(drop=True)
        df_yes_pruned = df_yes_pruned.replace(np.inf, max_dummy)
        df_yes_pruned = df_yes_pruned.replace(-np.inf, min_dummy)
        df_yes_pruned = rule_overlapping_score(df_yes_pruned, df_anomalies,
                                               feature_cols,
                                               [])
        
        df_no_pruned = df_no_pruned.drop_duplicates().reset_index(drop=True)
        df_no_pruned = df_no_pruned.replace(np.inf, max_dummy)
        df_no_pruned = df_no_pruned.replace(-np.inf, min_dummy)
        df_no_pruned = rule_overlapping_score(df_no_pruned, df_anomalies,
                                              feature_cols,
                                              [])
        ### Stability
        df_yes_pruned = check_stability(df_anomalies, df_yes_pruned, model,
                                        feature_cols,
                                        [],
                                        using_inliers = False)
        df_no_pruned = check_stability(df_anomalies, df_no_pruned, model,
                                        feature_cols,
                                        [],
                                        using_inliers = True)
        
        # Replace with original limits
        df_yes_pruned = df_yes_pruned.replace(max_dummy, np.inf)
        df_yes_pruned = df_yes_pruned.replace(min_dummy, -np.inf)
        df_no_pruned = df_no_pruned.replace(max_dummy, np.inf)
        df_no_pruned = df_no_pruned.replace(min_dummy, -np.inf)
    
    # Save to CSV
    if len(path)>0:
        print("Savinr results (P@1 rules)...")
        df_yes_pruned.to_csv("{path}/{file_name}_rules_outliers_pruned_SkopeRules.csv".format(path=path, file_name=file_name),
                                    index=False)
        df_no_pruned.to_csv("{path}/{file_name}_rules_inliers_pruned_SkopeRules.csv".format(path=path, file_name=file_name),
                                   index=False)
    
    df_rules_inliers_p1 = df_no_pruned
    df_rules_outliers_p1 = df_yes_pruned
    
    return (df_rules_inliers, df_rules_outliers,
            df_rules_inliers_p1, df_rules_outliers_p1)

        
        
def frl_rules(df_anomalies, model, numerical_cols, categorical_cols,
              metrics=True, path="",  file_name="default_name"):
    
    """
    Rules obtained using Bayesian Falling Rules List.

    Parameters
    ----------
    df_anomalies : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    numerical_cols : TYPE
        DESCRIPTION.
    categorical_cols : TYPE
        DESCRIPTION.
    metrics : TYPE, optional
        DESCRIPTION. The default is True.
    path : TYPE, optional
        DESCRIPTION. The default is "".
    file_name : TYPE, optional
        DESCRIPTION. The default is "default_name".

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    
    def turn_ruleset_to_df(df_anomalies, list_rules, list_cols):
        df_rules = pd.DataFrame()
        
        # Iter for each rule
        for rule in list_rules:
            dct_aux = {}
            
            # Default values
            for col in list_cols:
                dct_aux[col + '_max'] = np.inf
                dct_aux[col + '_min'] = -np.inf
                
            list_subrules = rule
            
            # Iter for each component of the rule and obtain the limits
            for subrule in list_subrules:
                for col in list_cols:
                    
                    if col in subrule:
                        aux = subrule.split("<") 
                        # col < XXXX
                        if aux[0] in list_cols:
                            if np.float(aux[1]) <= dct_aux[col + '_max']:
                                dct_aux[col + '_max'] = np.float(aux[1])
                        # XXXX < col
                        else:
                            if np.float(aux[0]) >= dct_aux[col + '_min']:
                                dct_aux[col + '_min'] = np.float(aux[0])
             
            df_rules = df_rules.append(pd.DataFrame(dct_aux, index=[0]))
        
        return df_rules
    
    feature_cols = numerical_cols + categorical_cols
    X = df_anomalies[feature_cols]
    
    ### Rules for Inliers
    y = df_anomalies['predictions'].values
    y = np.array([x if x > 0 else 0 for x in y])
    model_rules = ruleset.BayesianRuleSet()
    model_rules.fit(X, y)
    dict_rules = model_rules.rule_explainations
    list_rules = [x[0] for i,x in dict_rules.items()]
    df_rules_inliers = turn_ruleset_to_df(df_anomalies, list_rules, feature_cols)
    df_rules_inliers['size_rules'] = [len(x) for x in list_rules]
    
    #### Rules for Outliers
    y = df_anomalies['predictions'].values
    y = np.array([1 if x < 0 else 0 for x in y])
    model_rules = ruleset.BayesianRuleSet()
    model_rules.fit(X, y)
    dict_rules = model_rules.rule_explainations
    list_rules = [x[0] for i,x in dict_rules.items()]
    df_rules_outliers = turn_ruleset_to_df(df_anomalies, list_rules, feature_cols)
    df_rules_outliers['size_rules'] = [len(x) for x in list_rules]
    
    ### Check datapoints inside rules
    # Check datapoints inside hypercubes (inliers)
    print("Checking inliers inside hypercubes...")
    df_rules_inliers['n_inliers_included'] = 0
    df_rules_outliers['n_inliers_included'] = 0
    
    for i, data_point in df_anomalies[df_anomalies['predictions']==1].iterrows():
        df_rules_inliers['n_inliers_included'] += check_datapoint_inside(data_point,
                                                                         df_rules_inliers,
                                                                         feature_cols,
                                                                         [])['check']
        
        df_rules_outliers['n_inliers_included'] += check_datapoint_inside(data_point,
                                                                         df_rules_outliers,
                                                                         feature_cols,
                                                                         [])['check']
    
    ### Check datapoints inside hypercube (outliers)
    print("Checking outliers inside hypercubes...")
    df_rules_inliers['n_outliers_included'] = 0
    df_rules_outliers['n_outliers_included'] = 0
    
    for i, data_point in df_anomalies[df_anomalies['predictions']==-1].iterrows():   
        df_rules_inliers['n_outliers_included'] += check_datapoint_inside(data_point,
                                                                          df_rules_inliers,
                                                                          feature_cols,
                                                                          [])['check']
        
        df_rules_outliers['n_outliers_included'] += check_datapoint_inside(data_point,
                                                                           df_rules_outliers,
                                                                           feature_cols,
                                                                           [])['check']
        
    # Check how many datapoints are included with the rules with Precision=1
    print("Checking inliers/outliers inside hypercubes with Precision=1...")
    n_inliers_p1 = 0
    n_inliers_p0 = 0
    n_outliers_p1 = 0
    n_outliers_p0 = 0
    n_inliers = len(df_anomalies[df_anomalies['predictions']==1])
    n_outliers = len(df_anomalies[df_anomalies['predictions']==-1])
        
    for i, data_point in df_anomalies.iterrows():
        df_rules_inliers['check'] = check_datapoint_inside(data_point,
                                                           df_rules_inliers,
                                                           feature_cols,
                                                           [])['check']
        
        df_rules_outliers['check'] = check_datapoint_inside(data_point,
                                                           df_rules_outliers,
                                                           feature_cols,
                                                           [])['check']
        # If inlier
        if data_point['predictions']==1:
            # Rules with any P and that include this datapoint
            df_aux = df_rules_inliers[(df_rules_inliers['check']==1)] 
            if len(df_aux) > 0:
                n_inliers_p0 += 1
            
            # Rules with P=1 and that include this datapoint
            df_aux = df_rules_inliers[(df_rules_inliers['n_outliers_included']==0)
                                       & (df_rules_inliers['check']==1)] 
            if len(df_aux) > 0:
                n_inliers_p1 += 1
            
        # If outlier
        elif data_point['predictions']==-1:
            # Rules with any P and that include this datapoint
            df_aux = df_rules_outliers[(df_rules_outliers['check']==1)] 
            if len(df_aux) > 0:
                n_outliers_p0 += 1
            
            # Rules with P=1 and that include this datapoint
            df_aux = df_rules_outliers[(df_rules_outliers['n_inliers_included']==0)
                                       & (df_rules_outliers['check']==1)] 
            if len(df_aux) > 0:
                n_outliers_p1 += 1
    
    df_rules_inliers['n_inliers'] = n_inliers
    df_rules_inliers['n_inliers_p0'] = n_inliers_p0
    df_rules_inliers['n_inliers_p1'] = n_inliers_p1
    df_rules_outliers['n_outliers_p1'] = n_outliers_p1
    df_rules_outliers['n_outliers_p0'] = n_outliers_p0
    df_rules_outliers['n_outliers'] = n_outliers
    del df_rules_inliers['check'], df_rules_outliers['check']
    
    # Save to CSV
    if len(path)>0:
        print("Saving results (all rules)...")
        df_rules_outliers.to_csv("{path}/{file_name}_rules_outliers_FRL.csv".format(path=path, file_name=file_name), index=False)
        df_rules_inliers.to_csv("{path}/{file_name}_rules_inliers_FRL.csv".format(path=path, file_name=file_name), index=False)
        
    # Prune rules
    df_yes_pruned = df_rules_outliers[(df_rules_outliers['n_inliers_included'] == 0) &
                                      (df_rules_outliers['n_outliers_included'] > 0)]
    df_yes_pruned = df_yes_pruned.reset_index(drop=True)
    
    if len(df_yes_pruned) > 1:
        df_yes_pruned = simplify_rules_alt([], df_yes_pruned).drop_duplicates()
        
    df_no_pruned = df_rules_inliers[(df_rules_inliers['n_inliers_included'] > 0) &
                                    (df_rules_inliers['n_outliers_included'] == 0)]
    df_no_pruned = df_no_pruned.reset_index(drop=True)
    
    if len(df_no_pruned) > 1:
        df_no_pruned = simplify_rules_alt([], df_no_pruned).drop_duplicates()
        
    # Obtain additional metrics
    if metrics:
        print("Obtaining additional metrics...")
        coeff = 1000
        ### Overlapping
        df_rules = df_rules_inliers.append(df_rules_outliers)
        df_dummy = df_rules.copy().replace(np.inf, 0).replace(-np.inf, 0) # discard infinities
        max_dummy = df_dummy[[col for col in list(df_rules.columns) if '_max' in col]].max()
        max_dummy = [x for x in list(max_dummy.values) if x != np.inf][0]*coeff # arbitrary large value
        min_dummy = df_dummy[[col for col in list(df_rules.columns) if '_min' in col]].min() # arbitrary low value
        min_dummy = [x for x in list(min_dummy.values) if x != -np.inf][0]
        min_dummy = min_dummy*coeff if min_dummy < 0 else min_dummy/coeff
         
        df_yes_pruned = df_yes_pruned.drop_duplicates().reset_index(drop=True)
        df_yes_pruned = df_yes_pruned.replace(np.inf, max_dummy)
        df_yes_pruned = df_yes_pruned.replace(-np.inf, min_dummy)
        df_yes_pruned = rule_overlapping_score(df_yes_pruned, df_anomalies,
                                               feature_cols,
                                               [])
        
        df_no_pruned = df_no_pruned.drop_duplicates().reset_index(drop=True)
        df_no_pruned = df_no_pruned.replace(np.inf, max_dummy)
        df_no_pruned = df_no_pruned.replace(-np.inf, min_dummy)
        df_no_pruned = rule_overlapping_score(df_no_pruned, df_anomalies,
                                               feature_cols,
                                               [])
        
        ### Stability
        df_yes_pruned = check_stability(df_anomalies, df_yes_pruned, model,
                                        feature_cols,
                                        [],
                                        using_inliers = False)
        df_no_pruned = check_stability(df_anomalies, df_no_pruned, model,
                                        feature_cols,
                                        [],
                                        using_inliers = True)
        
        # Replace with original limits
        df_yes_pruned = df_yes_pruned.replace(max_dummy, np.inf)
        df_yes_pruned = df_yes_pruned.replace(min_dummy, -np.inf)
        df_no_pruned = df_no_pruned.replace(max_dummy, np.inf)
        df_no_pruned = df_no_pruned.replace(min_dummy, -np.inf)
        
    # Save to CSV
    if len(path)>0:
        print("Saving results (P@1 rules)...")
        df_yes_pruned.to_csv("{path}/{file_name}_rules_outliers_pruned_FRL.csv".format(path=path, file_name=file_name),
                                    index=False)
        df_no_pruned.to_csv("{path}/{file_name}_rules_inliers_pruned_FRL.csv".format(path=path, file_name=file_name),
                                   index=False)
        
    df_rules_inliers_p1 = df_no_pruned
    df_rules_outliers_p1 = df_yes_pruned
    
    return (df_rules_inliers, df_rules_outliers,
            df_rules_inliers_p1, df_rules_outliers_p1)



def aix360_rules_wrapper(df_anomalies, model, numerical_cols, categorical_cols,
                         use_oversampling=False, rule_algorithm="", 
                         metrics=True, path="", file_name=""):
    """
    TODO

    Parameters
    ----------
    df_anomalies : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    numerical_cols : TYPE
        DESCRIPTION.
    categorical_cols : TYPE
        DESCRIPTION.
    use_oversampling : TYPE, optional
        DESCRIPTION. The default is False.
    rule_algorithm : TYPE, optional
        DESCRIPTION. The default is "".
    metrics : TYPE, optional
        DESCRIPTION. The default is True.
    path : TYPE, optional
        DESCRIPTION. The default is "".
    file_name : TYPE, optional
        DESCRIPTION. The default is "".

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    df_rules_inliers : TYPE
        DESCRIPTION.
    df_rules_outliers : TYPE
        DESCRIPTION.
    df_rules_inliers_p1 : TYPE
        DESCRIPTION.
    df_rules_outliers_p1 : TYPE
        DESCRIPTION.

    """


    
    # Define variables
    feature_cols = numerical_cols + categorical_cols
    X = df_anomalies[feature_cols].astype(float)
    y = df_anomalies['predictions'].astype(int)
    y_inliers = np.array([x if x > 0 else 0 for x in y]) # Defined for inliers levels
    y_outliers = np.array([1 if x < 0 else 0 for x in y]) # Defined for outlier levels
    
    # Perform oversampling
    if use_oversampling:
        sm = SMOTE(sampling_strategy="minority",
                   k_neighbors=int(np.round(np.sqrt(len(X)))),
                   random_state=42)
        X_inliers, y_inliers = sm.fit_resample(X, y_inliers)
        X_outliers, y_outliers = sm.fit_resample(X, y_outliers)
  
    # Feature binarize 
    fb = FeatureBinarizer(negations=True, returnOrd=True)
    X_fb_inliers, X_std_inliers = fb.fit_transform(X_inliers)
    X_fb_outliers, X_std_outliers = fb.fit_transform(X_outliers)

    # Choose model
    if rule_algorithm=="brlg":
        
        # Inliers
        model_rules = BooleanRuleCG(lambda0=1e-3, lambda1=1e-3, CNF=False)
        model_rules.fit(X_fb_inliers, y_inliers)
        list_rules_inliers = model_rules.explain()['rules']
        
        # Outliers
        model_rules = BooleanRuleCG(lambda0=1e-3, lambda1=1e-3, CNF=False)
        model_rules.fit(X_fb_outliers, y_outliers)
        list_rules_outliers = model_rules.explain()['rules']
        
    elif rule_algorithm=="logrr":
        
        # Obtain rules [Inliers]
        model_rules_inliers = LogisticRuleRegression(lambda0=0.005, lambda1=0.001, useOrd=True)
        model_rules_inliers.fit(X_fb_inliers, y_inliers, X_std_inliers)
        df_rules_inliers = model_rules_inliers.explain()
        
        # Obtain rules [Outliers]
        model_rules_outliers = LogisticRuleRegression(lambda0=0.005, lambda1=0.001, useOrd=True)
        model_rules_outliers.fit(X_fb_outliers, y_outliers, X_std_outliers)
        df_rules_outliers = model_rules_outliers.explain()    
        
        try:
            # Inliers
            df_rules_inliers = df_rules_inliers[(df_rules_inliers['coefficient']>0) &
                                        (df_rules_inliers['rule/numerical feature'] != "(intercept)")]
            list_rules_inliers = list(df_rules_inliers['rule/numerical feature'])
            
            # Outliers
            df_rules_outliers = df_rules_outliers[(df_rules_outliers['coefficient']<0) &
                                        (df_rules_outliers['rule/numerical feature'] != "(intercept)")]
            list_rules_outliers = list(df_rules_outliers['rule/numerical feature'])
        except KeyError:
            # Inliers
            df_rules_inliers = df_rules_inliers[(df_rules_inliers['coefficient']>0) &
                                        (df_rules_inliers['rule'] != "(intercept)")]
            list_rules_inliers = list(df_rules_inliers['rule'])
            
            # Outliers
            df_rules_outliers = df_rules_outliers[(df_rules_outliers['coefficient']<0) &
                                        (df_rules_outliers['rule'] != "(intercept)")]
            list_rules_outliers = list(df_rules_outliers['rule'])
           
    elif rule_algorithm=="glrm":
        logistic_model = LogisticRuleRegression(maxSolverIter=2000)
        model_rules_inliers = GLRMExplainer(logistic_model)
        model_rules_inliers.fit(X_fb_inliers, y_inliers)
        df_rules_inliers = model_rules_inliers.explain()
        
        logistic_model = LogisticRuleRegression(maxSolverIter=2000)
        model_rules_outliers = GLRMExplainer(logistic_model)
        model_rules_outliers.fit(X_fb_outliers, y_outliers)
        df_rules_outliers = model_rules_outliers.explain()
        
        # Inliers
        df_rules_inliers = df_rules_inliers[(df_rules_inliers['coefficient']>0) &
                                    (df_rules_inliers['rule'] != "(intercept)")]
        list_rules_inliers = list(df_rules_inliers['rule'])
        # Outliers
        df_rules_outliers = df_rules_outliers[(df_rules_outliers['coefficient']<0) &
                                    (df_rules_outliers['rule'] != "(intercept)")]
        list_rules_outliers = list(df_rules_outliers['rule'])
        
    else:
        raise ValueError("Argument {0} not recognised -- use 'brlg', 'logrr', 'glrm' instead")
        
    # Turn to DF
    list_rules_inliers = [x.replace("AND", "&") for x in list_rules_inliers]
    list_rules_outliers = [x.replace("AND", "&") for x in list_rules_outliers]
    df_inliers = turn_rules_to_df(df_anomalies, list_rules_inliers, feature_cols)
    df_outliers = turn_rules_to_df(df_anomalies, list_rules_outliers, feature_cols)
    
    df_rules_inliers = df_inliers.reset_index(drop=True)
    df_rules_inliers['size_rules'] = [len(x.split("&")) for x in list_rules_inliers]
    df_rules_outliers = df_outliers.reset_index(drop=True)
    df_rules_outliers['size_rules'] = [len(x.split("&")) for x in list_rules_outliers]
    
    ### Check datapoints inside rules
    # Check datapoints inside hypercubes (inliers)
    print("Checking inliers inside hypercubes...")
    df_rules_inliers['n_inliers_included'] = 0
    df_rules_outliers['n_inliers_included'] = 0
    
    for i, data_point in df_anomalies[df_anomalies['predictions']==1].iterrows():
        df_rules_inliers['n_inliers_included'] += check_datapoint_inside(data_point,
                                                                         df_rules_inliers,
                                                                         feature_cols,
                                                                         [])['check']
        
        df_rules_outliers['n_inliers_included'] += check_datapoint_inside(data_point,
                                                                         df_rules_outliers,
                                                                         feature_cols,
                                                                         [])['check']
        
    
    ### Check datapoints inside hypercube (outliers)
    print("Checking outliers inside hypercubes...")
    df_rules_inliers['n_outliers_included'] = 0
    df_rules_outliers['n_outliers_included'] = 0
    
    for i, data_point in df_anomalies[df_anomalies['predictions']==-1].iterrows():   
        df_rules_inliers['n_outliers_included'] += check_datapoint_inside(data_point,
                                                                          df_rules_inliers,
                                                                          feature_cols,
                                                                          [])['check']
        
        df_rules_outliers['n_outliers_included'] += check_datapoint_inside(data_point,
                                                                           df_rules_outliers,
                                                                           feature_cols,
                                                                           [])['check']
        
    # Check how many datapoints are included with the rules with Precision=1
    print("Checking inliers/outliers inside hypercubes with Precision=1...")
    n_inliers_p1 = 0
    n_inliers_p0 = 0
    n_outliers_p1 = 0
    n_outliers_p0 = 0
    n_inliers = len(df_anomalies[df_anomalies['predictions']==1])
    n_outliers = len(df_anomalies[df_anomalies['predictions']==-1])
        
    for i, data_point in df_anomalies.iterrows():
        df_rules_inliers['check'] = check_datapoint_inside(data_point,
                                                           df_rules_inliers,
                                                           feature_cols,
                                                           [])['check']
        
        df_rules_outliers['check'] = check_datapoint_inside(data_point,
                                                           df_rules_outliers,
                                                           feature_cols,
                                                           [])['check']
        # If inlier
        if data_point['predictions']==1:
            # Rules with any P and that include this datapoint
            df_aux = df_rules_inliers[(df_rules_inliers['check']==1)] 
            if len(df_aux) > 0:
                n_inliers_p0 += 1
            
            # Rules with P=1 and that include this datapoint
            df_aux = df_rules_inliers[(df_rules_inliers['n_outliers_included']==0)
                                       & (df_rules_inliers['check']==1)] 
            if len(df_aux) > 0:
                n_inliers_p1 += 1
            
        # If outlier
        elif data_point['predictions']==-1:
            # Rules with any P and that include this datapoint
            df_aux = df_rules_outliers[(df_rules_outliers['check']==1)] 
            if len(df_aux) > 0:
                n_outliers_p0 += 1
            
            # Rules with P=1 and that include this datapoint
            df_aux = df_rules_outliers[(df_rules_outliers['n_inliers_included']==0)
                                       & (df_rules_outliers['check']==1)] 
            if len(df_aux) > 0:
                n_outliers_p1 += 1
    
    df_rules_inliers['n_inliers'] = n_inliers
    df_rules_inliers['n_inliers_p0'] = n_inliers_p0
    df_rules_inliers['n_inliers_p1'] = n_inliers_p1
    df_rules_outliers['n_outliers_p1'] = n_outliers_p1
    df_rules_outliers['n_outliers_p0'] = n_outliers_p0
    df_rules_outliers['n_outliers'] = n_outliers
    del df_rules_inliers['check'], df_rules_outliers['check']
    
    # Save to CSV
    if len(path)>0:
        print("Saving results (all rules)...")
        df_rules_outliers.to_csv("{path}/{file_name}_rules_outliers_{rule_algorithm}.csv".format(path=path,
                                                                                                 file_name=file_name,
                                                                                                 rule_algorithm=rule_algorithm), index=False)
        df_rules_inliers.to_csv("{path}/{file_name}_rules_inliers_{rule_algorithm}.csv".format(path=path,
                                                                                               file_name=file_name,
                                                                                               rule_algorithm=rule_algorithm), index=False)
      
    # Prune rules
    df_yes_pruned = df_rules_outliers[(df_rules_outliers['n_inliers_included'] == 0) &
                                      (df_rules_outliers['n_outliers_included'] > 0)]
    df_yes_pruned = df_yes_pruned.reset_index(drop=True)
    
    if len(df_yes_pruned) > 1:
        df_yes_pruned = simplify_rules_alt([], df_yes_pruned).drop_duplicates()
        
    df_no_pruned = df_rules_inliers[(df_rules_inliers['n_inliers_included'] > 0) &
                                    (df_rules_inliers['n_outliers_included'] == 0)]
    df_no_pruned = df_no_pruned.reset_index(drop=True)
    
    if len(df_no_pruned) > 1:
        df_no_pruned = simplify_rules_alt([], df_no_pruned).drop_duplicates()
        
    # Obtain additional metrics
    if metrics:
        print("Obtaining additional metrics...")
        coeff = 1000
        ### Overlapping
        df_rules = df_rules_inliers.append(df_rules_outliers)
        df_dummy = df_rules.copy().replace(np.inf, 0).replace(-np.inf, 0) # discard infinities
        max_dummy = df_dummy[[col for col in list(df_rules.columns) if '_max' in col]].max()
        max_dummy = [x for x in list(max_dummy.values) if x != np.inf][0]*coeff # arbitrary large value
        min_dummy = df_dummy[[col for col in list(df_rules.columns) if '_min' in col]].min() # arbitrary low value
        min_dummy = [x for x in list(min_dummy.values) if x != -np.inf][0]
        min_dummy = min_dummy*coeff if min_dummy < 0 else min_dummy/coeff
            
        df_yes_pruned = df_yes_pruned.drop_duplicates().reset_index(drop=True)
        df_yes_pruned = df_yes_pruned.replace(np.inf, max_dummy)
        df_yes_pruned = df_yes_pruned.replace(-np.inf, min_dummy)
        df_yes_pruned = rule_overlapping_score(df_yes_pruned, df_anomalies,
                                               feature_cols,
                                               [])
        
        df_no_pruned = df_no_pruned.drop_duplicates().reset_index(drop=True)
        df_no_pruned = df_no_pruned.replace(np.inf, max_dummy)
        df_no_pruned = df_no_pruned.replace(-np.inf, min_dummy)
        df_no_pruned = rule_overlapping_score(df_no_pruned, df_anomalies,
                                               feature_cols,
                                               [])
        
        ### Stability
        df_yes_pruned = check_stability(df_anomalies, df_yes_pruned, model,
                                        feature_cols, [],
                                        using_inliers = False)
        df_no_pruned = check_stability(df_anomalies, df_no_pruned, model,
                                        feature_cols, [],
                                        using_inliers = True)
        
        # Replace with original limits
        df_yes_pruned = df_yes_pruned.replace(max_dummy, np.inf)
        df_yes_pruned = df_yes_pruned.replace(min_dummy, -np.inf)
        df_no_pruned = df_no_pruned.replace(max_dummy, np.inf)
        df_no_pruned = df_no_pruned.replace(min_dummy, -np.inf)
    
    # Save to CSV
    if len(path)>0:
        print("Saving results (P@1 rules)...")
        df_yes_pruned.to_csv("{path}/{file_name}_rules_outliers_pruned_{rule_algorithm}.csv".format(path=path,
                                                                                                    file_name=file_name,
                                                                                                    rule_algorithm=rule_algorithm),
                                    index=False)
        df_no_pruned.to_csv("{path}/{file_name}_rules_inliers_pruned_{rule_algorithm}.csv".format(path=path,
                                                                                                  file_name=file_name,
                                                                                                  rule_algorithm=rule_algorithm),
                                   index=False)
    
    
    df_rules_inliers_p1 = df_no_pruned
    df_rules_outliers_p1 = df_yes_pruned
    
    return (df_rules_inliers, df_rules_outliers,
            df_rules_inliers_p1, df_rules_outliers_p1)



def interpretML_DecisionListClassifier(df_anomalies, model, numerical_cols, categorical_cols,
                                       use_oversampling=False, rule_algorithm="", path="",
                                       file_name=""):
    """
    
    NOT IMPLEMENTED
    
    Parameters
    ----------
    df_anomalies : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    numerical_cols : TYPE
        DESCRIPTION.
    categorical_cols : TYPE
        DESCRIPTION.
    use_oversampling : TYPE, optional
        DESCRIPTION. The default is False.
    rule_algorithm : TYPE, optional
        DESCRIPTION. The default is "".
    path : TYPE, optional
        DESCRIPTION. The default is "".
    file_name : TYPE, optional
        DESCRIPTION. The default is "".

    Returns
    -------
    df_rules_inliers : TYPE
        DESCRIPTION.
    df_rules_outliers : TYPE
        DESCRIPTION.
    df_no_pruned : TYPE
        DESCRIPTION.
    df_yes_pruned : TYPE
        DESCRIPTION.

    """

    
    feature_cols = numerical_cols + categorical_cols
    X = df_anomalies[feature_cols]
    
    ### Rules for Inliers
    y = df_anomalies[['predictions']]
    # y = y.apply(lambda x: 0 if x['predictions'] == 1 else 1, axis=1)
    
    from imblearn.over_sampling import SMOTE
    sm = SMOTE(random_state=42)
    X, y = sm.fit_resample(X, y['predictions'])
    
    dlc = DecisionListClassifier()
    dlc.fit(X, y)
    dlc_global = dlc.explain_global(name='Decision List Classifier')
    dct_dlc = dlc_global.data()
    
    
    df_rules = pd.DataFrame({'rules':dct_dlc['rule'],
                             'predictions':dct_dlc['outcome']})
    df_rules = df_rules[df_rules['rules'] != "No Rules Triggered"]
    df_aux = df_rules[df_rules['predictions']==1]
    list_rules_inliers = list(df_aux['rules'])
    list_rules_inliers = [x.replace(" and ", " & ") for x in list_rules_inliers]
    df_rules_inliers = turn_rules_to_df(df_anomalies, list_rules_inliers, feature_cols)
    df_rules_inliers['size_rules'] = [len(x.split("&")) for x in list_rules_inliers]
    
    df_aux = df_rules[df_rules['predictions']==-1]
    list_rules_outliers = list(df_aux['rules'])
    list_rules_outliers = [x.replace(" and ", " & ") for x in list_rules_outliers]
    df_rules_outliers = turn_rules_to_df(df_anomalies, list_rules_outliers, feature_cols)
    
    ### Check datapoints inside rules
    # Check datapoints inside hypercubes (inliers)
    print("Checking inliers inside hypercubes...")
    df_rules_inliers['n_inliers_included'] = 0
    df_rules_outliers['n_inliers_included'] = 0
    
    for i, data_point in df_anomalies[df_anomalies['predictions']==1].iterrows():
        df_rules_inliers['n_inliers_included'] += check_datapoint_inside(data_point,
                                                                         df_rules_inliers,
                                                                         feature_cols,
                                                                         [])['check']
        
        df_rules_outliers['n_inliers_included'] += check_datapoint_inside(data_point,
                                                                         df_rules_outliers,
                                                                         feature_cols,
                                                                         [])['check']
    
    ### Check datapoints inside hypercube (outliers)
    print("Checking outliers inside hypercubes...")
    df_rules_inliers['n_outliers_included'] = 0
    df_rules_outliers['n_outliers_included'] = 0
    
    for i, data_point in df_anomalies[df_anomalies['predictions']==-1].iterrows():   
        df_rules_inliers['n_outliers_included'] += check_datapoint_inside(data_point,
                                                                          df_rules_inliers,
                                                                          feature_cols,
                                                                          [])['check']
        
        df_rules_outliers['n_outliers_included'] += check_datapoint_inside(data_point,
                                                                           df_rules_outliers,
                                                                           feature_cols,
                                                                           [])['check']
        
    # Check how many datapoints are included with the rules with Precision=1
    print("Checking inliers/outliers inside hypercubes with Precision=1...")
    n_inliers_p1 = 0
    n_inliers_p0 = 0
    n_outliers_p1 = 0
    n_outliers_p0 = 0
    n_inliers = len(df_anomalies[df_anomalies['predictions']==1])
    n_outliers = len(df_anomalies[df_anomalies['predictions']==-1])
        
    for i, data_point in df_anomalies.iterrows():
        df_rules_inliers['check'] = check_datapoint_inside(data_point,
                                                           df_rules_inliers,
                                                           feature_cols,
                                                           [])['check']
        
        df_rules_outliers['check'] = check_datapoint_inside(data_point,
                                                           df_rules_outliers,
                                                           feature_cols,
                                                           [])['check']
        # If inlier
        if data_point['predictions']==1:
            # Rules with any P and that include this datapoint
            df_aux = df_rules_inliers[(df_rules_inliers['check']==1)] 
            if len(df_aux) > 0:
                n_inliers_p0 += 1
            
            # Rules with P=1 and that include this datapoint
            df_aux = df_rules_inliers[(df_rules_inliers['n_outliers_included']==0)
                                       & (df_rules_inliers['check']==1)] 
            if len(df_aux) > 0:
                n_inliers_p1 += 1
            
        # If outlier
        elif data_point['predictions']==-1:
            # Rules with any P and that include this datapoint
            df_aux = df_rules_outliers[(df_rules_outliers['check']==1)] 
            if len(df_aux) > 0:
                n_outliers_p0 += 1
            
            # Rules with P=1 and that include this datapoint
            df_aux = df_rules_outliers[(df_rules_outliers['n_inliers_included']==0)
                                       & (df_rules_outliers['check']==1)] 
            if len(df_aux) > 0:
                n_outliers_p1 += 1
    
    df_rules_inliers['n_inliers'] = n_inliers
    df_rules_inliers['n_inliers_p0'] = n_inliers_p0
    df_rules_inliers['n_inliers_p1'] = n_inliers_p1
    df_rules_outliers['n_outliers_p1'] = n_outliers_p1
    df_rules_outliers['n_outliers_p0'] = n_outliers_p0
    df_rules_outliers['n_outliers'] = n_outliers
    del df_rules_inliers['check'], df_rules_outliers['check']
    
    # Save to CSV
    df_rules_outliers.to_csv("{path}/{file_name}_rules_outliers_DLC.csv".format(path=path, file_name=file_name), index=False)
    df_rules_inliers.to_csv("{path}/{file_name}_rules_inliers_DLC.csv".format(path=path, file_name=file_name), index=False)
    
    # Prune rules
    df_yes_pruned = df_rules_outliers[(df_rules_outliers['n_inliers_included'] == 0) &
                                      (df_rules_outliers['n_outliers_included'] > 0)]
    df_yes_pruned = df_yes_pruned.reset_index(drop=True)
    
    if len(df_yes_pruned) > 1:
        df_yes_pruned = simplify_rules_alt([], df_yes_pruned).drop_duplicates()
        
    df_no_pruned = df_rules_inliers[(df_rules_inliers['n_inliers_included'] > 0) &
                                    (df_rules_inliers['n_outliers_included'] == 0)]
    df_no_pruned = df_no_pruned.reset_index(drop=True)
    
    if len(df_no_pruned) > 1:
        df_no_pruned = simplify_rules_alt([], df_no_pruned).drop_duplicates()
        
    # Obtain additional metrics
    if True:
        print("Obtaining additional metrics...")
        coeff = 1000
        ### Overlapping
        df_rules = df_rules_inliers.append(df_rules_outliers)
        df_dummy = df_rules.copy().replace(np.inf, 0).replace(-np.inf, 0) # discard infinities
        max_dummy = df_dummy[[col for col in list(df_rules.columns) if '_max' in col]].max()
        max_dummy = [x for x in list(max_dummy.values) if x != np.inf][0]*coeff # arbitrary large value
        min_dummy = df_dummy[[col for col in list(df_rules.columns) if '_min' in col]].min() # arbitrary low value
        min_dummy = [x for x in list(min_dummy.values) if x != -np.inf][0]
        min_dummy = min_dummy*coeff if min_dummy < 0 else min_dummy/coeff
         
        df_yes_pruned = df_yes_pruned.drop_duplicates().reset_index(drop=True)
        df_yes_pruned = df_yes_pruned.replace(np.inf, max_dummy)
        df_yes_pruned = df_yes_pruned.replace(-np.inf, min_dummy)
        df_yes_pruned = rule_overlapping_score(df_yes_pruned, df_anomalies,
                                               feature_cols,
                                               [])
        
        df_no_pruned = df_no_pruned.drop_duplicates().reset_index(drop=True)
        df_no_pruned = df_no_pruned.replace(np.inf, max_dummy)
        df_no_pruned = df_no_pruned.replace(-np.inf, min_dummy)
        df_no_pruned = rule_overlapping_score(df_no_pruned, df_anomalies,
                                               feature_cols,
                                               [])
        
        ### Stability
        df_yes_pruned = check_stability(df_anomalies, df_yes_pruned, model,
                                        feature_cols,
                                        using_inliers = False)
        df_no_pruned = check_stability(df_anomalies, df_no_pruned, model,
                                        feature_cols,
                                        using_inliers = True)
        
        # Replace with original limits
        df_yes_pruned = df_yes_pruned.replace(max_dummy, np.inf)
        df_yes_pruned = df_yes_pruned.replace(min_dummy, -np.inf)
        df_no_pruned = df_no_pruned.replace(max_dummy, np.inf)
        df_no_pruned = df_no_pruned.replace(min_dummy, -np.inf)
        
    # Save to CSV
    df_yes_pruned.to_csv("{path}/{file_name}_rules_outliers_pruned_DLC.csv".format(path=path, file_name=file_name),
                                index=False)
    df_no_pruned.to_csv("{path}/{file_name}_rules_inliers_pruned_DLC.csv".format(path=path, file_name=file_name),
                               index=False)

    return (df_rules_inliers, df_rules_outliers,
            df_no_pruned, df_yes_pruned)
    
    
    