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
# import ruleset
from lib.external import ruleset
import arff
import time
import pickle

import six
sys.modules['sklearn.externals.six'] = six

from joblib import Parallel, delayed
from sklearn import tree
from sklearn.tree import export_text
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
from sklearn.tree import _tree

N_JOBS = 1

def save_df_as_arff(df, folder='', file_name=''):
    
    arff.dump('{0}/{1}.arff'.format(folder, file_name)
          , df.values
          , relation='relation name'
          , names=df.columns)

def check_datapoint_inside(
    data_point, df_rules, numerical_cols, categorical_cols, check_opposite=True
):
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

    if len(df_rules) == 0:
        df_plot["check"] = 0
        return df_plot
    # Default value
    df_plot["check"] = 1

    # Check for categorical
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            value = data_point[col]
            df_plot["check"] = df_plot["check"] * (
                df_plot.apply(lambda x: 1 if (x[col] == value) else 0, axis=1)
            )
    # Check for numerical
    if len(numerical_cols) > 0:
        for col in numerical_cols:
            value = data_point[col]
            if check_opposite:
                df_plot["check"] = df_plot["check"] * (
                    df_plot.apply(
                        lambda x: 1
                        if ((x[col + "_max"] >= value) & (value >= x[col + "_min"]))
                        else 0,
                        axis=1,
                    )
                )
            else:
                df_plot["check"] = df_plot["check"] * (
                    df_plot.apply(
                        lambda x: 1
                        if ((x[col + "_max"] > value) & (value > x[col + "_min"]))
                        else 0,
                        axis=1,
                    )
                )
    return df_plot



def check_datapoint_inside_only(
    data_point, df_rules, numerical_cols, categorical_cols, check_opposite=True
):
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

    if len(df_rules) == 0:
        df_plot["check"] = 0
        return df_plot
    # Default value
    df_plot["check"] = 1

    # Check for categorical
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            value = data_point[col]
            df_plot["check"] = df_plot["check"] * (
                df_plot.apply(lambda x: 1 if (x[col] == value) else 0, axis=1)
            )
    # Check for numerical
    if len(numerical_cols) > 0:
        for col in numerical_cols:
            value = data_point[col]
            if check_opposite:
                df_plot["check"] = df_plot["check"] * (
                    df_plot.apply(
                        lambda x: 1
                        if ((x[col + "_max"] >= value) & (value >= x[col + "_min"]))
                        else 0,
                        axis=1,
                    )
                )
            else:
                df_plot["check"] = df_plot["check"] * (
                    df_plot.apply(
                        lambda x: 1
                        if ((x[col + "_max"] > value) & (value > x[col + "_min"]))
                        else 0,
                        axis=1,
                    )
                )
    return df_plot[["check"]]



def _dt_rules(clf, df_mat):
    
    tree = clf
    feature_names = list(df_mat.columns)
    class_names = "target_class"
    
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    df_rules = pd.DataFrame()
    # rules = []
    for path in paths:
        
        # Prediction
        classes = path[-1][0][0]
        pred = np.argmax(classes)
        
        if pred == 0:
            pred = -1 # inliers
        else:
            pred = 1 # outliers
        
        # Path rule
        rule = ""
        for p in path[:-1]:
            rule += str(p)
            if p != path[-2:-1]:
                rule += " & "
        rule = rule.replace("(", "").replace(")", "")
        df_rules = df_rules.append(
            pd.DataFrame(
                {'rule': [rule],
                 'prediction': [pred],
                 'len_rule': [len(path[:-1])]
                 }
                )
            )
    
    return df_rules



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
    
    # Obtain Rules
    r = export_text(clf, feature_names=list(df_mat.columns))
    r_s = r.split("class: ")
    leaf_yes = [x for x in r_s if x[0:2] == "-1"]
    leaf_no = [x for x in r_s if x[0] == "1"]
    print("tree rules not anomalies: ", len(leaf_yes))
    print("tree rules anomalies: ", len(leaf_no))

    df_rules = _dt_rules(clf, df_mat)
    df_rules = df_rules.rename(columns={"len_rule": "size_rules"})
    return df_rules


def dt_rules_v0(clf, df_mat):
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


def turn_rules_to_df(df_anomalies, list_rules, list_cols):
    """
    Function to transform the results from dt_rules() into a dataframe with the
    max and min values for each feature.
    
    There are only two limis per feature (max and min). If there are not enough
    information on the rule to obtain both values (p.e. rule = "x > 10") then 
    a default value is applied over the missing limit (-np.inf for min and np.inf
    for max; p.e. "x > 10" turns to "x_min = 10, x_max = np.inf").
    If there are duplicated information, the limits keep the strictest value
    (p.e. "x > 10 & x > 8 & x < 30" turns to "x_max = 30, x_min = 10").

    Parameters
    ----------
    df_anomalies : TYPE
        DESCRIPTION.
    list_rules : TYPE
        DESCRIPTION.
    list_cols : TYPE
        DESCRIPTION.

    Returns
    -------
    df_rules : TYPE
        DESCRIPTION.

    """
    
    df_rules = pd.DataFrame()
    
    if len(list_rules)==0:
        print("Warning: Rule list is empty: returning a DF without Rules")
        return  pd.DataFrame({col+'_max':[] for col in list_cols}).append(pd.DataFrame({col+'_min':[] for col in list_cols}))
    
    # Iter for each rule
    for rule in list_rules:
        dct_aux = {}
        
        # Default values
        for col in list_cols:
            dct_aux[col + '_max'] = np.inf
            dct_aux[col + '_min'] = -np.inf
        
        list_subrules = rule.split("&")
        
        # Iter for each component of the rule and obtain the limits
        for subrule in list_subrules:
            for col in list_cols:
                
                if col in subrule:
                    if ">=" in subrule:
                        aux = subrule.split(">=")
                        if np.float(aux[1]) >= dct_aux[col + '_min']:
                            dct_aux[col + '_min'] = np.float(aux[1])
 
                    elif ">" in subrule:
                        aux = subrule.split(">")
                        if np.float(aux[1]) >= dct_aux[col + '_min']:
                            dct_aux[col + '_min'] = np.float(aux[1])

                    if "<=" in subrule:
                        aux = subrule.split("<=")
                        if np.float(aux[1]) <= dct_aux[col + '_max']:
                            dct_aux[col + '_max'] = np.float(aux[1])
 
                    elif "<" in subrule:
                        aux = subrule.split("<") 
                        if np.float(aux[1]) <= dct_aux[col + '_max']:
                            dct_aux[col + '_max'] = np.float(aux[1])
 
        df_rules = df_rules.append(pd.DataFrame(dct_aux, index=[0]))
    
    return df_rules

def file_naming_ocsvm(file_template, cluster_algorithm, method, use_inverse):
    """
    Parameters
    ----------
    file_template : TYPE
        DESCRIPTION.
    cluster_algorithm : TYPE
        DESCRIPTION.
    method : TYPE
        DESCRIPTION.
    use_inverse : TYPE
        DESCRIPTION.

    Returns
    -------
    file_name : TYPE
        DESCRIPTION.

    """
    if use_inverse: file_name = file_template + "_culstering_{cluster_algorithm}_outliers_method_{method}".format(cluster_algorithm=cluster_algorithm,
                                                                                                                  method=method)
    else: file_name = file_template + "_culstering_{cluster_algorithm}_inliers_method_{method}".format(cluster_algorithm=cluster_algorithm,
                                                                                                       method=method)
    return file_name


def plot_2D(df_rules, df_anomalies, folder = "", path_name=""):
    """
    Function to plot a 2D figure with the anomalies and the hypercubes. df_rules
    should be in a format like the one returned in turn_rules_to_df() but without
    np.inf values (they can be set instead to an arbitrary big/low enough value).

    Parameters
    ----------
    df_rules : TYPE
        DESCRIPTION.
    df_anomalies : TYPE
        DESCRIPTION.
    folder : TYPE, optional
        DESCRIPTION. The default is "".
    path_name : TYPE, optional
        DESCRIPTION. The default is "".

    Returns
    -------
    None.

    """
    
    ### Plot 2D
    plt.figure(figsize=(12, 8))
    
    # Add hypercubes
    for i in range(len(df_rules)):
        # Create a Rectangle patch
        x_1 = df_rules.iloc[i:i + 1]['gdenergy_min'].values[0]
        x_2 = df_rules.iloc[i:i + 1]['gdenergy_max'].values[0]
        y_1 = df_rules.iloc[i:i + 1]['gdpuls_min'].values[0]
        y_2 = df_rules.iloc[i:i + 1]['gdpuls_max'].values[0]
    
        # Add the patch to the Axes
        rect = patches.Rectangle(
            (x_1, y_1),
            x_2 - x_1,
            y_2 - y_1,
            linewidth=3,
            edgecolor='black',
            facecolor='none',
            zorder=15)
        currentAxis = plt.gca()
        currentAxis.add_patch(rect)
    
    # Plot points
    plt.plot(
        df_anomalies[df_anomalies['predictions'] == 1]['gdenergy'],
        df_anomalies[df_anomalies['predictions'] == 1]['gdpuls'],
        'o',
        color="blue",
        label='not anomaly',
        zorder=10)
    plt.plot(
        df_anomalies[df_anomalies['predictions'] == -1]['gdenergy'],
        df_anomalies[df_anomalies['predictions'] == -1]['gdpuls'],
        'o',
        color='red',
        label='anomaly',
        zorder=10)
    plt.legend(loc='upper left')
    plt.xlabel('gdenergy', fontsize=12)
    plt.ylabel('gdpuls', fontsize=12)
    
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, y1, y2))
    plt.title("Anomalies {0}".format(path_name))
    plt.savefig(folder + "/" + "plot_2D_{0}".format(path_name) + ".png")
    
    
def rule_overlapping_score(df_rules, df_anomalies, numerical_cols,
                           categorical_cols):
    """
    Function to measure "Diversity"; the rules are different with "few
    overlapping concepts". This is computed checking the area of the hypercubes 
    of the rules that overlaps with another one.
    
    The way to check this is by seeing the 2D planes of each hypercube (by keeping
    two degrees of freedom for the features in the hyperplane coordinates; n-2 features
    are maintained and the other two are changed between their max/min values in order
    to obtain the vertices of that 2D plane). Then, it is computed the area of the 
    2D planes for the rules that overlaps, adding for all possible 2D planes the total
    area overlapped for each rule.
    
    In order to compute a score, the features are normalized in order to have 
    values between 0 and 1.
    

    Parameters
    ----------
    df_rules : TYPE
        DESCRIPTION.
    df_anomalies : TYPE
        DESCRIPTION.
    numerical_cols : TYPE
        DESCRIPTION.
    categorical_cols : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    
    def sort_vertices(list_vertex):
        """
        TODO
        Sort values for the Shapely order needed 
        """
        list_x = list(set([v[0] for v in list_vertex]))
        list_y = list(set([v[1] for v in list_vertex]))
        
        result = [(min(list_x), max(list_y)), (max(list_x), max(list_y)),
                  (max(list_x), min(list_y)), (min(list_x), min(list_y))]
        return result
        
    list_cols = numerical_cols 
    # Obtain combinations of features to create the 2D planes
    comb_free_features = [c for c in combinations(list_cols, 2)]
    # Filter rules for each categorical combination state
    if len(categorical_cols) > 0:
        df_cat = df_rules[categorical_cols]
        df_cat_unique = df_cat.drop_duplicates()
    else:
        df_cat = df_rules[categorical_cols]
        df_cat_unique = df_cat.head(1) # Only one iter
    
    # Scale rules in order to treat all features equally
    df_rules_original = df_rules.copy()
    
    # If there is one rule, then there is no overlapping
    if len(df_rules_original) <= 1:
        df_rules_original['score'] = 0
        df_rules_original['n_intersects'] = 0
        
        return df_rules_original

    if len(numerical_cols):
        sc = MinMaxScaler()
        sc.fit_transform(df_anomalies[numerical_cols])
        cols_max = [x + '_max' for x in numerical_cols]
        cols_min = [x + '_min' for x in numerical_cols]
        df_rules[cols_max] = sc.transform(df_rules[cols_max])
        df_rules[cols_min] = sc.transform(df_rules[cols_min])

    # Obtain the vectors (dataframe) for each combination of 2 free features and n-2 fixed ones (n size hyperspace)
    df_return = pd.DataFrame()
    k = 0
    for i, row in df_cat_unique.iterrows():
        print("Iter {0}/{1}".format(k, len(df_cat_unique)))
        k += 1
        # Obtain sub-hypercube (not outliers)
        list_index = df_cat[df_cat[row.index] == row.values].dropna(
        ).index  # index for that sub-hypercube
        df_rules_sub = df_rules[(df_rules.index.isin(list_index))].copy()  # sub-hypercube
        
        # If no rules, skip
        if len(df_rules_sub)==0:
            continue
        
        # If len == 1, then no overlapping for this iter
        elif len(df_rules_sub)==1:
            df_final = pd.DataFrame({'rule_id':[df_rules_sub.index[0]],
                                     'score':[1],
                                     'n_intersects':[0],
                                     'n':[1]})
        
        # Generic case
        else:
            df_final = pd.DataFrame()
            for comb in comb_free_features:
                # Specify cols
                list_free = [col + '_max' for col in comb] + [col + '_min' for col in comb] # Cols to change
                cols_fixed = [col + '_max' for col in list_cols] + [col + '_min' for col in list_cols]
                cols_fixed = [col for col in cols_fixed if col not in list_free] # Cols to mantain
                # cols_fixed = cols_fixed + categorical_cols # categorical do not use _max or _min
                
                list_x1 = [comb[0] + '_max', comb[0] + '_min']
                list_x2 = [comb[1] + '_max', comb[1] + '_min']
                cols_free = list(product(list_x1, list_x2))
                 # Vertices of the 2D planes
                comb_vertices = [tuple(list(x) + [j for j in cols_fixed]) for x in cols_free]
                
                # Obtain polygons for those 2D planes
                list_aux = [[tuple(vector[list(x)].values) for x in comb_vertices] for _,vector in df_rules_sub[list_free+cols_fixed].iterrows()]
                list_aux = [sort_vertices(list_vertex) for list_vertex in list_aux]
                polys = [Polygon(x) for x in list_aux]
                   
                # Compute intersections of parallel 2D planes
                df_polys = pd.DataFrame({"rule_id":list(df_rules_sub.index),
                                         "rule_vertices":[x.bounds for x in polys]})
                df_results = pd.DataFrame({"rule_id":[pair[0] for pair in permutations(list(df_rules_sub.index), 2)],
                                           "score":[pair[0].intersection(pair[1]).area for pair in permutations(polys, 2)]})
    
                # Annotate the rules with intersections in this 2D subspace
                df_results['n_intersects'] = df_results.apply(lambda x: 0 if x['score']==0 else 1, axis=1)
                
                # Obtain % of area
                ref_d = [pair[0].intersection(pair[0]).area for pair in permutations(polys, 2)]
                ref_q = [1 if x == 0 else x for x in ref_d] # set to 1 to divide if it's 0
                df_results['score'] = abs(ref_d - df_results['score'])/ref_q # If score=0 and no rule overlapping means that the area for that rule is 0
                # if max(df_results['score'])>1:break

                # Keep iter results
                if df_final.empty:
                    n = 1
                    df_results['n'] = n
                    df_final = df_results
                else:
                    n += 1
                    df_final['n'] = n
                    # df_final = df_final.append(df_results).groupby(['rule_id']).agg({'score':"sum",
                    #                                                                 "n_intersects":"sum",
                    #                                                                 "n":"max"}).reset_index()
                    # df_final = df_final.merge(df_results, left_on=['rule_id'], right_on=['rule_id'])
                    df_final['score'] += df_results['score']
                    df_final['n_intersects'] += df_results['n_intersects']
        
        df_return = df_return.append(df_final)
        
    df_return['n'] = df_return.apply(lambda x: 1 if x['n']==0 else x['n'], axis=1)
    df_return['score'] = df_return['score']/df_return['n']
    df_return.drop(['n'], axis=1, inplace=True)
    df_return = df_return.groupby(by=['rule_id']).agg({"score":"mean",
                                                       "n_intersects":"sum"}).reset_index()
        
    df_rules_original = df_rules_original.reset_index().merge(df_return,
                                                              right_on=['rule_id'],
                                                              left_on=['index'])
    df_rules_original = df_rules_original.drop(['index'], axis=1)
    df_rules_original['score'] = df_rules_original['score'].round(2)
    
    return df_rules_original

    
def check_stability(df_anomalies, df_rules, model, numerical_cols,
                    categorical_cols, using_inliers):
    """
    Function that computes the "stability" metrics of the hypercubes. 
    First, it obtains the prototypes from the dataset and generates random samples
    near them.
    
    Then, it obtains the prediction of the original model for those dummy samples
    and checks if when the prediction is inlier/outlier, there is at least one rule
    that includes that datapoint within it.
    
    It also checks the level of agreement between all the rules. Since the prototypes
    belong to the same class, the function checks if the final prediction using all 
    rules is the same for all the prototypes.
    
    Rules agreement:
        - Choose N prototypes that represent the original hyperspace of data
        - Generate M samples close to each of those N prototypes; the hypothesis
        is that close points should be generally predicted belonging to the same class
        - For each of those N*M datapoints (M datapoints per each N prototype) check
        whether the rules (all of them) predict them as inliner or outlier; the datapoints
        that come into the function are either outliers or inliers. If they are inliers, 
        then the rules identify an artificial datapoint (of those M*N) as inlier if it
        is outside every rule. If the datapoints are outliers it's the same reversed: a
        datapoint is an inlier if no rule includes it.
        - Then, it is checked the % of datapoints labeled as the assumed correct class (inliers or 
        outliers), neighbours of that prototype compared to the total neighbours of that prototype.
        - All the % for each prototype are averaged into one %.
    
    Model agreement:
        - The % of predictions for the artificial datapoints aforementioned that are the same
        between the rules and the original OCSVM model.

    Parameters
    ----------
    df_anomalies : TYPE
        DESCRIPTION.
    df_rules : TYPE
        DESCRIPTION.
    model : TYPE
        DESCRIPTION.
    numerical_cols : TYPE
        DESCRIPTION.
    categorical_cols : TYPE
        DESCRIPTION.
    sc : TYPE
        DESCRIPTION.
    using_inliers : TYPE
        DESCRIPTION.

    Returns
    -------
    df_rules : TYPE
        DESCRIPTION.

    """
    
    # Ignore prints in this function
    ff = open(os.devnull, 'w')
    xx = sys.stdout # save sys.stdout
    sys.stdout = ff
    
    if len(df_rules)==0:
        df_rules['precision_vs_model'] = 0
        df_rules['rules_agreement'] = 0
        
        return df_rules
    
    # Choose the type of datapoints and define params
    label = 1 if using_inliers else -1
    df_data = df_anomalies[df_anomalies['predictions']==label].copy()
    n_samples = np.round(len(df_rules))
    n_samples = n_samples if n_samples > 20 else 20 # at least 20 samples
    
    if n_samples > len(df_data):
        n_samples = len(df_data)
    
    df_rules_aux = df_rules.copy()
    df_anomalies_aux = df_anomalies.copy()
    
    # Scaling
    if len(numerical_cols):
        sc = StandardScaler()
        sc.fit_transform(df_anomalies[numerical_cols])
        # df_data[numerical_cols] = sc.transform(df_data[numerical_cols])
        df_anomalies_aux[numerical_cols] = sc.transform(df_anomalies_aux[numerical_cols])
        cols_max = [x + '_max' for x in numerical_cols]
        cols_min = [x + '_min' for x in numerical_cols]
        # df_rules_aux[cols_max] = sc.transform(df_rules_aux[cols_max])
        # df_rules_aux[cols_min] = sc.transform(df_rules_aux[cols_min])
        # df_rules_aux = df_rules_aux.astype(float)
    
    # Generate Prototypes
    explainer = ProtodashExplainer()
    list_cols = numerical_cols + categorical_cols
    (W, S, _) = explainer.explain(df_data[list_cols].values,
                                  df_data[list_cols].values,
                                  m=n_samples,
                                  kernelType='Gaussian',
                                  sigma=2)
    df_prototypes = df_anomalies[df_anomalies.index.isin(list(S))][list_cols].reset_index(drop=True)
    
    # Generate artificial samples around the prototypes
    df_samples_total = pd.DataFrame()
    base_size = len(df_anomalies)
    for i, row in df_prototypes.iterrows():
        iter_size = np.round(base_size*W[i])
        iter_size = iter_size if iter_size > 10 else 10
        iter_size = int(np.round(iter_size))
        df_samples = pd.DataFrame({col:np.random.uniform(low=row[col]*0.9, high=row[col]*1.1, size=(iter_size,)) for col in numerical_cols})
        if len(categorical_cols)>0:
            for col in categorical_cols:
                df_samples[col] = row[col]
        df_samples['prototype_id'] = i
        df_samples_total = df_samples_total.append(df_samples)
        
    df_samples_unscaled = df_samples_total.copy()
    df_samples_scaled = df_samples_total.copy()
    df_samples_scaled[numerical_cols] = sc.transform(df_samples_scaled[numerical_cols])
        
    # Check two things:
    # 1) Classifications are the same for all similar neighbours of a prototype
    # 2) Rules give the same output as the original model (-1 if not included in any rule, 1 if included in at least one of them)
    list_proto_id = list(df_samples_total['prototype_id'].unique())
    precision_rules = 0
    df_agree = pd.DataFrame()
    j = 0 # Number of datapoints inside the rules
    for proto_id in list_proto_id:
        print(proto_id)
        df_proto_subset_scaled = df_samples_scaled[df_samples_scaled['prototype_id']==proto_id][list_cols]
        df_proto_subset_unscaled = df_samples_unscaled[df_samples_unscaled['prototype_id']==proto_id][list_cols]
        
        for row_scaled, row_unscaled in zip(df_proto_subset_scaled.iterrows(), df_proto_subset_unscaled.iterrows()):
            i = row_scaled[0]
            data_point_scaled = row_scaled[1]
            data_point_unscaled = row_unscaled[1]
            df_aux = pd.DataFrame(check_datapoint_inside(data_point_unscaled, df_rules, numerical_cols, categorical_cols)['check'])
            
            # Only if the prediction of this datapoint belongs to the same class...
            rules_prediction = df_aux['check'].max()
            df_agree = df_agree.append(pd.DataFrame({'proto_id':proto_id,
                                                     'rules_prediction':rules_prediction}, index=[0]))

            # Check if the predictions are the same as the model
            y_model = model.predict(data_point_scaled.values.reshape(1, -1))[0]
            if using_inliers:
                # Model=inlier, Rules=inlier -> correct
                if df_aux['check'].max() == 1 and y_model==1:
                    j += 1
                    precision_rules += 1 # If inside any rule, check as correct if the model also predicted it
                # Model=outlier, Rules=Inlier -> incorrect
                elif df_aux['check'].max() == 1:
                    j += 1
                # Model=outlier, Rules=outlier -> correct
                elif df_aux['check'].max() == 0 and y_model==-1:
                    j += 1
                    precision_rules += 1 # If outside any rule, check as correct if the model also predicted it as outlier
                # Model=inlier, Rules=outlier -> incorrect
                elif df_aux['check'].max() == 0 and y_model==1:
                    j += 1
                    
            else:
                # Model=outlier, Rules=outlier -> correct
                if df_aux['check'].max() == 1 and y_model==-1:
                    j += 1
                    precision_rules += 1 # If inside any rule, check as correct if the model also predicted it
                # Model=inlier, Rules=outlier -> incorrect
                elif df_aux['check'].max() == 1:
                    j += 1
                # Model=inlier, Rules=inlier -> correct
                elif df_aux['check'].max() == 0 and y_model==1:
                    j += 1
                    precision_rules += 1 # If outside any rule, check as correct if the model also predicted it as inlier
               # Model=outlier, Rules=Inlier -> incorrect
                elif df_aux['check'].max() == 0 and y_model==-1:
                    j += 1
    
    rules_0 = (df_agree[df_agree['rules_prediction']==0]
               .groupby(by=['proto_id'])
               .count()
               .reset_index()
               .rename(columns={'rules_prediction':'rules_0'}))
    rules_1 = (df_agree[df_agree['rules_prediction']==1]
               .groupby(by=['proto_id'])
               .count()
               .reset_index()
               .rename(columns={'rules_prediction':'rules_1'}))
    rules_agreement = rules_0.merge(rules_1, how="outer").fillna(0)
    rules_agreement['per_agree'] = rules_agreement.apply(lambda x: max([x['rules_0'], x['rules_1']])/(x['rules_0']+x['rules_1']), axis=1)
    j = j if j != 0 else 1
    precision_vs_model = precision_rules/j # % of points with the same values as the model 
    final_agreement = np.round(np.mean(rules_agreement['per_agree']), 4)
    
    df_rules['precision_vs_model'] = precision_vs_model # % of datapoints with the same prediction as the original model
    df_rules['rules_agreement'] = final_agreement # % of agreement bewtween rules.
    df_rules['precision_vs_model'] = df_rules['precision_vs_model'].round(2)
    
    # Revert print
    sys.stdout = xx
    
    return df_rules


    
def surrogate_dt_rules(df_anomalies, model, numerical_cols,
                       categorical_cols, path="",  file_name=""):
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
    if True:
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
    if path != "":
        df_yes_pruned.to_csv("{path}/{file_name}_rules_outliers_pruned_DT.csv".format(path=path, file_name=file_name),
                                    index=False)
        df_no_pruned.to_csv("{path}/{file_name}_rules_inliers_pruned_DT.csv".format(path=path, file_name=file_name),
                                   index=False)
    
    
    return (df_rules_inliers, df_rules_outliers,
            df_no_pruned, df_yes_pruned)
    


def ocsvm_rules_completion(df_anomalies, df_rules, numerical_cols, 
                           categorical_cols, inliers_used=True,
                           clustering_algorithm="kmeans", path="",  file_name=""):
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
    
    # print("Checking inliers inside rules...")
    from datetime import datetime
    # time_ini = datetime.now()
    # df_check = df_anomalies[df_anomalies['predictions']==1]
    # l = df_check.apply(lambda x: check_datapoint_inside_only(x, df_rules,numerical_cols,categorical_cols), axis=1)
    # print("Execution time {0}(s)".format(datetime.now() - time_ini))
    
    # import swifter
    # time_ini = datetime.now()
    # df_check = df_anomalies[df_anomalies['predictions']==1]
    # l = df_check.swifter.apply(lambda x: check_datapoint_inside_only(x, df_rules,numerical_cols,categorical_cols), axis=1)
    # print("Execution time {0}(s)".format(datetime.now() - time_ini))

    time_ini = datetime.now()
    df_check = Parallel(n_jobs=N_JOBS)(delayed(check_datapoint_inside_only)(data_point,df_rules,numerical_cols,categorical_cols) for i, data_point in df_anomalies[df_anomalies['predictions']==1].iterrows())
    print("Execution time {0}(s)".format(datetime.now() - time_ini))
    
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
    df_rules.to_csv("{path}/{file_name}_rules_{type_r}_pruned_ocsvm.csv".format(path=path,
                                                                                file_name=file_name,
                                                                                type_r = path_aux),
                                index=False)

    return df_rules
    
    
def anchors_rules(df_anomalies, numerical_cols, categorical_cols,
                  model, scaler, path="",  file_name=""):
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
    path : TYPE, optional
        DESCRIPTION. The default is "".
    file_name : TYPE, optional
        DESCRIPTION. The default is "".

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
    
    # from nltk.classify import SklearnClassifier
    # model_new = SklearnClassifier(clf)

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
    
    # list_rules_anchors_yes = [str(j) for j in list_rules_anchors_yes]
    # list_rules_anchors_no = [str(j) for j in list_rules_anchors_no]
    
    #clf.predict(df_scaled.head(1)[feature_cols])
    
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
    if True:
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
    df_yes_pruned.to_csv("{path}/{file_name}_rules_outliers_pruned_Anchors.csv".format(path=path, file_name=file_name),
                                index=False)
    df_no_pruned.to_csv("{path}/{file_name}_rules_inliers_pruned_Anchors.csv".format(path=path, file_name=file_name),
                               index=False)
    
    print("Process succsesfully finished!")
    
    return (list_rules_transformed_no, df_rules_anchors_no,
            df_rules_anchors_yes, list_rules_anchors_no,
            df_yes_pruned, df_no_pruned)


def rulefit_rules(df_anomalies, model, numerical_cols, categorical_cols,
                  path="",  file_name=""):
    """
    TODO

    Parameters
    ----------
    df_anomalies : TYPE
        DESCRIPTION.
    model: TYPE
        DESCRIPTION.
    numerical_cols : TYPE
        DESCRIPTION.
    categorical_cols : TYPE
        DESCRIPTION.
    path : TYPE, optional
        DESCRIPTION. The default is "".
    file_name : TYPE, optional
        DESCRIPTION. The default is "".

    Returns
    -------
    df_check : TYPE
        DESCRIPTION.
    df_rules_outliers : TYPE
        DESCRIPTION.
    df_rules_inliers : TYPE
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
    df_rules_outliers.to_csv("{path}/{file_name}_rules_outliers_pruned_RuleFit.csv".format(path=path, file_name=file_name), index=False)
    df_rules_inliers.to_csv("{path}/{file_name}_rules_inliers_pruned_RuleFit.csv".format(path=path, file_name=file_name), index=False)
    
    print("Process succsesfully finished!")
    return df_check, df_rules_outliers, df_rules_inliers
 
    
def skoperules_rules(df_anomalies, model, numerical_cols, categorical_cols,
                     path="",  file_name=""):
    """
    TODO

    Parameters
    ----------
    df_anomalies : TYPE
        DESCRIPTION.
    model: TYPE
        DESCRIPTION.
    numerical_cols : TYPE
        DESCRIPTION.
    categorical_cols : TYPE
        DESCRIPTION.
    path : TYPE, optional
        DESCRIPTION. The default is "".
    file_name : TYPE, optional
        DESCRIPTION. The default is "".

    Returns
    -------
    df_rules_info_inliers : TYPE
        DESCRIPTION.
    df_rules_info_outliers : TYPE
        DESCRIPTION.
    df_rules_inliers : TYPE
        DESCRIPTION.
    df_rules_outliers : TYPE
        DESCRIPTION.
    df_no_pruned : TYPE
        DESCRIPTION.
    df_yes_pruned : TYPE
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

    if len(rules)>0:
        print("Checking inliers inside hypercubes...")
        df_rules_info_inliers = pd.DataFrame({"rule":[v[0].replace(" and ", " & ") for v in rules],
                                          "precision":[v[1][0] for v in rules],
                                          "recall":[v[1][1] for v in rules],
                                          "n_points_correct":[v[1][2] for v in rules]})
        if not df_rules_info_inliers.empty:
            df_rules_info_inliers['size_rules'] = df_rules_info_inliers.apply(lambda x: len(x['rule'].split("&")), axis=1)
        else:
            df_rules_info_inliers['size_rules'] = 0
        rules = [v[0].replace(" and ", " & ") for v in rules]
        
        # Obtain rules in df format
        print("Turning Rules to hypercubes...")
        df_rules_inliers = turn_rules_to_df(df_anomalies,
                                            rules,
                                            feature_cols)
        df_rules_inliers = df_rules_inliers.reset_index(drop=True)
        df_rules_inliers['size_rules'] = df_rules_info_inliers['size_rules']
    
        df_rules_inliers['n_inliers_included'] = 0
        
        for i, data_point in df_anomalies[df_anomalies['predictions']==1].iterrows():
            df_rules_inliers['n_inliers_included'] += check_datapoint_inside(data_point,
                                                                             df_rules_inliers,
                                                                             feature_cols,
                                                                             [])['check']
        
        ### Check datapoints inside hypercube (outliers)
        print("Checking outliers inside hypercubes...")
        df_rules_inliers['n_outliers_included'] = 0
        
        for i, data_point in df_anomalies[df_anomalies['predictions']==-1].iterrows():   
            df_rules_inliers['n_outliers_included'] += check_datapoint_inside(data_point,
                                                                              df_rules_inliers,
                                                                              feature_cols,
                                                                              [])['check']
            
        # Check how many datapoints are included with the rules with Precision=1
        print("Checking inliers/outliers inside hypercubes with Precision=1...")
        n_inliers_p1 = 0
        n_inliers_p0 = 0
        n_inliers = len(df_anomalies[df_anomalies['predictions']==1])
        n_outliers = len(df_anomalies[df_anomalies['predictions']==-1])
        for i, data_point in df_anomalies.iterrows():
            df_rules_inliers['check'] = check_datapoint_inside(data_point,
                                                               df_rules_inliers,
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

        
        df_rules_inliers['n_inliers'] = n_inliers
        df_rules_inliers['n_inliers_p0'] = n_inliers_p0
        df_rules_inliers['n_inliers_p1'] = n_inliers_p1
        del df_rules_inliers['check']
        
        # Save to CSV
        df_rules_inliers.to_csv("{path}/{file_name}_rules_inliers_SkopeRules.csv".format(path=path, file_name=file_name), index=False)
            
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
            df_rules = df_rules_inliers
            df_dummy = df_rules.copy().replace(np.inf, 0).replace(-np.inf, 0) # discard infinities
            max_dummy = df_dummy[[col for col in list(df_rules.columns) if '_max' in col]].max()
            max_dummy = [x for x in list(max_dummy.values) if x != np.inf][0]*coeff # arbitrary large value
            min_dummy = df_dummy[[col for col in list(df_rules.columns) if '_min' in col]].min() # arbitrary low value
            min_dummy = [x for x in list(min_dummy.values) if x != -np.inf][0]
            min_dummy = min_dummy*coeff if min_dummy < 0 else min_dummy/coeff

            ### Stability
            df_no_pruned = check_stability(df_anomalies, df_no_pruned, model,
                                            feature_cols,
                                            [],
                                            using_inliers = True)
            
            # Replace with original limits
            df_no_pruned = df_no_pruned.replace(max_dummy, np.inf)
            df_no_pruned = df_no_pruned.replace(min_dummy, -np.inf)
            
        
        # Save to CSV
        df_no_pruned.to_csv("{path}/{file_name}_rules_inliers_pruned_SkopeRules.csv".format(path=path, file_name=file_name),
                                   index=False)
    
    
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
    
    
    if len(rules)>0:
        df_rules_info_outliers = pd.DataFrame({"rule":[v[0].replace(" and ", " & ") for v in rules],
                                      "precision":[v[1][0] for v in rules],
                                      "recall":[v[1][1] for v in rules],
                                      "n_points_correct":[v[1][2] for v in rules]})
        if not df_rules_info_outliers.empty:
            df_rules_info_outliers['size_rules'] = df_rules_info_outliers.apply(lambda x: len(x['rule'].split("&")), axis=1)
        else:
            df_rules_info_outliers['size_rules'] = 0
        
        rules = [v[0].replace(" and ", " & ") for v in rules]
        
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
        df_rules_outliers['n_inliers_included'] = 0
        
        for i, data_point in df_anomalies[df_anomalies['predictions']==1].iterrows():            
            df_rules_outliers['n_inliers_included'] += check_datapoint_inside(data_point,
                                                                             df_rules_outliers,
                                                                             feature_cols,
                                                                             [])['check']
        ### Check datapoints inside hypercube (outliers)
        print("Checking outliers inside hypercubes...")
        df_rules_outliers['n_outliers_included'] = 0
        
        for i, data_point in df_anomalies[df_anomalies['predictions']==-1].iterrows():   
            df_rules_outliers['n_outliers_included'] += check_datapoint_inside(data_point,
                                                                               df_rules_outliers,
                                                                               feature_cols,
                                                                               [])['check']
            
        # Check how many datapoints are included with the rules with Precision=1
        print("Checking inliers/outliers inside hypercubes with Precision=1...")
        n_outliers_p1 = 0
        n_outliers_p0 = 0
        n_inliers = len(df_anomalies[df_anomalies['predictions']==1])
        n_outliers = len(df_anomalies[df_anomalies['predictions']==-1])
            
        for i, data_point in df_anomalies.iterrows():
            df_rules_outliers['check'] = check_datapoint_inside(data_point,
                                                               df_rules_outliers,
                                                               feature_cols,
                                                               [])['check']                
            # If outlier
            if data_point['predictions']==-1:
                # Rules with any P and that include this datapoint
                df_aux = df_rules_outliers[(df_rules_outliers['check']==1)] 
                if len(df_aux) > 0:
                    n_outliers_p0 += 1
                
                # Rules with P=1 and that include this datapoint
                df_aux = df_rules_outliers[(df_rules_outliers['n_inliers_included']==0)
                                           & (df_rules_outliers['check']==1)] 
                if len(df_aux) > 0:
                    n_outliers_p1 += 1
                    
        df_rules_outliers['n_outliers_p1'] = n_outliers_p1
        df_rules_outliers['n_outliers_p0'] = n_outliers_p0
        df_rules_outliers['n_outliers'] = n_outliers
        del df_rules_outliers['check']
        
        # Save to CSV
        df_rules_outliers.to_csv("{path}/{file_name}_rules_outliers_SkopeRules.csv".format(path=path, file_name=file_name), index=False)
          
        # Prune rules
        df_yes_pruned = df_rules_outliers[(df_rules_outliers['n_inliers_included'] == 0) &
                                          (df_rules_outliers['n_outliers_included'] > 0)]
        df_yes_pruned = df_yes_pruned.reset_index(drop=True)
        
        if len(df_yes_pruned) > 1:
            df_yes_pruned = simplify_rules_alt([], df_yes_pruned).drop_duplicates()
            
        # Obtain additional metrics
        if True:
            print("Obtaining additional metrics...")
            coeff = 1000
            ### Overlapping
            df_rules = df_rules_outliers
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
            
            ### Stability
            df_yes_pruned = check_stability(df_anomalies, df_yes_pruned, model,
                                            feature_cols,
                                            [],
                                            using_inliers = False)
            
            # Replace with original limits
            df_yes_pruned = df_yes_pruned.replace(max_dummy, np.inf)
            df_yes_pruned = df_yes_pruned.replace(min_dummy, -np.inf)

        # Save to CSV
        df_yes_pruned.to_csv("{path}/{file_name}_rules_outliers_pruned_SkopeRules.csv".format(path=path, file_name=file_name),
                                    index=False)
        
        return (df_rules_info_inliers, df_rules_info_outliers,
                df_rules_inliers, df_rules_outliers,
                df_no_pruned, df_yes_pruned)
    
    else:
        return (None, None, None, None, None, None)
        
        
        
def frl_rules(df_anomalies, model, numerical_cols, categorical_cols,
              path="",  file_name=""):
    """
    Rules obtained using Bayesian Falling Rules List.

    Parameters
    ----------
    df_anomalies : TYPE
        DESCRIPTION.
    model: TYPE
        DESCRIPTION.
    numerical_cols : TYPE
        DESCRIPTION.
    categorical_cols : TYPE
        DESCRIPTION.
    path : TYPE, optional
        DESCRIPTION. The default is "".
    file_name : TYPE, optional
        DESCRIPTION. The default is "".

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
    df_yes_pruned.to_csv("{path}/{file_name}_rules_outliers_pruned_FRL.csv".format(path=path, file_name=file_name),
                                index=False)
    df_no_pruned.to_csv("{path}/{file_name}_rules_inliers_pruned_FRL.csv".format(path=path, file_name=file_name),
                               index=False)

    return (df_rules_inliers, df_rules_outliers,
            df_no_pruned, df_yes_pruned)



def aix360_rules_wrapper(df_anomalies, model, numerical_cols, categorical_cols,
                         use_oversampling=False, rule_algorithm="", path="",
                         file_name=""):
    """
    TODO

    Parameters
    ----------
    df_anomalies : TYPE
        DESCRIPTION.
    model: TYPE
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

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    df_rules_info_inliers : TYPE
        DESCRIPTION.
    df_rules_info_outliers : TYPE
        DESCRIPTION.
    df_rules_inliers : TYPE
        DESCRIPTION.
    df_rules_outliers : TYPE
        DESCRIPTION.
    df_no_pruned : TYPE
        DESCRIPTION.
    df_yes_pruned : TYPE
        DESCRIPTION.

    """

    
    # Define variables
    feature_cols = numerical_cols + categorical_cols
    X = df_anomalies[feature_cols].astype(float)
    y = df_anomalies['predictions'].astype(int)
    y_inliers = np.array([x if x > 0 else 0 for x in y]) # Defined for inliers levels
    y_outliers = np.array([1 if x < 0 else 0 for x in y]) # Defined for outlier levels
    
    # Perform oversampling
    X_inliers = X
    X_outliers = X
  
    # Feature binarize 
    print("Feature binarizer...")
    fb = FeatureBinarizer(negations=True, returnOrd=True, colsCateg=categorical_cols)
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
    df_yes_pruned.to_csv("{path}/{file_name}_rules_outliers_pruned_{rule_algorithm}.csv".format(path=path,
                                                                                                file_name=file_name,
                                                                                                rule_algorithm=rule_algorithm),
                                index=False)
    df_no_pruned.to_csv("{path}/{file_name}_rules_inliers_pruned_{rule_algorithm}.csv".format(path=path,
                                                                                              file_name=file_name,
                                                                                              rule_algorithm=rule_algorithm),
                               index=False)
    
    
    return (df_rules_inliers, df_rules_outliers,
            df_no_pruned, df_yes_pruned)


def interpretML_DecisionListClassifier(df_anomalies, model, numerical_cols, categorical_cols,
                                       use_oversampling=False, rule_algorithm="", path="",
                                       file_name=""):
    """

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
    

def rule_overlapping_score_new(df_rules, numerical_cols, categorical_cols):
    """
    Function to measure "Diversity"; the rules are different with "few
    overlapping concepts". This is computed checking the area of the hypercubes
    of the rules that overlaps with another one.

    The way to check this is by seeing the 2D planes of each hypercube (by keeping
    two degrees of freedom for the features in the hyperplane coordinates; n-2 features
    are maintained and the other two are changed between their max/min values in order
    to obtain the vertices of that 2D plane). Then, it is computed the area of the
    2D planes for the rules that overlaps, adding for all possible 2D planes the total
    area overlapped for each rule.

    In order to compute a score, the features are normalized in order to have
    values between 0 and 1.


    DEBUG:
        numerical_cols = ['gdenergy', 'gdpuls']
        categorical_cols = []
        df_rules = pd.DataFrame({'gdenergy_max':[50, 30],
                                 'gdenergy_min':[-20, -10],
                                 'gdpuls_max':[84, 100],
                                 'gdpuls_min':[-50, 20],
                                 'rule_id':[1, 2]
                                }
                                )
        # Area intersect: 64*40 = 2560
        # Area union: 9380 + 3200 - 2560
        # Jaccard: 2560 / (9380 + 3200 - 2560) = 0.255
        
    
    DEBUG 2:
        df_rules = pd.read_csv("df_multi_debug.csv")
        numerical_cols = ['seismic','seismoacoustic', 'genergy','gplus',
                          'gdenergy','gdpuls','bumps','bumps2','bumps3',
                          'bumps4', 'bumps5','bumps6', 'bumps7', 'bumps8',
                          'energy', 'maxenergy', 'class']
        categorical_cols = ['hazard_1', 'hazard_2', 'shift_1']
        
    DEGUB 3:
        df_rules = pd.read_csv("df_multi_debug2.csv")
        numerical_cols = ['seismic','seismoacoustic', 'genergy','gplus',
                          'gdenergy','gdpuls','bumps','bumps2','bumps3',
                          'bumps4', 'bumps5','bumps6', 'bumps7', 'bumps8',
                          'energy', 'maxenergy', 'class']
        categorical_cols = ['hazard_1_max', 'hazard_2_max', 'hazard_1_min',
                            'hazard_2_min', 'shift_1_max', 'shift_1_min']
        
    DEGUB 4:
        df_rules = pd.read_csv("df_multi_debug3.csv")
        numerical_cols = ['seismic', 'seismoacoustic', 'genergy','gplus',
                          'gdenergy','gdpuls','bumps', 'bumps2', 'bumps3',
                          'bumps4', 'bumps5','bumps6', 'bumps7', 'bumps8',
                          'energy', 'maxenergy', 'class']
        categorical_cols = ['hazard_1_max', 'hazard_1_min','hazard_2_max',
                            'hazard_2_min', 'shift_1_max', 'shift_1_min']
        
    DEGUB 5:
        df_rules = pd.read_csv("df_multi_debug4.csv")
        numerical_cols = ['hours_speed_control', 'fuel_idle_day',
                          'fuel_idle_accumulated', 'max_engine_cool_temp',
                          'max_engine_oil_temp','total_odometer','count_drive',
                          'reverse','count_neutral','count_park', 'count_forward',
                          'count_idle_events','idle_time','hour_drive', 'hour_park',
                          'hour_reverse', 'hour_forward','hour_neutral',
                          'count_harsh_brakes', 'speed_over_120', 'count_harsh_turns',
                          'count_jackrabbit', 'cruise_control_on','lights_left_on',
                          'engine_oil_variation', 'fuel_filter_life_variation',
                          'fuel_exhaust_fluid_variation','mean_transmission_oil_temp',
                          'mean_forward_acc', 'mean_braking_acc', 'mean_oil_temp',
                          'with_passenger','trip_fuel_used','trip_kms',
                          'max_tire_pressure_rl', 'max_tire_pressure_rr',
                          'max_tire_pressure_fl', 'max_tire_pressure_fr',
                          'rpm_high', 'rpm_low','rpm_medium_low', 'rpm_medium',
                          'rpm_medium_high', 'rpm_stopped','diff_tire_pressure_rl',
                          'diff_tire_pressure_fl', 'diff_tire_pressure_fr',
                          'diff_tire_pressure_rr', 'avg_fuel_consumption']
        categorical_cols = ['vehicle_group_1','vehicle_group_10',
                            'vehicle_group_12', 'vehicle_group_14',
                            'vehicle_group_2','vehicle_group_3', 
                            'vehicle_group_4', 'vehicle_group_5',
                            'vehicle_group_6', 'vehicle_group_7',
                            'vehicle_group_8','vehicle_group_9',
                            'vehicle_group_10','vehicle_group_12',
                            'vehicle_group_14']
        
    Parameters
    ----------
    df_rules : TYPE
        DESCRIPTION.
    numerical_cols : TYPE
        DESCRIPTION.
    categorical_cols : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    def sort_vertices(list_vertex):
        """
        TODO
        Sort values for the Shapely order needed
        """
        list_x = list(set([v[0] for v in list_vertex]))
        list_y = list(set([v[1] for v in list_vertex]))

        result = [
            (min(list_x), max(list_y)),
            (max(list_x), max(list_y)),
            (max(list_x), min(list_y)),
            (min(list_x), min(list_y)),
        ]
        return result

    list_cols = numerical_cols
    # Obtain combinations of features to create the 2D planes
    comb_free_features = [c for c in combinations(list_cols, 2)]
    # Filter rules for each categorical combination state
    if len(categorical_cols) > 0:
        df_cat = df_rules[categorical_cols]
        df_cat_unique = df_cat.drop_duplicates()
    else:
        df_cat_unique = pd.DataFrame({'dummy':[1]})  # Only one iter
    # Simple copy
    df_rules_original = df_rules.copy()

    # If there is one rule, then there is no overlapping
    if len(df_rules_original) <= 1:
        df_rules_original["score"] = 0
        df_rules_original["n_intersects"] = 0
        return df_rules_original
    
    # Obtain the vectors (dataframe) for each combination of 2 free features and n-2 fixed ones (n size hyperspace)
    df_return = pd.DataFrame()
    k = 0
    for i, row in df_cat_unique.iterrows():
        k += 1
        print("Iter {0}/{1}".format(k, len(df_cat_unique)))
        
        # If no categorical features, all rules together
        if not df_cat_unique.empty:
            # Obtain sub-hypercube (not outliers)
            list_index = (
                df_cat[df_cat[row.index] == row.values].dropna().index
            )  # index for that sub-hypercube
            df_rules_sub = df_rules[
                (df_rules.index.isin(list_index))
            ].copy()  # sub-hypercube
        else:
            df_rules_sub = df_rules

        # If no rules, skip
        if len(df_rules_sub) == 0:
            continue
        # If len == 1, then no overlapping for this iter
        elif len(df_rules_sub) == 1:
            df_final = pd.DataFrame(
                {
                    "rule_id": [df_rules_sub.index[0]],
                    "score": [1],
                    "n_intersects": [0],
                    "n": [1],
                }
            )
        # Generic case
        else:
            df_final = pd.DataFrame()
            for comb in comb_free_features:
                # Specify cols
                list_free = [col + "_max" for col in comb] + [
                    col + "_min" for col in comb
                ]  # Cols to change
                cols_fixed = [col + "_max" for col in list_cols] + [
                    col + "_min" for col in list_cols
                ]
                cols_fixed = [
                    col for col in cols_fixed if col not in list_free
                ]  # Cols to mantain
                # cols_fixed = cols_fixed + categorical_cols # categorical do not use _max or _min

                list_x1 = [comb[0] + "_max", comb[0] + "_min"]
                list_x2 = [comb[1] + "_max", comb[1] + "_min"]
                cols_free = list(product(list_x1, list_x2))
                # Vertices of the 2D planes
                comb_vertices = [
                    tuple(list(x) + [j for j in cols_fixed]) for x in cols_free
                ]

                # Obtain polygons for those 2D planes
                list_aux = [
                    [tuple(vector[list(x)].values) for x in comb_vertices]
                    for _, vector in df_rules_sub[list_free + cols_fixed].iterrows()
                ]
                list_aux = [sort_vertices(list_vertex) for list_vertex in list_aux]
                polys = [Polygon(x) for x in list_aux]

                # Compute intersections of parallel 2D planes
                df_polys = pd.DataFrame(
                    {
                        "rule_id": list(df_rules_sub.index),
                        "rule_vertices": [x.bounds for x in polys],
                    }
                )

                df_results = pd.DataFrame(
                    {
                        "rules": [c for c in combinations(list(df_rules_sub.index), 2)],
                        "area_inter": [
                            pair[0].intersection(pair[1]).area
                            for pair in combinations(polys, 2)
                        ],
                        "area_1": [pair[0].area for pair in combinations(polys, 2)],
                        "area_2": [pair[1].area for pair in combinations(polys, 2)],
                    }
                )
                                
                # If some area is 0, there is no overlapping
                df_results["rule_1"] = df_results.apply(lambda x: x["rules"][0], axis=1)
                df_results["rule_2"] = df_results.apply(lambda x: x["rules"][1], axis=1)
                df_results["area_union"] = (
                    df_results["area_1"]
                    + df_results["area_2"]
                    - df_results["area_inter"]
                )
                df_results["area_union"] = df_results.apply(
                    lambda x: 1 if x["area_union"] == 0 else x["area_union"], axis=1
                )
                df_results["jaccard"] = (
                    df_results["area_inter"] / df_results["area_union"]
                )
                df_results["score"] = 1 - df_results["jaccard"]

                # Annotate the rules with intersections in this 2D subspace
                df_results["n_intersects"] = df_results.apply(
                    lambda x: 0 if x["jaccard"] == 0 else 1, axis=1
                )
                
                # If score negative, set to 1 (no overlapping)
                df_results['score'] = df_results.apply(lambda x: 1 if x['score']<0
                                                       else x['score'],
                                                       axis=1)
                                                    
                # Keep iter results
                if df_final.empty:
                    n = 1
                    df_results["n"] = n
                    df_final = df_results
                else:
                    n += 1
                    df_final["n"] = n
                    df_final["score"] += df_results["score"]
                    df_final["n_intersects"] += df_results["n_intersects"]
                                        
        df_return = df_return.append(df_final)
    df_return["n"] = df_return.apply(lambda x: 1 if x["n"] == 0 else x["n"], axis=1)
    df_return["score"] = df_return["score"] / df_return["n"]
    df_return.drop(["n"], axis=1, inplace=True)
    # df_return = (
    #     df_return.groupby(by=["rule_id"])
    #     .agg({"score": "mean", "n_intersects": "sum"})
    #     .reset_index()
    # )

    return df_return
    
    