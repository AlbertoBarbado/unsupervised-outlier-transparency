# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 10:13:42 2019

@author: alber
"""

import numpy as np
import pandas as pd
import arff
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    
    
def save_df_as_arff(df, folder='', file_name=''):
    
    arff.dump('{0}/{1}.arff'.format(folder, file_name)
          , df.values
          , relation='relation name'
          , names=df.columns)


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
    
    if len(folder)>0:
        plt.savefig(folder + "/" + "plot_2D_{0}".format(path_name) + ".png")
    