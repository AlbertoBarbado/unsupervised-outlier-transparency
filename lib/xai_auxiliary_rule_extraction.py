# -*- coding: utf-8 -*-
"""
Created on Tue May 18 21:45:54 2021

@author: alber
"""
import sys, os
import pandas as pd
import numpy as np

# import ruleset
import time
import pickle
import six

sys.modules["sklearn.externals.six"] = six

from joblib import Parallel, delayed
from sklearn import tree
from sklearn.tree import export_text
from alibi.explainers import AnchorTabular
from rulefit import RuleFit
from skrules import SkopeRules
from lib.xai_rule_processing import simplifyRules
from aix360.algorithms.rbm import (
    BooleanRuleCG,
    FeatureBinarizer,
    LogisticRuleRegression,
    GLRMExplainer,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from interpret.glassbox import DecisionListClassifier
from lib.external import ruleset
from lib.xai_rule_processing import turn_rules_to_df

N_JOBS = 1


def generateRuleHypercubes(
    df_anomalies,
    numerical_cols,
    categorical_cols,
    method,
    simplify_rules=False,
    model_params={},
):

    """
        Function for applying directly rule extarction techniques
        over an anomaly prediction output. It generates a hypercube with the
        corresponding rules.

        Parameters
        ----------
        df_anomalies : dataframe
            Must contain the following columns:
                - features: one column per feature considered
                - 'dist': distance to the decision function. < 0 for outliers
                - 'predictions'':  (1 or -1): 1 for inliers, -1 for outliers
                - 'score': score for the predictions. the bigger the absolute value,
                the surer it its about the prediction. score < 0 for outliers.
        numerical_cols : list
            list of numerical columns.
        categorical_cols : list
            list of categorical columns.
        method : str:
            One of the methods from the following:
                ['DecisionTree', 'RuleFit', 'FRL']
        simplify_rules : boolean, optional
            whether to prune the rules generated or not. The default is False.
        model_params : dict, optional
            Parameters for the rule extraction algorithm used inside. The default is {}.

        Returns
        -------
        df_rules_inliers : dataframe
            Example output:
       gdenergy_max  gdenergy_min  gdpuls_max  gdpuls_min  size_rules  rule_prediction
    0           inf         -83.5         inf       -68.5           3                1
    1           inf         -81.5         inf       -71.0           3                1
    2           inf         -81.5         inf       -69.5           3                1

        df_rules_outliers : dataframe
            Example output:
       gdenergy_max  gdenergy_min  gdpuls_max  gdpuls_min  size_rules  rule_prediction
    0           inf          -inf        91.5        -inf           2               -1
    1         110.5         -80.5         inf       -76.0           3               -1
    2         115.5         -80.0        89.0        -inf           5               -1

    """

    list_methods = [
        "DecisionTree",
        "RuleFit",
        "FRL",
        "SkopeRules",
        "DecisionRuleList",
        "brlg",
        "logrr",
    ]

    if method not in list_methods:
        msg_err = "Method {0} not supported. " "Choose one from {1}".format(
            method, list_methods
        )
        raise ValueError(msg_err)
    ## Decision Tree
    if method == "DecisionTree":
        df_rules_inliers, df_rules_outliers = surrogate_dt_rules(
            df_anomalies=df_anomalies,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            simplify_rules=simplify_rules,
            model_params=model_params,
        )
    ## RuleFit
    elif method == "RuleFit":
        df_rules_inliers, df_rules_outliers = rulefit_rules(
            df_anomalies=df_anomalies,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            simplify_rules=simplify_rules,
            model_params=model_params,
        )
    ## FRL
    elif method == "FRL":
        df_rules_inliers, df_rules_outliers = frl_rules(
            df_anomalies=df_anomalies,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            simplify_rules=simplify_rules,
            model_params=model_params,
        )
    ## SkopeRules
    elif method == "SkopeRules":
        df_rules_inliers, df_rules_outliers = skoperules_rules(
            df_anomalies=df_anomalies,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            simplify_rules=simplify_rules,
            model_params=model_params,
        )
    ## Decision Rules List
    elif method == "DecisionRuleList":
        df_rules_inliers, df_rules_outliers = decisionListClassifier(
            df_anomalies=df_anomalies,
            numerical_cols=numerical_cols,
            categorical_cols=categorical_cols,
            simplify_rules=simplify_rules,
            model_params=model_params,
        )
    ## BRLG or LOGRR
    elif method == "brlg" or method == "logrr":
        df_rules_inliers, df_rules_outliers = aix360_rules_wrapper(
            df_anomalies,
            numerical_cols,
            categorical_cols,
            method,
            simplify_rules,
            model_params,
        )
    return df_rules_inliers, df_rules_outliers


def _dt_rules(clf, df_mat):
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
    df_splits = pd.DataFrame({"levels": list_splits})
    df_splits = (
        df_splits[df_splits["levels"] != ""].reset_index(drop=True).reset_index()
    )
    df_splits["index"] += 1

    df_rules = pd.DataFrame()

    for i, point in df_mat.iterrows():
        node_indices = clf.decision_path(point.values.reshape(1, -1))
        rule = ""
        node_indices = pd.DataFrame(node_indices.toarray().T).reset_index()
        node_indices = node_indices.merge(df_splits)
        node_indices = node_indices[node_indices[0] == 1]
        for i in list(node_indices["levels"]):
            if rule == "":
                rule = i
            else:
                rule = rule + " & " + i
        dct_aux = {
            "rule": rule,
            "prediction": clf.predict(point.values.reshape(1, -1)),
            "len_rule": len(node_indices),
        }

        df_rules = df_rules.append(pd.DataFrame(dct_aux, index=[0]))
    df_rules = df_rules.drop_duplicates()

    return df_rules


def surrogate_dt_rules(
    df_anomalies,
    numerical_cols,
    categorical_cols,
    simplify_rules=False,
    model_params={},
):
    """
    Rule Extraction based on Decision Trees

    Parameters
    ----------
    df_anomalies : TYPE
        DESCRIPTION.
    numerical_cols : TYPE
        DESCRIPTION.
    categorical_cols : TYPE
        DESCRIPTION.
    simplify_rules : TYPE, optional
        DESCRIPTION. The default is False.
    model_params : TYPE, optional
        DESCRIPTION. The default is {}.

    Returns
    -------
    df_rules_inliers : TYPE
        DESCRIPTION.
    df_rules_outliers : TYPE
        DESCRIPTION.

    """

    # Init
    y_real = df_anomalies["predictions"]
    feature_cols = numerical_cols + categorical_cols
    df_mat = df_anomalies[feature_cols]

    # Fit model (overfitted)
    clf = tree.DecisionTreeClassifier(**model_params)
    clf = clf.fit(df_anomalies[feature_cols], y_real)
    rules_tree = clf.tree_.value
    print("Depth tree: ", clf.tree_.max_depth)
    print("Nodes tree: ", clf.tree_.node_count)

    leaf_nodes = clf.tree_.value
    leaf_nodes_anomalies = [
        clf.classes_[np.argmax(x)]
        for x in leaf_nodes
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

    df_rules_original = _dt_rules(clf, df_mat[feature_cols])
    df_rules_original = df_rules_original.rename(columns={"len_rule": "size_rules"})

    # Check there are enough rules
    if len(df_rules_original) == 0:
        print("No rules extracted!")
        return None
    # Split inlier vs outlier rules
    df_inliers_original = df_rules_original[df_rules_original["prediction"] == 1].copy()
    df_outliers_original = df_rules_original[
        df_rules_original["prediction"] == -1
    ].copy()

    ### Inliers
    if len(df_inliers_original) > 0:
        # Turn list of rules to dataframe
        print("Turning rules to hypercubes...")
        list_rules_inliers = list(df_inliers_original["rule"])
        df_rules_inliers = turn_rules_to_df(
            list_rules=list_rules_inliers, list_cols=feature_cols
        )
        # Get corresponding rule size from the original rule extraction model,
        # not on the hypercubes obtained later
        df_rules_inliers["size_rules"] = list(df_inliers_original["size_rules"].values)

        # Prune rules
        if simplify_rules:
            print("Prunning the rules obtained...")
            df_rules_pruned = df_rules_inliers.drop(columns=["size_rules"]).copy()
            df_rules_pruned = simplifyRules(df_rules_pruned, categorical_cols)
            df_rules_pruned = df_rules_pruned.reset_index().merge(
                df_rules_inliers.reset_index()[["index", "size_rules"]], how="left"
            )
            df_rules_pruned.index = df_rules_pruned["index"]
            df_rules_pruned = df_rules_pruned.drop(columns=["index"], errors="ignore")
            df_rules_inliers = df_rules_pruned
        # Col predicted
        df_rules_inliers["rule_prediction"] = 1
    else:
        df_rules_inliers = pd.DataFrame()
    ### Outliers
    if len(df_outliers_original) > 0:
        # Turn list of rules to dataframe
        print("Turning rules to hypercubes...")
        list_rules_outliers = list(df_outliers_original["rule"])
        df_rules_outliers = turn_rules_to_df(
            list_rules=list_rules_outliers, list_cols=feature_cols
        )
        # Get corresponding rule size from the original rule extraction model,
        # not on the hypercubes obtained later
        df_rules_outliers["size_rules"] = list(
            df_outliers_original["size_rules"].values
        )

        # Prune rules
        if simplify_rules:
            print("Prunning the rules obtained...")
            df_rules_pruned = df_rules_outliers.drop(columns=["size_rules"]).copy()
            df_rules_pruned = simplifyRules(df_rules_pruned, categorical_cols)
            df_rules_pruned = df_rules_pruned.reset_index().merge(
                df_rules_outliers.reset_index()[["index", "size_rules"]], how="left"
            )
            df_rules_pruned.index = df_rules_pruned["index"]
            df_rules_pruned = df_rules_pruned.drop(columns=["index"], errors="ignore")
            df_rules_outliers = df_rules_pruned
        # Col predicted
        df_rules_outliers["rule_prediction"] = -1
    else:
        df_rules_outliers = pd.DataFrame()
    return df_rules_inliers, df_rules_outliers


def rulefit_rules(
    df_anomalies,
    numerical_cols,
    categorical_cols,
    simplify_rules=False,
    model_params={},
):
    """
    Rule Extraction based on RuleFit algorithm.

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
    simplify_rules : TYPE, optional
        DESCRIPTION. The default is False.
    model_params : TYPE, optional
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    def _getRulesRulefit(df_aux, model_params):
        # Prepare data
        X_train = df_aux[feature_cols]
        y_train = df_aux["predictions"]

        # Fit model
        if "tree_size" not in model_params.keys():
            model_params["tree_size"] = len(feature_cols) * 2
        if "rfmode" not in model_params.keys():
            model_params["rfmode"] = "classify"
        rf = RuleFit(**model_params)
        rf.fit(X_train.values, y_train.values, feature_names=feature_cols)

        # Get rules
        print("Obtaining Rules using RuleFit...")
        rules_all = rf.get_rules()
        rules_all = rules_all[rules_all.coef != 0]
        rules_all = rules_all[rules_all.importance > 0].sort_values(
            "support", ascending=False
        )
        rules_all = rules_all[rules_all.coef > 0]
        rules_all = rules_all.sort_values("support", ascending=False)
        rules_all = rules_all[rules_all["type"] == "rule"]
        rules_all["size_rules"] = rules_all.apply(
            lambda x: len(x["rule"].split("&")), axis=1
        )

        # Turn list of rules to dataframe
        print("Turning rules to hypercubes...")
        df_rules = turn_rules_to_df(
            list_rules=list(rules_all["rule"].values), list_cols=feature_cols
        )

        # Get corresponding rule size from the original rule extraction model,
        # not on the hypercubes obtained later
        df_rules["size_rules"] = list(rules_all["size_rules"].values)

        # Prune rules
        if simplify_rules:
            print("Prunning the rules obtained...")
            df_rules_pruned = df_rules.drop(columns=["size_rules"]).copy()
            df_rules_pruned = simplifyRules(df_rules_pruned, categorical_cols)
            df_rules_pruned = df_rules_pruned.reset_index().merge(
                df_rules.reset_index()[["index", "size_rules"]], how="left"
            )
            df_rules_pruned.index = df_rules_pruned["index"]
            df_rules_pruned = df_rules_pruned.drop(columns=["index"], errors="ignore")
            df_rules = df_rules_pruned
        return df_rules

    # Prepare Data
    feature_cols = numerical_cols + categorical_cols

    ### Inliers
    df_inliers = df_anomalies.copy()
    df_inliers["predictions"] = df_inliers.apply(
        lambda x: 0 if x["predictions"] == -1 else 1, axis=1
    )
    if len(df_inliers) > 0:
        df_rules_inliers = _getRulesRulefit(df_inliers, model_params)
        df_rules_inliers["rule_prediction"] = -1
    else:
        df_rules_inliers = pd.DataFrame()
    ### Outliers
    df_outliers = df_anomalies.copy()
    df_outliers["predictions"] = df_outliers.apply(
        lambda x: 0 if x["predictions"] == 1 else 1, axis=1
    )
    if len(df_outliers) > 0:
        df_rules_outliers = _getRulesRulefit(df_outliers, model_params)
        df_rules_outliers["rule_prediction"] = -1
    else:
        df_rules_outliers = pd.DataFrame()
    return df_rules_inliers, df_rules_outliers


def frl_rules(
    df_anomalies,
    numerical_cols,
    categorical_cols,
    simplify_rules=False,
    model_params={},
):
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

    def _turn_ruleset_to_df(df_anomalies, list_rules, list_cols):
        df_rules = pd.DataFrame()

        # Iter for each rule
        for rule in list_rules:
            dct_aux = {}

            # Default values
            for col in list_cols:
                dct_aux[col + "_max"] = np.inf
                dct_aux[col + "_min"] = -np.inf
            list_subrules = rule

            # Iter for each component of the rule and obtain the limits
            for subrule in list_subrules:
                for col in list_cols:

                    if col in subrule:
                        aux = subrule.split("<")
                        # col < XXXX
                        if aux[0] in list_cols:
                            if np.float(aux[1]) <= dct_aux[col + "_max"]:
                                dct_aux[col + "_max"] = np.float(aux[1])
                        # XXXX < col
                        else:
                            if np.float(aux[0]) >= dct_aux[col + "_min"]:
                                dct_aux[col + "_min"] = np.float(aux[0])
            df_rules = df_rules.append(pd.DataFrame(dct_aux, index=[0]))
        return df_rules

    feature_cols = numerical_cols + categorical_cols
    X = df_anomalies[feature_cols]

    ### Rules for Inliers
    y = df_anomalies["predictions"].values
    y = np.array([x if x > 0 else 0 for x in y])
    model_rules = ruleset.BayesianRuleSet(**model_params)
    model_rules.fit(X, y)
    dict_rules = model_rules.rule_explainations
    list_rules = [x[0] for i, x in dict_rules.items()]
    df_rules_inliers = _turn_ruleset_to_df(df_anomalies, list_rules, feature_cols)
    df_rules_inliers["size_rules"] = [len(x) for x in list_rules]
    df_inliers = df_rules_inliers.copy()

    if len(df_inliers) > 0:
        # Prune rules
        if simplify_rules:
            print("Prunning the rules obtained...")
            df_inliers = df_inliers.reset_index(drop=True)
            df_rules_pruned = df_inliers.drop(columns=["size_rules"]).copy()
            df_rules_pruned = simplifyRules(df_rules_pruned, categorical_cols)
            df_rules_pruned = df_rules_pruned.reset_index().merge(
                df_inliers.reset_index()[["index", "size_rules"]], how="left"
            )
            df_rules_pruned.index = df_rules_pruned["index"]
            df_rules_pruned = df_rules_pruned.drop(columns=["index"], errors="ignore")
            df_rules_inliers = df_rules_pruned
            df_rules_inliers["rule_prediction"] = 1
    else:
        df_rules_inliers = pd.DataFrame()
    #### Rules for Outliers
    y = df_anomalies["predictions"].values
    y = np.array([1 if x < 0 else 0 for x in y])
    model_rules = ruleset.BayesianRuleSet(**model_params)
    model_rules.fit(X, y)
    dict_rules = model_rules.rule_explainations
    list_rules = [x[0] for i, x in dict_rules.items()]
    df_rules_outliers = _turn_ruleset_to_df(df_anomalies, list_rules, feature_cols)
    df_rules_outliers["size_rules"] = [len(x) for x in list_rules]
    df_outliers = df_rules_outliers.copy()

    if len(df_outliers) > 0:
        # Prune rules
        if simplify_rules:
            print("Prunning the rules obtained...")
            df_outliers = df_outliers.reset_index(drop=True)
            df_rules_pruned = df_outliers.drop(columns=["size_rules"]).copy()
            df_rules_pruned = simplifyRules(df_rules_pruned, categorical_cols)
            df_rules_pruned = df_rules_pruned.reset_index().merge(
                df_outliers.reset_index()[["index", "size_rules"]], how="left"
            )
            df_rules_pruned.index = df_rules_pruned["index"]
            df_rules_pruned = df_rules_pruned.drop(columns=["index"], errors="ignore")
            df_rules_outliers = df_rules_pruned
            df_rules_outliers["rule_prediction"] = -1
    else:
        df_rules_outliers = pd.DataFrame()
    return df_rules_inliers, df_rules_outliers


def skoperules_rules(
    df_anomalies,
    numerical_cols,
    categorical_cols,
    simplify_rules=False,
    model_params={},
):
    """

    Rules obtained using Skope Rules.

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
    simplify_rules : TYPE, optional
        DESCRIPTION. The default is False.
    model_params : TYPE, optional
        DESCRIPTION. The default is {}.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    def _getSkopeRules(X_train, y_train, model_params):
        # Rules
        print("Obtaining Rules using SkopeRules...")
        clf = SkopeRules(**model_params)
        clf.fit(X_train, y_train)
        rules = clf.rules_

        if len(rules) > 0:
            print("Checking inliers inside hypercubes...")
            df_rules = pd.DataFrame(
                {
                    "rule": [v[0].replace(" and ", " & ") for v in rules],
                    "precision": [v[1][0] for v in rules],
                    "recall": [v[1][1] for v in rules],
                    "n_points_correct": [v[1][2] for v in rules],
                }
            )
            if not df_rules.empty:
                df_rules["size_rules"] = df_rules.apply(
                    lambda x: len(x["rule"].split("&")), axis=1
                )
            else:
                df_rules["size_rules"] = 0
            rules = [v[0].replace(" and ", " & ") for v in rules]

            # Obtain rules in df format
            if len(rules) > 0:
                print("Turning rules to hypercubes...")
                df_rules_results = turn_rules_to_df(
                    list_rules=rules, list_cols=feature_cols
                )

                df_rules_pruned = simplifyRules(df_rules_results, categorical_cols)
                df_rules_pruned = df_rules_pruned.reset_index().merge(
                    df_rules.reset_index()[["index", "size_rules"]], how="left"
                )
                df_rules_pruned.index = df_rules_pruned["index"]
                df_rules_pruned = df_rules_pruned.drop(
                    columns=["index"], errors="ignore"
                )
                df_rules_results = df_rules_pruned
            else:
                df_rules_results = pd.DataFrame()
            return df_rules_results

    # Default params
    feature_cols = numerical_cols + categorical_cols

    if "random_state" not in model_params.keys():
        rng = np.random.RandomState(42)
        model_params["random_state"] = rng
    if "precision_min" not in model_params.keys():
        model_params["precision_min"] = 0.5
    if "recall_min" not in model_params.keys():
        model_params["recall_min"] = 0.01
    if "feature_names" not in model_params.keys():
        model_params["feature_names"] = feature_cols
    ### Inliers
    # Prepare Data
    df_aux = df_anomalies.copy()
    df_aux["predictions"] = df_aux.apply(
        lambda x: 0 if x["predictions"] < 0 else 1, axis=1
    )

    X_train = df_aux[feature_cols]
    y_train = df_aux["predictions"]

    df_rules_inliers = _getSkopeRules(X_train, y_train, model_params)
    df_rules_inliers["rulse_prediction"] = 1

    ### Outliers
    # Prepare Data
    feature_cols = numerical_cols + categorical_cols
    df_aux = df_anomalies.copy()
    df_aux["predictions"] = df_aux.apply(
        lambda x: 0 if x["predictions"] > 0 else 1, axis=1
    )

    X_train = df_aux[feature_cols]
    y_train = df_aux["predictions"]

    df_rules_outliers = _getSkopeRules(X_train, y_train, model_params)
    df_rules_outliers["rulse_prediction"] = -1

    return df_rules_inliers, df_rules_outliers


def decisionListClassifier(
    df_anomalies,
    numerical_cols,
    categorical_cols,
    simplify_rules=False,
    model_params={},
):
    """

    Rules obtained using Skope RulesDecision Lists.

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
    simplify_rules : TYPE, optional
        DESCRIPTION. The default is False.
    model_params : TYPE, optional
        DESCRIPTION. The default is {}.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    # Prepare data
    print(numerical_cols, categorical_cols)
    feature_cols = numerical_cols + categorical_cols
    X = df_anomalies[feature_cols]
    y = df_anomalies[["predictions"]]

    # Get Rules
    dlc = DecisionListClassifier(**model_params)
    dlc.fit(X, y)
    dlc_global = dlc.explain_global(name="Decision List Classifier")
    dct_dlc = dlc_global.data()
    df_rules = pd.DataFrame(
        {"rules": dct_dlc["rule"], "predictions": dct_dlc["outcome"]}
    )
    df_rules = df_rules[df_rules["rules"] != "No Rules Triggered"]

    ### Rules for Inliers
    print("Rules for inliers...")
    df_aux = df_rules[df_rules["predictions"] == 1]

    if len(df_aux) > 0:
        list_rules_inliers = list(df_aux["rules"])
        list_rules_inliers = [x.replace(" and ", " & ") for x in list_rules_inliers]
        print("Turning rules to hypercubes...")
        df_rules_results = turn_rules_to_df(
            list_rules=list_rules_inliers, list_cols=feature_cols
        )
        df_rules_results["size_rules"] = [len(x.split("&")) for x in list_rules_inliers]
        df_rules_pruned = simplifyRules(
            df_rules_results.drop(columns=["size_rules"]), categorical_cols
        )
        df_rules_pruned = df_rules_pruned.reset_index().merge(
            df_rules_results.reset_index()[["index", "size_rules"]], how="left"
        )
        df_rules_pruned.index = df_rules_pruned["index"]
        df_rules_pruned = df_rules_pruned.drop(columns=["index"], errors="ignore")
        df_rules_inliers = df_rules_pruned.copy()
        df_rules_inliers["rule_prediction"] = 1
    else:
        df_rules_inliers = pd.DataFrame()
    ### Rules for Outliers
    print("Rules for inliers...")
    df_aux = df_rules[df_rules["predictions"] == -1]

    if len(df_aux) > 0:
        list_rules_outliers = list(df_aux["rules"])
        list_rules_outliers = [x.replace(" and ", " & ") for x in list_rules_outliers]
        print("Turning rules to hypercubes...")
        df_rules_results = turn_rules_to_df(
            list_rules=list_rules_outliers, list_cols=feature_cols
        )
        df_rules_results["size_rules"] = [
            len(x.split("&")) for x in list_rules_outliers
        ]
        df_rules_pruned = simplifyRules(
            df_rules_results.drop(columns=["size_rules"]), categorical_cols
        )
        df_rules_pruned = df_rules_pruned.reset_index().merge(
            df_rules_results.reset_index()[["index", "size_rules"]], how="left"
        )
        df_rules_pruned.index = df_rules_pruned["index"]
        df_rules_pruned = df_rules_pruned.drop(columns=["index"], errors="ignore")
        df_rules_outliers = df_rules_pruned.copy()
        df_rules_outliers["rule_prediction"] = -1
    else:
        df_rules_outliers = pd.DataFrame()
    return df_rules_inliers, df_rules_outliers


def aix360_rules_wrapper(
    df_anomalies,
    numerical_cols,
    categorical_cols,
    rule_algorithm="",
    simplify_rules=False,
    model_params={},
):
    """
    Rules obtained using brlg or logrr.

    Parameters
    ----------
    df_anomalies : TYPE
        DESCRIPTION.
    numerical_cols : TYPE
        DESCRIPTION.
    categorical_cols : TYPE
        DESCRIPTION.
    rule_algorithm : TYPE, optional
        DESCRIPTION. The default is "".
    simplify_rules : TYPE, optional
        DESCRIPTION. The default is False.
    model_params : TYPE, optional
        DESCRIPTION. The default is {}.

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

    """

    # Define variables
    feature_cols = numerical_cols + categorical_cols
    X = df_anomalies[feature_cols].astype(float)
    y = df_anomalies["predictions"].astype(int)
    y_inliers = np.array([x if x > 0 else 0 for x in y])  # Defined for inliers levels
    y_outliers = np.array([1 if x < 0 else 0 for x in y])  # Defined for outlier levels

    # Feature binarize
    fb = FeatureBinarizer(
        negations=True, returnOrd=True, colsCateg=categorical_cols, numThres=90
    )
    X_fb, X_std = fb.fit_transform(X)

    # Choose model
    if rule_algorithm == "brlg":

        # Default params
        if "lambda0" not in model_params.keys():
            model_params["lambda0"] = 1e-3
        if "lambda1" not in model_params.keys():
            model_params["lambda1"] = 1e-3
        if "CNF" not in model_params.keys():
            model_params["CNF"] = False
        # Inliers
        model_rules = BooleanRuleCG(**model_params)
        model_rules.fit(X_fb, y_inliers)
        list_rules_inliers = model_rules.explain()["rules"]

        # Outliers
        model_rules = BooleanRuleCG(**model_params)
        model_rules.fit(X_fb, y_outliers)
        list_rules_outliers = model_rules.explain()["rules"]
    elif rule_algorithm == "logrr":

        # Default params
        if "lambda0" not in model_params.keys():
            model_params["lambda0"] = 0.005
        if "lambda1" not in model_params.keys():
            model_params["lambda1"] = 0.001
        # Obtain rules [Inliers]
        model_rules = LogisticRuleRegression(**model_params)
        model_rules.fit(X_fb, y_inliers, X_std)
        df_rules = model_rules.explain()

        try:
            # Inliers
            df_rules_inliers = df_rules[
                (df_rules["coefficient"] > 0)
                & (df_rules["rule/numerical feature"] != "(intercept)")
            ]
            list_rules_inliers = list(df_rules_inliers["rule/numerical feature"])

            # Outliers
            df_rules_outliers = df_rules[
                (df_rules["coefficient"] < 0)
                & (df_rules["rule/numerical feature"] != "(intercept)")
            ]
            list_rules_outliers = list(df_rules_outliers["rule/numerical feature"])
        except KeyError:
            # Inliers
            df_rules_inliers = df_rules[
                (df_rules["coefficient"] > 0) & (df_rules["rule"] != "(intercept)")
            ]
            list_rules_inliers = list(df_rules_inliers["rule"])

            # Outliers
            df_rules_outliers = df_rules[
                (df_rules["coefficient"] < 0) & (df_rules["rule"] != "(intercept)")
            ]
            list_rules_outliers = list(df_rules_outliers["rule"])
    else:
        raise ValueError("Argument {0} not recognised -- use 'brlg' or 'logrr' instead")
    # Turn to DF
    list_rules_inliers = [x.replace("AND", "&") for x in list_rules_inliers]
    list_rules_outliers = [x.replace("AND", "&") for x in list_rules_outliers]
    df_inliers = turn_rules_to_df(list_rules=list_rules_inliers, list_cols=feature_cols)
    df_outliers = turn_rules_to_df(
        list_rules=list_rules_outliers, list_cols=feature_cols
    )

    # Get rule size
    df_inliers = df_inliers.reset_index(drop=True)
    df_inliers["size_rules"] = [len(x.split("&")) for x in list_rules_inliers]
    df_outliers = df_outliers.reset_index(drop=True)
    df_outliers["size_rules"] = [len(x.split("&")) for x in list_rules_outliers]

    # Prune rules
    if simplify_rules:
        if len(df_inliers) > 0:
            df_rules_pruned = simplifyRules(
                df_inliers.drop(columns=["size_rules"]), categorical_cols
            )
            df_rules_pruned = df_rules_pruned.reset_index().merge(
                df_inliers.reset_index()[["index", "size_rules"]], how="left"
            )
            df_rules_pruned.index = df_rules_pruned["index"]
            df_rules_pruned = df_rules_pruned.drop(columns=["index"], errors="ignore")
            df_rules_inliers = df_rules_pruned.copy()
            df_rules_inliers["rule_prediction"] = 1
        else:
            df_rules_inliers = pd.DataFrame()
        if len(df_outliers) > 0:
            df_rules_pruned = simplifyRules(
                df_outliers.drop(columns=["size_rules"]), categorical_cols
            )
            df_rules_pruned = df_rules_pruned.reset_index().merge(
                df_outliers.reset_index()[["index", "size_rules"]], how="left"
            )
            df_rules_pruned.index = df_rules_pruned["index"]
            df_rules_pruned = df_rules_pruned.drop(columns=["index"], errors="ignore")
            df_rules_outliers = df_rules_pruned.copy()
            df_rules_outliers["rule_prediction"] = -1
        else:
            df_rules_outliers = pd.DataFrame()
    else:
        df_rules_inliers = df_inliers
        df_rules_inliers["rule_prediction"] = 1
        df_rules_outliers = df_outliers
        df_rules_outliers["rule_prediction"] = -1
    return df_rules_inliers, df_rules_outliers
