# -*- coding: utf-8 -*-
"""
Created on Tue May 18 21:32:14 2021

@author: alber
"""
import pandas as pd
import numpy as np


def turn_rules_to_df(list_rules, list_cols):
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
    list_rules : list
        A list with the following structure (example):
        ['gdpuls > -83.5',
         'gdpuls > -79.5',
         'gdenergy > -76.0',
         'gdenergy > -79.0 & gdpuls > -70.0'
         ]

    list_cols : list
        List of features considered. For instance, for the example above:
            feature_cols = ['gdenergy', 'gdpuls']

    Returns
    -------
    df_rules : dataframe
        A dataframe with two columns per feature (max/min) and one row
        per rule. For the example above:
            gdenergy_max  gdenergy_min  gdpuls_max  gdpuls_min
                    inf          -inf         inf       -83.5
                    inf          -inf         inf       -79.5
                    inf         -76.0         inf        -inf
                    inf         -79.0         inf       -70.0
                    inf          -inf         inf       -78.5

    """

    df_rules = pd.DataFrame()

    if len(list_rules) == 0:
        print("Warning: Rule list is empty: returning a DF without Rules")
        return pd.DataFrame({col + "_max": [] for col in list_cols}).append(
            pd.DataFrame({col + "_min": [] for col in list_cols})
        )
    # Iter for each rule
    for rule in list_rules:
        dct_aux = {}

        # Default values
        for col in list_cols:
            dct_aux[col + "_max"] = np.inf
            dct_aux[col + "_min"] = -np.inf
        list_subrules = rule.split("&")

        # Iter for each component of the rule and obtain the limits
        for subrule in list_subrules:
            for col in list_cols:

                if col in subrule:
                    if ">=" in subrule:
                        aux = subrule.split(">=")
                        if np.float(aux[1]) >= dct_aux[col + "_min"]:
                            dct_aux[col + "_min"] = np.float(aux[1])
                    elif ">" in subrule:
                        aux = subrule.split(">")
                        if np.float(aux[1]) >= dct_aux[col + "_min"]:
                            dct_aux[col + "_min"] = np.float(aux[1])
                    if "<=" in subrule:
                        aux = subrule.split("<=")
                        if np.float(aux[1]) <= dct_aux[col + "_max"]:
                            dct_aux[col + "_max"] = np.float(aux[1])
                    elif "<" in subrule:
                        aux = subrule.split("<")
                        if np.float(aux[1]) <= dct_aux[col + "_max"]:
                            dct_aux[col + "_max"] = np.float(aux[1])
        df_rules = df_rules.append(pd.DataFrame(dct_aux, index=[0])).reset_index(
            drop=True
        )
    return df_rules


def _rule_prunning(list_cols, rule_check, rule_ref):
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


def simplifyRules(df_all, categorical_cols=[]):
    """
    Function to reduce the number of rules checking if they are contained inside
    another hypercube.

    It can receive a list of categorical_cols in order to analyze the subsets
    belonging to the different categorical combinations independently; in order
    to analyze all the hyperspace together, that list should be empty.

    # TODO
    """

    def iter_prunning(df_all, categorical_cols=[]):
        def choose_one(i, rule_check, cols, df_cat):
            print("Iter {0}/{1}".format(i, len(df_all)))
            df_new = pd.concat(
                [
                    pd.DataFrame(_rule_prunning(cols, rule_check, rule_ref)).T
                    for j, rule_ref in df_cat.iterrows()
                    if i != j
                ]
            )
            cols_max = [col for col in list(df_new.columns) if "max" in col]
            cols_min = [col for col in list(df_new.columns) if "min" in col]
            df_new = df_new.sort_values(
                by=cols_max + cols_min,
                ascending=[False] * len(cols_max) + [True] * len(cols_min),
            )
            return df_new.head(1)

        cols = [x for x in list(df_all.columns) if x not in categorical_cols]

        if len(categorical_cols) > 0:
            df_end = pd.DataFrame()
            if len(categorical_cols) > 0:
                df_cat = df_all[categorical_cols]
                df_cat_unique = df_cat.drop_duplicates()
                for i, row in df_cat_unique.iterrows():
                    print("Iter {0}".format(i))
                    list_index = df_cat[df_cat[row.index] == row.values].dropna().index
                    df_all_aux = df_all[(df_all.index.isin(list_index))].copy()

                    if len(df_all_aux) > 1:
                        df_iter = pd.concat(
                            [
                                choose_one(i, rule_check, cols, df_all_aux)
                                for i, rule_check in df_all_aux.iterrows()
                            ]
                        )
                    else:
                        df_iter = df_all_aux
                    if df_end.empty:
                        df_end = df_iter
                    else:
                        df_end = df_end.append(df_iter)
            df_end = df_end.drop_duplicates()
        else:
            df_end = pd.concat(
                [
                    choose_one(i, rule_check, cols, df_all)
                    for i, rule_check in df_all.iterrows()
                ]
            )
        return df_end

    check = True
    while check:
        df_end = iter_prunning(df_all, categorical_cols)

        if len(df_end) == len(df_all):
            print("No more improvements... finishing up")
            check = False
        else:
            # New iter with pruned rules
            df_all = df_end.drop_duplicates().reset_index(drop=True)
    return df_end.drop_duplicates()
