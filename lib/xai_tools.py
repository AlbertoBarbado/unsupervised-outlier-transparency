# -*- coding: utf-8 -*-
"""
Created on Wed May 19 20:47:56 2021

@author: alber
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_2D(df_rules, df_anomalies, title=""):
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
    plt.title("Anomalies and Rules - {0}".format(title))