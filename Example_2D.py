# -*- coding: utf-8 -*-
"""
# Rule Extraction for Unsupervised Outlier Detection
Example of usage of a library that wrapping an unsupervised outlier detection algorithm (OneClassSVM) of scikit-learn 
it can infer rules that are comprehensible for human beings, so the'll be able to easily understand why an specific data point is labeled as an outlier, 
using to do so a method called SVM+Prototypes. To show it's capabilities the outlier analysis is applied on a dataset
"""

# Libraries
import os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
from dateutil.parser import parse
from scipy.io import arff
from io import StringIO
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Import library
f_folder = os.getcwd()
f_path_lib = os.path.join(f_folder, "code")
lib_path = os.path.abspath(f_path_lib)
sys.path.append(lib_path)

# Import dependencies
from lib.unsupervised_rules import ocsvm_rule_extractor
from lib.pipelines import ocsvm_rules_experiments_pipeline
from lib.common import grid_search, train_one_class_svm
from lib.tools import (
    dt_rules, turn_rules_to_df, plot_2D, anchors_rules,
    rulefit_rules, skoperules_rules, surrogate_dt_rules,
    ocsvm_rules_completion, file_naming_ocsvm,
    aix360_rules_wrapper, frl_rules
    )

"""### 1. Load & Prepare Data"""

# =============================================================================
# Prepare Data
# =============================================================================
# Load dataset
f_path_datasets = os.path.join(f_folder, "dataset")
f_name = "seismic-bumps.csv"
df_raw = pd.read_csv(os.path.join(f_path_datasets, f_name))

# Encoding categorical columns
obj_df = df_raw.select_dtypes(
    include=['object']).copy() 
print(obj_df.columns)

lb_encoder = LabelEncoder()

for col in list(obj_df.columns):
    df_raw[col] = lb_encoder.fit_transform(df_raw[col])

# Save ground truth
df_ground_truth = shuffle(df_raw)
X = df_ground_truth.copy()
del X['seismic']
y = df_ground_truth['seismic'].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42)

# Anomalies used
df_data = X_train.copy()
df_data['seismic'] = y_train

# Choose subsample
df_data = df_data.sort_values(
    by=['seismic'], ascending=True).reset_index(drop=True)

df_data = df_data.iloc[:int(np.round(0.66 * len(df_raw)))]
df_data = shuffle(df_data).reset_index(drop=True)
print("% of seismic: ",
      100 * np.round(len(df_data[df_data['seismic'] == 1]) / len(df_data), 3),
      "%")
df_data.head()

# =============================================================================
# Choose columns
# =============================================================================
df_mat = df_data[[
    'seismoacoustic', 'shift', 'genergy', 'gplus', 'gdenergy', 'gdpuls',
    'hazard', 'bumps', 'bumps2'
]]

categorical_cols = ['hazard', 'shift']
numerical_cols = [x for x in list(df_mat.columns) if x not in categorical_cols]
df_mat = df_data[numerical_cols + categorical_cols]
df_mat.head()

# =============================================================================
#  Grid Hyperparams
# =============================================================================
dct_joblib = {'n_jobs': 2, 'verbose': 0, 'backend': "loky"}
#dct_params = grid_search(df_mat, numerical_cols, categorical_cols, dct_joblib)

df_mat = df_mat[(df_mat['hazard'] == 0) & (df_mat['shift'] == 0)]  # 1 Cat
df_mat = df_mat[['gdenergy', 'gdpuls']].drop_duplicates()  # 2D
categorical_cols = []
numerical_cols = list(df_mat.columns)
df_mat = df_mat.reset_index(drop=True)

# =============================================================================
# Define hyperparams and path names
# =============================================================================
#### Define Hyperparams
# Hyperparameters to use
dct_params = {'nu': 0.1, 'kernel': "rbf", 'gamma': 0.1}
script_name = "seismic_2D"
path_folder = "results/seismic_2D"
file_template = "{script_name}_kernel_{kernel}".format(
    script_name=script_name,
    kernel=dct_params['kernel']
    )
# Create results folder if it does not exist
f_path_results = os.path.join(f_folder, path_folder)
if not os.path.exists(f_path_results):
    os.makedirs(f_path_results)

# =============================================================================
# Obtain Rules (OCSVM) - K means
# =============================================================================
#### K Means + Discard
CLUSTER_ALGORITHM = "kmeans"
METHOD = "discard"

ocsvm_rules_experiments_pipeline(
    df_mat = df_mat,
    numerical_cols = numerical_cols,
    categorical_cols = categorical_cols,
    cluster_algorithm = CLUSTER_ALGORITHM,
    method = METHOD,
    rules_used = "all",
    dct_params = dct_params,
    path_folder = path_folder,
    file_template = file_template,
    plot_fig = True
    )

#### K Means + Keep (Reset)
CLUSTER_ALGORITHM = "kmeans"
METHOD = "keep_reset"

ocsvm_rules_experiments_pipeline(
    df_mat = df_mat,
    numerical_cols = numerical_cols,
    categorical_cols = categorical_cols,
    cluster_algorithm = CLUSTER_ALGORITHM,
    method = METHOD,
    rules_used = "all",
    dct_params = dct_params,
    path_folder = path_folder,
    file_template = file_template,
    plot_fig = True
    )

#### K Means + Keep
CLUSTER_ALGORITHM = "kmeans"
METHOD = "keep"

ocsvm_rules_experiments_pipeline(
    df_mat = df_mat,
    numerical_cols = numerical_cols,
    categorical_cols = categorical_cols,
    cluster_algorithm = CLUSTER_ALGORITHM,
    method = METHOD,
    rules_used = "all",
    dct_params = dct_params,
    path_folder = path_folder,
    file_template = file_template,
    plot_fig = True
    )

# =============================================================================
# Obtain General Model for the rest of the experiments
# =============================================================================
# Does not matter 'clustering_algorithm' or 'method'
CLUSTER_ALGORITHM = "kmeans"
METHOD = "discard"
clf, sc, _, df_anomalies = ocsvm_rule_extractor(
    dataset_mat=df_mat,
    numerical_cols=numerical_cols,
    categorical_cols=categorical_cols,
    clustering_algorithm=CLUSTER_ALGORITHM,
    method=METHOD,
    use_inverse=False,
    path_save_model=path_folder,
    dct_params=dct_params
     )

# =============================================================================
# Surrogate Decision Tree
# =============================================================================
#### Obtain Rules
df_rules_inliers, df_rules_outliers, df_no_pruned, df_yes_pruned = surrogate_dt_rules(
     df_anomalies,
     clf,
     numerical_cols,
     categorical_cols,
     path=path_folder,
     file_name=file_template
     )
                                                   
#### Plot Rules [Inliers]
df_no_pruned = df_no_pruned.copy()
df_no_pruned = df_no_pruned.drop_duplicates().reset_index(drop=True)
df_no_pruned = df_no_pruned.replace(np.inf, 350)
df_no_pruned = df_no_pruned.replace(-np.inf, -120)
plot_2D(df_no_pruned,
        df_anomalies,
        folder = path_folder,
        path_name=file_template + '_inliers_DT')

#### Plot Rules [Outliers]
df_yes_pruned = df_yes_pruned.copy()
df_yes_pruned = df_yes_pruned.drop_duplicates().reset_index(drop=True)
df_yes_pruned = df_yes_pruned.replace(np.inf, 350)
df_yes_pruned = df_yes_pruned.replace(-np.inf, -120)
plot_2D(df_yes_pruned,
        df_anomalies,
        folder = path_folder,
        path_name=file_template + '_outliers_DT')

# =============================================================================
# Anchors (Rules)
# =============================================================================
#### Obtain Rules
(list_rules_transformed_no, df_rules_anchors_no,
 df_rules_anchors_yes, list_rules_anchors_no,
 df_yes_pruned, df_no_pruned) = anchors_rules(
     df_anomalies,
     numerical_cols,
     categorical_cols,
     model=clf,
     scaler=sc,
     path=path_folder,
     file_name=file_template
     )

#### Plot Rules [Inliers]
df_plot = df_no_pruned.copy()
df_plot = df_plot.replace(np.inf, 350)
df_plot = df_plot.replace(-np.inf, -120)
df_plot = df_plot.drop_duplicates().reset_index(drop=True)
plot_2D(df_plot,
        df_anomalies,
        folder = path_folder,
        path_name=file_template + '_inliers_Anchors')

#### Plot Rules [Outliers]
df_plot = df_yes_pruned.copy()
df_plot = df_plot.replace(np.inf, 350)
df_plot = df_plot.replace(-np.inf, -120)
df_plot = df_plot.drop_duplicates().reset_index(drop=True)
plot_2D(df_plot,
        df_anomalies,
        folder = path_folder,
        path_name=file_template + '_outliers_Anchors')

# =============================================================================
# RuleFit
# =============================================================================
# Obtain rules
df_check, df_rules_outliers, df_rules_inliers = rulefit_rules(
    df_anomalies,
    clf,
    numerical_cols,
    categorical_cols,
    path=path_folder,
    file_name=file_template
    )

#### Plot Rules [Inliers]
df_rules_inliers = df_rules_inliers.copy()
df_rules_inliers = df_rules_inliers.drop_duplicates().reset_index(drop=True)
df_rules_inliers = df_rules_inliers.replace(np.inf, 350)
df_rules_inliers = df_rules_inliers.replace(-np.inf, -120)
plot_2D(df_rules_inliers,
        df_anomalies,
        folder = path_folder,
        path_name=file_template + '_inliers_RuleFit')


#### Plot Rules [Outliers]
df_rules_outliers = df_rules_outliers.copy()
df_rules_outliers = df_rules_outliers.drop_duplicates().reset_index(drop=True)
df_rules_outliers = df_rules_outliers.replace(np.inf, 350)
df_rules_outliers = df_rules_outliers.replace(-np.inf, -120)
plot_2D(df_rules_outliers,
        df_anomalies,
        folder = path_folder,
        path_name=file_template + '_outliers_RuleFit')

# =============================================================================
# SkopeRules               
# =============================================================================
### Obtain Rules
(df_rules_info_inliers, df_rules_info_outliers,
 df_rules_inliers, df_rules_outliers,
 df_no_pruned, df_yes_pruned) = skoperules_rules(
     df_anomalies,
     clf,
     numerical_cols,
     categorical_cols,
     path=path_folder,
     file_name=file_template
     )

#### Plot Rules [Inliers]
df_plot = df_no_pruned.copy()
df_plot = df_plot.drop_duplicates().reset_index(drop=True)
df_plot = df_plot.replace(np.inf, 350)
df_plot = df_plot.replace(-np.inf, -120)
plot_2D(df_plot,
        df_anomalies,
        folder = path_folder,
        path_name=file_template + '_inliers_SkopeRules')

#### Plot Rules [Outliers]
df_plot = df_yes_pruned.copy()
df_plot = df_plot.drop_duplicates().reset_index(drop=True)
df_plot = df_plot.replace(np.inf, 350)
df_plot = df_plot.replace(-np.inf, -120)
plot_2D(df_plot,
        df_anomalies,
        folder = path_folder,
        path_name=file_template + '_outliers_SkopeRules')

# =============================================================================
# Rules from AIX360
# =============================================================================
#### Choose algorithm
for rule_algorithm in ["brlg", "logrr", "glrm"]:
    print("Rules for: {0}".format(rule_algorithm))

    ### Obtain Rules
    (df_rules_inliers, df_rules_outliers,
    df_no_pruned, df_yes_pruned) = aix360_rules_wrapper(
        df_anomalies,
        clf,
        numerical_cols,
        categorical_cols,
        use_oversampling=True,
        rule_algorithm=rule_algorithm,
        path=path_folder,
        file_name=file_template
    )
    
    #### Plot Rules [Inliers]
    if len(df_no_pruned) > 0:
        path_add = ""
        df_plot = df_no_pruned.copy()
    else:
        path_add = "with_errors"
        df_plot = df_rules_inliers.copy()
        
    df_plot = df_plot.drop_duplicates().reset_index(drop=True)
    df_plot = df_plot.replace(np.inf, 350)
    df_plot = df_plot.replace(-np.inf, -120)
    plot_2D(df_plot,
            df_anomalies,
            folder = path_folder,
            path_name=file_template + '_inliers_{0}_{1}'.format(rule_algorithm, path_add))
    
    #### Plot Rules [Outliers]
    if len(df_yes_pruned) > 0:
        path_add = ""
        df_plot = df_yes_pruned.copy()
    else:
        path_add = "with_errors"
        df_plot = df_rules_outliers.copy()
        
    df_plot = df_plot.drop_duplicates().reset_index(drop=True)
    df_plot = df_plot.replace(np.inf, 350)
    df_plot = df_plot.replace(-np.inf, -120)
    plot_2D(df_plot,
            df_anomalies,
            folder = path_folder,
            path_name=file_template + '_outliers_{0}_{1}'.format(rule_algorithm, path_add))

# =============================================================================
# Falling Rule List (FRL)
# =============================================================================
### Obtain Rules
(df_rules_inliers, df_rules_outliers,
df_no_pruned, df_yes_pruned) = frl_rules(
    df_anomalies,
    clf,
    numerical_cols,
    categorical_cols,
    path=path_folder,
     file_name=file_template
     )
                                         
#### Plot Rules [Inliers]
if len(df_no_pruned) > 0:
    path_add = ""
    df_plot = df_no_pruned.copy()
else:
    path_add = "with_errors"
    df_plot = df_rules_inliers.copy()
                                     
#### Plot Rules [Inliers]
df_plot = df_plot.drop_duplicates().reset_index(drop=True)
df_plot = df_plot.replace(np.inf, 350)
df_plot = df_plot.replace(-np.inf, -120)
plot_2D(df_plot,
        df_anomalies,
        folder = path_folder,
        path_name=file_template + '_inliers_FRL')

#### Plot Rules [Outliers]
if len(df_yes_pruned) > 0:
    path_add = ""
    df_plot = df_yes_pruned.copy()
else:
    path_add = "with_errors"
    df_plot = df_rules_outliers.copy()

#### Plot Rules [Outliers]
df_plot = df_plot.drop_duplicates().reset_index(drop=True)
df_plot = df_plot.replace(np.inf, 350)
df_plot = df_plot.replace(-np.inf, -120)
plot_2D(df_plot,
        df_anomalies,
        folder = path_folder,
        path_name=file_template + '_outliers_FRL')

