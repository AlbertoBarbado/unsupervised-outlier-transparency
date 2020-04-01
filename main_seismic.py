# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 20:05:29 2019

@author: alber
"""
# Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import os
import sys
from lib.unsupervised_rules import ocsvm_rule_extractor
from lib.tools import (dt_rules, turn_rules_to_df, plot_2D, anchors_rules,
                       rulefit_rules, skoperules_rules, surrogate_dt_rules,
                       ocsvm_rules_completion, file_naming_ocsvm,
                       aix360_rules_wrapper, frl_rules)
from lib.pipelines import ocsvm_rules_experiments_pipeline
from lib.common import grid_search, train_one_class_svm
from dateutil.parser import parse
from scipy.io import arff
from io import StringIO
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# Load & Preprocess data
# =============================================================================

"""
Attribute information:
1. seismic: result of shift seismic hazard assessment in the mine working obtained by the seismic
method (a - lack of hazard, b - low hazard, c - high hazard, d - danger state);
2. seismoacoustic: result of shift seismic hazard assessment in the mine working obtained by the
seismoacoustic method;
3. shift: information about type of a shift (W - coal-getting, N -preparation shift);
4. genergy: seismic energy recorded within previous shift by the most active geophone (GMax) out of
geophones monitoring the longwall;
5. gpuls: a number of pulses recorded within previous shift by GMax;
6. gdenergy: a deviation of energy recorded within previous shift by GMax from average energy recorded
during eight previous shifts;
7. gdpuls: a deviation of a number of pulses recorded within previous shift by GMax from average number
of pulses recorded during eight previous shifts;
8. ghazard: result of shift seismic hazard assessment in the mine working obtained by the
seismoacoustic method based on registration coming form GMax only;
9. nbumps: the number of seismic bumps recorded within previous shift;
10. nbumps2: the number of seismic bumps (in energy range [10^2,10^3)) registered within previous shift;
11. nbumps3: the number of seismic bumps (in energy range [10^3,10^4)) registered within previous shift;
12. nbumps4: the number of seismic bumps (in energy range [10^4,10^5)) registered within previous shift;
13. nbumps5: the number of seismic bumps (in energy range [10^5,10^6)) registered within the last shift;
14. nbumps6: the number of seismic bumps (in energy range [10^6,10^7)) registered within previous shift;
15. nbumps7: the number of seismic bumps (in energy range [10^7,10^8)) registered within previous shift;
16. nbumps89: the number of seismic bumps (in energy range [10^8,10^10)) registered within previous shift;
17. energy: total energy of seismic bumps registered within previous shift;
18. maxenergy: the maximum energy of the seismic bumps registered within previous shift;
19. class: the decision attribute - '1' means that high energy seismic bump occurred in the next shift
('hazardous state'), '0' means that no high energy seismic bumps occurred in the next shift
('non-hazardous state').

"""
# Load dataset
df_raw = pd.read_csv("dataset/seismic-bumps.csv")

# Encoding categorical columns 
obj_df = df_raw.select_dtypes(include=['object']).copy() # se eligen las variables categoricas (object)
print(obj_df.columns)

lb_encoder = LabelEncoder()
for col in list(obj_df.columns):
    df_raw[col] = lb_encoder.fit_transform(df_raw[col])

# Choose columns
categorical_cols = ['hazard', 'shift']
numerical_cols = [x for x in list(df_raw.columns) if x not in categorical_cols]
    
# One-hot encoding of categorical features
for col in categorical_cols:
    df_aux = pd.get_dummies(df_raw[col], prefix=col, drop_first=True)
    df_raw.drop([col], axis=1, inplace=True)
    df_raw = df_raw.join(df_aux)
df_raw = df_raw.astype(float)

# New categorical cols
categorical_cols = [x for x in list(df_raw.columns) if x not in numerical_cols]

# Save ground truth  
df_ground_truth = shuffle(df_raw)
X = df_ground_truth.copy()
del X['seismic']
y = df_ground_truth['seismic'].copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Anomalies used
df_data = X_train.copy()
df_data['seismic'] = y_train
    
# Choose subsample
df_data = df_data.sort_values(by=['seismic'], ascending=True).reset_index(drop=True)
df_data = df_data.iloc[:int(np.round(0.66*len(df_raw)))]
df_data = shuffle(df_data).reset_index(drop=True)  
print("% of seismic: ", 100*np.round(len(df_data[df_data['seismic']==1])/len(df_data),3), "%")

# Choose columns
df_mat = df_data[numerical_cols + categorical_cols]

# =============================================================================
# Define hyperparams and path names
# =============================================================================
#### Define Hyperparams
# Hyperparameters to use
#dct_params = {'nu':0.1, 'kernel':"rbf", 'gamma':0.7}
dct_params = {'nu': 0.1, 'kernel': "rbf", 'gamma': 0.1}
script_name = "seismic"
path_folder = "results/seismic"
file_template = "{script_name}_kernel_{kernel}".format(script_name=script_name,
                                                       kernel=dct_params['kernel'])

# =============================================================================
# Obtain Rules (OCSVM) - K means
# =============================================================================
#### K Means + Discard
CLUSTER_ALGORITHM = "kmeans"
METHOD = "discard"

ocsvm_rules_experiments_pipeline(df_mat = df_mat,
                                 numerical_cols = numerical_cols,
                                 categorical_cols = categorical_cols,
                                 cluster_algorithm = CLUSTER_ALGORITHM,
                                 method = METHOD,
                                 rules_used = "all",
                                 dct_params = dct_params,
                                 path_folder = path_folder,
                                 file_template = file_template,
                                 store_intermediate=True,
                                 plot_fig = False)

#### K Means + Keep (Reset)
CLUSTER_ALGORITHM = "kmeans"
METHOD = "keep_reset"

ocsvm_rules_experiments_pipeline(df_mat = df_mat,
                                 numerical_cols = numerical_cols,
                                 categorical_cols = categorical_cols,
                                 cluster_algorithm = CLUSTER_ALGORITHM,
                                 method = METHOD,
                                 rules_used = "all",
                                 dct_params = dct_params,
                                 path_folder = path_folder,
                                 file_template = file_template,
                                 store_intermediate=True,
                                 plot_fig = False)

#### K Means + Keep
CLUSTER_ALGORITHM = "kmeans"
METHOD = "keep"

ocsvm_rules_experiments_pipeline(df_mat = df_mat,
                                 numerical_cols = numerical_cols,
                                 categorical_cols = categorical_cols,
                                 cluster_algorithm = CLUSTER_ALGORITHM,
                                 method = METHOD,
                                 rules_used = "all",
                                 dct_params = dct_params,
                                 path_folder = path_folder,
                                 file_template = file_template,
                                 store_intermediate=True,
                                 plot_fig = False)

# =============================================================================
# Obtain Rules (OCSVM) - K Prototypes
# =============================================================================
#### K Means + Discard
CLUSTER_ALGORITHM = "kprototypes"
METHOD = "discard"

ocsvm_rules_experiments_pipeline(df_mat = df_mat,
                                 numerical_cols = numerical_cols,
                                 categorical_cols = categorical_cols,
                                 cluster_algorithm = CLUSTER_ALGORITHM,
                                 method = METHOD,
                                 rules_used = "all",
                                 dct_params = dct_params,
                                 path_folder = path_folder,
                                 file_template = file_template,
                                 store_intermediate=True,
                                 plot_fig = False)

#### K Means + Keep (Reset)
CLUSTER_ALGORITHM = "kprototypes"
METHOD = "keep_reset"

ocsvm_rules_experiments_pipeline(df_mat = df_mat,
                                 numerical_cols = numerical_cols,
                                 categorical_cols = categorical_cols,
                                 cluster_algorithm = CLUSTER_ALGORITHM,
                                 method = METHOD,
                                 rules_used = "all",
                                 dct_params = dct_params,
                                 path_folder = path_folder,
                                 file_template = file_template,
                                 store_intermediate=True,
                                 plot_fig = False)

#### K Means + Keep
CLUSTER_ALGORITHM = "kprototypes"
METHOD = "keep"

ocsvm_rules_experiments_pipeline(df_mat = df_mat,
                                 numerical_cols = numerical_cols,
                                 categorical_cols = categorical_cols,
                                 cluster_algorithm = CLUSTER_ALGORITHM,
                                 method = METHOD,
                                 rules_used = "all",
                                 dct_params = dct_params,
                                 path_folder = path_folder,
                                 file_template = file_template,
                                 store_intermediate=True,
                                 plot_fig = False)

# =============================================================================
# Obtain General Model for the rest of the experiments
# =============================================================================
# Does not matter 'clustering_algorithm' or 'method'
CLUSTER_ALGORITHM = "kmeans"
METHOD = "discard"
clf, sc, _, df_anomalies = ocsvm_rule_extractor(dataset_mat=df_mat,
                                                numerical_cols=numerical_cols,
                                                categorical_cols=categorical_cols,
                                                clustering_algorithm=CLUSTER_ALGORITHM,
                                                method=METHOD,
                                                use_inverse=False,
                                                dct_params=dct_params)

# =============================================================================
# Surrogate Decision Tree
# =============================================================================
#### Obtain Rules
CLUSTER_ALGORITHM = "kmeans"
METHOD = "discard"
(df_rules_inliers, df_rules_outliers,
 df_no_pruned, df_yes_pruned) = surrogate_dt_rules(df_anomalies,
                                                   clf,
                                                   numerical_cols,
                                                   categorical_cols,
                                                   path=path_folder,
                                                   file_name=file_template)

# =============================================================================
# Anchors (Rules)
# =============================================================================
#### Obtain Rules
(list_rules_transformed_no, df_rules_anchors_no,
 df_rules_anchors_yes, list_rules_anchors_no,
 df_yes_pruned, df_no_pruned) = anchors_rules(df_anomalies,
                                              numerical_cols,
                                              categorical_cols,
                                              model=clf,
                                              scaler=sc,
                                              path=path_folder,
                                              file_name=file_template)

# =============================================================================
# RuleFit
# =============================================================================
# Obtain rules
df_check, df_rules_outliers, df_rules_inliers = rulefit_rules(df_anomalies,
                                                              clf,
                                                              numerical_cols,
                                                              categorical_cols,
                                                              path=path_folder,
                                                              file_name=file_template)

# =============================================================================
# SkopeRules               
# =============================================================================
### Obtain Rules
(df_rules_info_inliers, df_rules_info_outliers,
 df_rules_inliers, df_rules_outliers,
 df_no_pruned, df_yes_pruned) = skoperules_rules(df_anomalies,
                                                 clf,
                                                 numerical_cols,
                                                 categorical_cols,
                                                 path=path_folder,
                                                 file_name=file_template)


# =============================================================================
# Rules from AIX360
# =============================================================================
#### Choose algorithm
for rule_algorithm in ["brlg", "logrr", "glrm"]:
    print("Rules for: {0}".format(rule_algorithm))

    ### Obtain Rules
    (df_rules_inliers, df_rules_outliers,
    df_no_pruned, df_yes_pruned) = aix360_rules_wrapper(df_anomalies,
                                                        clf,
                                                        numerical_cols,
                                                        categorical_cols,
                                                        use_oversampling=True,
                                                        rule_algorithm=rule_algorithm,
                                                        path=path_folder,
                                                        file_name=file_template)

# =============================================================================
# Falling Rule List (FRL)
# =============================================================================
### Obtain Rules
(df_rules_inliers, df_rules_outliers,
df_no_pruned, df_yes_pruned) = frl_rules(df_anomalies,
                                         clf,
                                         numerical_cols,
                                         categorical_cols,
                                         path=path_folder,
                                         file_name=file_template)
                                         
                                                        
                                                        
                                                        