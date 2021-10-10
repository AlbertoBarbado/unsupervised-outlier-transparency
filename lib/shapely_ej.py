# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 10:41:21 2020

@author: alber
"""
import pandas as pd

from shapely.geometry import Polygon
polygon = Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])
other_polygon = Polygon([(1, 1), (4, 1), (4, 3.5), (1, 3.5)])
intersection = polygon.intersection(other_polygon)
print(intersection.area)


p = Polygon([[1,1,1],[2,1,1],[2,2,1],[2,1,2],
             [2,2,2],[1,2,2],[1,2,1],[1,1,2]])

q = Polygon([[2.5,1.5,2.5],[2.5,2.5,2.5],[1.5,1.5,2.5],[1.5,2.5,2.5],
             [1.5,1.5,1.5],[1.5,2.5,1.5],[2.5,2.5,1.5],[2.5,1.5,1.5]])


intersection = p.intersection(q)
print(intersection.area)



polygon = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
other_polygon = Polygon([(1.5, 1.5), (3, 1.5), (1.5, 0.5), (1.5, 0.5)])
intersection = polygon.intersection(other_polygon)
print(intersection.area)



from itertools import combinations, permutations, product
# combinations: ["AB", "AC", "BC"]
# permutations: ["AB", "AC", "BC", "BA", "CA", "CB"]

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


# Load rules
df_rules = pd.read_csv("results/seismic_2D/df_rules_seismic_2D_kernel_rbf_culstering_kmeans_inliers_method_discard.csv")
list_cols_rules = list(df_rules.columns)
list_cols = ['gdenergy', 'gdpuls']
# Obtain combinations of features to create the 2D planes
comb_free_features = [c for c in combinations(list_cols, 2)]
comb_limits = [c for c in combinations(['_max', '_min'], 2)]
# Filter rules for each categorical combination state
# TODO

# Obtain the vectors (dataframe) for each combination of 2 free features and n-2 fixed ones (n size hyperspace)
df_final = pd.DataFrame()
for comb in comb_free_features:
    # Specify cols
    list_free = [col + '_max' for col in comb] + [col + '_min' for col in comb] # Cols to change
    cols_fixed = [col for col in list_cols if 'max' in col and col not in comb and col not in list_categorical] # Cols to mantain
    cols_fixed = cols_fixed + list_categorical # categorical do not use _max or _min
    
    list_x1 = [comb[0] + '_max', comb[0] + '_min']
    list_x2 = [comb[1] + '_max', comb[1] + '_min']
    cols_free = list(product(list_x1, list_x2))
     # Vertices of the 2D planes
    comb_vertices = [tuple(list(x) + [j for j in cols_fixed]) for x in cols_free]
    
    # Obtain polygons for those 2D planes
    list_aux = [[tuple(vector[list(x)].values) for x in comb_vertices] for _,vector in df_rules[list_free+cols_fixed].iterrows()]
    list_aux = [sort_vertices(list_vertex) for list_vertex in list_aux]
    polys = [Polygon(x) for x in list_aux]
       
    # Compute intersections of parallel 2D planes
    df_polys = pd.DataFrame({"rule_id":list(df_rules.index),
                             "rule_vertices":[x.bounds for x in polys]})
    df_results = pd.DataFrame({"rule_vertices":[pair[0].bounds for pair in permutations(polys, 2)],
                                # "rule_2":[pair[1].bounds for pair in permutations(polys, 2)],
                              "score":[pair[0].intersection(pair[1]).area for pair in permutations(polys, 2)]})
    df_results = (df_results.merge(df_polys, how="left",
                                   right_on=['rule_vertices'],
                                   left_on=['rule_vertices'])
                             .drop(['rule_vertices'], axis=1))
    
    # Annotate the rules with intersections in this 2D subspace
    df_results['n_intersects'] = df_results.apply(lambda x: 0 if x['score']==0 else 1, axis=1)
    
    # Keep iter results
    if df_final.empty:
        df_final = df_results
        df_final = df_final.rename(columns={'score':'score_total',
                                            'n_intersects_total':'n_intersects'})
        df_final
    else:
        df_final = df_final.merge(df_results)
        df_final['score_total'] += df_final['score']
        df_final['n_intersects_total'] += df_final['n_intersects']
        df_final = df_final.drop(['score', 'n_intersects'], axis=1)
    






i = 0
for pair in combinations(polys, 2):
    print(i)
    i += 1
    pair[0].intersection(pair[1]).area


df_rules[["gdenergy_max", "gdpuls_max"] + resto_columnas]
# gather the 15 Shapely polygons in something iterable, like a list
list_p = [[1,1,1],[2,1,1],[2,2,1],[2,1,2],
             [2,2,2],[1,2,2],[1,2,1],[1,1,2]]
list_q = 
shapes = [poly1, poly2, ..., poly15]

# test the intersection on the combinations of pairs
intersections = [pair[0].intersects(pair[1]) for pair in combinations(shapes, 2)]
# intersections is a list with 105 elements of True or False

if any(intersections):
    print("yes, %d of %d combinations intersect" % (sum(intersections), len(intersections)))
else:
    print("no intersections")
    
    
Polygon([(-68, 15), (37, 15), (37, -75), (-68, -75)]) # Good sorting!





from aix360.algorithms.protodash import ProtodashExplainer, get_Gaussian_Data
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


using_inliers = True
df_data = df_anomalies[df_anomalies['predictions']==1].copy()

explainer = ProtodashExplainer()
list_cols = numerical_cols + categorical_cols
(W, S, _) = explainer.explain(df_data[list_cols].values,
                              df_data[list_cols].values,
                              m=np.round(len(df_rules)),
                              kernelType='Gaussian',
                              sigma=2)
df_prototypes = df_anomalies[df_anomalies.index.isin(list(S))][list_cols].reset_index(drop=True)


df_samples_total = pd.DataFrame()
base_size = len(df_anomalies)
for i, row in df_prototypes.iterrows():
    iter_size = np.round(base_size*W[i])
    iter_size = iter_size if iter_size > 10 else 10
    df_samples = pd.DataFrame({col:np.random.uniform(low=row[col]*0.9, high=row[col]*1.1, size=(iter_size,)) for col in list_cols})
    df_samples['prototype_id'] = i
    df_samples_total = df_samples_total.append(df_samples)
    

# Check two things:
# 1) Classifications are the same for all similar neighbours of a prototype
# 2) Rules give the same output as the original model (-1 if not included in any rule, 1 if included in at least one of them)
list_proto_id = list(df_samples_total['prototype_id'].unique())

precision_rules = 0
df_agree = pd.DataFrame()
j = 0
for proto_id in list_proto_id:
    df_proto_subset = df_samples_total[df_samples_total['prototype_id']==proto_id][list_cols]
    for i, data_point in df_proto_subset.iterrows():
        df_aux = pd.DataFrame(check_datapoint_inside(data_point, df_rules, numerical_cols, categorical_cols)['check'])
        # Check % of same classifications
        n_agree = len(df_aux[df_aux['check']==1])/len(df_aux)
        n_agree = n_agree if n_agree > 0.5 else 1-n_agree # Use the max % because we do not know what did the rules yield
        df_agree = df_agree.append(pd.DataFrame({'proto_id':proto_id, 'n_agree':n_agree}, index=[0]))
        # Check if the predictions are the same as the model
        y_model = clf.predict(data_point.values.reshape(1, -1))[0]
        if using_inliers & y_model==1:
            # Here, predict value is 1
            j += 1
            if df_aux['check'].max() == 1: precision_rules += 1 # If inside any rule, check as correct
        elif y_model==-1:
            # Here, predict value is -1
            j += 1
            if df_aux['check'].max() == 1: precision_rules += 1 # If inside any rule, check as correct

precision_rules = precision_rules/j # % of points with the same values as the model  
n_agree_total = df_agree.groupby(by=['proto_id']).median().reset_index()['n_agree'].median()

y_pred = []
for i, data_point in df_samples_total.iterrows():
    y_pred.append(check_datapoint_inside(data_point, df_rules, numerical_cols, categorical_cols)['check'])




from sklearn.neighbors import NearestNeighbors
neigh = NearestNeighbors(2, 0.4)
neigh.fit(X_train)
df_nn = pd.DataFrame(neigh.kneighbors(X_train, 2, return_distance=False)).reset_index()
df_nn['nn'] = df_nn.apply(lambda x: x[1] if x[1] != x['index'] else x[0], axis=1)
list_nn = list(df_nn['nn'].drop_duplicates())


a = neigh.kneighbors(X_train)
# Scaling numerical data
sc = StandardScaler()

if len(numerical_cols):
    X_train = df_anomalies[numerical_cols]
    X_train = sc.fit_transform(X_train)
else:
    X_train = dataset_mat

X_train_model = X_train

i = 0
for col in categorical_cols:
    i += 1
    print(i, "/", len(categorical_cols))
    X_train_model = np.insert(
        X_train_model,
        np.shape(X_train_model)[1],
        dataset_mat[col].values,
        axis=1)

clustering = DBSCAN(eps=0.1,
                    metric="euclidean",
                    min_samples=20).fit(X_train_model)
core_samples = clustering.core_sample_indices_
cluster_labels = clustering.labels_

kp = KPrototypes(n_clusters=5,
                        init='Huang',
                        max_iter=5,
                        n_init=5,
                        verbose=0)
labels = kp.fit_predict(df_anomalies[list_cols], categorical=[])
centroid = kp.cluster_centroids_







