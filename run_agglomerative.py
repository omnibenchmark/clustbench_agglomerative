#!/usr/bin/env python

"""
Omnibenchmark-izes Marek Gagolewski's https://github.com/gagolews/clustering-results-v1/blob/eae7cc00e1f62f93bd1c3dc2ce112fda61e57b58/.devel/do_benchmark_agglomerative.py

Takes the true number of clusters into account and outputs a 2D matrix with as many columns as ks tested,
being true number of clusters `k` and tested range `k plusminus 2`

linkage_matrix computations are based on the codenby Mathew Kallada, Andreas Mueller (License: BSD 3 clause)
https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
"""

import os, sys
import sklearn.cluster
import numpy as np
import warnings
import scipy.cluster.hierarchy
import argparse

VALID_LINKAGES = ['average', 'complete', 'ward']

def load_labels(data_file):
    data = np.loadtxt(data_file, ndmin=1)
    
    if data.ndim != 1:
        raise ValueError("Invalid data structure, not a 1D matrix?")
    
    return(data)

def load_dataset(data_file):
    data = np.loadtxt(data_file, ndmin=2)
    
    ##data.reset_index(drop=True,inplace=True)
    
    if data.ndim != 2:
        raise ValueError("Invalid data structure, not a 2D matrix?")
    
    return(data)

## this maps the ks to their true offset to the truth, e.g.:
# >>> generate_k_range(5)
# {'k-2': 3, 'k-1': 4, 'k': 5, 'k+1': 6, 'k+2': 7}
# >>> generate_k_range(1)
# {'k-2': 2, 'k-1': 2, 'k': 2, 'k+1': 2, 'k+2': 3}
# >>> generate_k_range(2)
# {'k-2': 2, 'k-1': 2, 'k': 2, 'k+1': 3, 'k+2': 4}
## k is the true k
def generate_k_range(k):
    Ks = [k-2, k-1, k, k+1, k+2] # ks tested, including the true number
    replace = lambda x: x if x >= 2 else 2 ## but we never run k < 2; those are replaced by a k=2 run (to not skip the calculation)
    Ks = list(map(replace, Ks))
    
    # ids = ['k-2', 'k-1', 'k', 'k+1', 'k+2']
    ids = list(range(0,5))
    assert(len(ids) == len(Ks))
    
    k_ids_dict = dict.fromkeys(ids, 0)
    for i in range(len(ids)):
        key = ids[i]
        
        k_ids_dict[key] = Ks[i]
    return(k_ids_dict)

def do_agglomerative(X, Ks, linkage):
    res = dict()

    c = sklearn.cluster.AgglomerativeClustering(
        distance_threshold=0,
        n_clusters=None,
        compute_full_tree=True,
        linkage=linkage
    )
    c.fit(X)

    #```````````````````````````````````````````````````````````````````````````
    # See https://scikit-learn.org/stable/_downloads/6c3126e55d97d68efdd8da229311ac00/plot_agglomerative_dendrogram.py
    # create the counts of samples under each node::
    counts = np.zeros(c.children_.shape[0])
    n_samples = len(c.labels_)
    for i, merge in enumerate(c.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    #```````````````````````````````````````````````````````````````````````````

    linkage_matrix = np.column_stack([c.children_, c.distances_,
                                      counts]).astype(float)

    # for k in range(len(Ks)):
    for item in Ks.keys():
        K_id = item  ## just an unique identifier
        K = Ks[K_id] ## the tested k perhaps repeated

        labels_pred_matrix = scipy.cluster.hierarchy.\
            cut_tree(linkage_matrix, n_clusters=K) # 0-based -> 1-based!!!

        labels_pred_matrix +=1 ## 0-based -> 1-based
        res[K_id] = labels_pred_matrix[:,-1]
        
    return np.array([res[key] for key in res.keys()]).T

def main():
    parser = argparse.ArgumentParser(description='clustbench sklearn agglomerative runner')

    parser.add_argument('--data_matrix', type=str,
                        help='gz-compressed textfile containing the comma-separated data to be clustered.', required = True)
    parser.add_argument('--data_true_labels', type=str,
                        help='gz-compressed textfile with the true labels; used to select a range of ks.', required = True)
    parser.add_argument('--output', action='append', default=[],
                        help='id=PATH, one per output the stage declares (api 0.8.0).')
    parser.add_argument('--linkage', type=str,
                        help='linkage',
                        required = True)
    # ob still passes --output_dir and --name until api 0.10; they are ignored.
    args, _ = parser.parse_known_args()
    outputs = dict(o.split('=', 1) for o in args.output)

    if args.linkage not in VALID_LINKAGES:
        raise ValueError(f"Invalid method `{args.linkage}`")

    truth = load_labels(args.data_true_labels)
    k = int(max(truth)) # true number of clusters
    Ks = generate_k_range(k)
    
    data = args.data_matrix
    curr = do_agglomerative(X= load_dataset(data), Ks = Ks, linkage = args.linkage)
    

    header=['k=%s'%s for s in Ks.values()]
    
    curr = np.append(np.array(header).reshape(1,5), curr.astype(str), axis=0)
    np.savetxt(outputs['clustering_predicted_ks_range'],
               curr, fmt='%s', delimiter=",")#,
               # header = ','.join(header)) 

if __name__ == "__main__":
    main()
