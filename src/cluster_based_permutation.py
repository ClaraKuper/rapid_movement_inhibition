import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage

def cluster_based_permutation_test(condition_a, condition_b, critical_t, n_reps, percentile, random_seed = 22092023):
    """
    gets clusters above a critical t-value compares them to clusters arrising by chance
    condition_a: pandas data frame, with n (repetitions) rows and t (timepoints) columns.
    repetitions need to be in the same order (row 0 in condition a is from the same participant as row 0 in condition b)
    a random seed is set for replicability

    returns
    clusters: all clusters above a critical t-value
    cutoff_value: the value of cluster mass that denotes given upper percentile
    cluster_over_thresh: information of all clusters that are over the critical cluster mass
    """
    # clusters in our data
    condition_difference = condition_a - condition_b
    if condition_difference.shape[0]>condition_difference.shape[1]:
        print(f"The data frame seems to have more rows (repetitions of the measurement) - "
                                f"{condition_difference.shape[0]} rows - "
                                f"than columns (time points) - {condition_difference.shape[1]} columns - in the "
                                f"measurement. \nPlease make sure that this is correct.")
    t_values = t_stats(condition_difference)
    clusters = find_clusters(t_values, critical_t)
    cluster_df = pd.DataFrame.from_dict(clusters).T
    permutated_clusters, cutoff_value = random_permutation(condition_difference, critical_t, n_reps, percentile, random_seed)
    cluster_over_thresh = cluster_df[cluster_df['cluster_weight'] > cutoff_value]

    return clusters, cutoff_value, cluster_over_thresh


def random_permutation(data, critical_t, n_reps, percentile, random_seed):
    """
    takes an array, flips conditions randomly and computes sizes of clusters
    that arrise by chance.

    "data" is the difference between the conditions
    """
    n_subjects = data.shape[0]
    random_permutation_matrix = get_random_permutation_matrix(n_subjects, n_reps, random_seed)

    permutated_cluster_df = pd.DataFrame(columns = ['clusterID', 'nRep', 'value'])

    for rep in range(n_reps):
        if len(data.shape) == 2:
            permutated_data = data.mul(random_permutation_matrix[rep], axis = 0)
        if len(data.shape) == 3:
            permutated_data = []
            for idx, part in enumerate(data):
                permutated_data.append(part*random_permutation_matrix[rep][idx])
            permutated_data = np.array(permutated_data)
        t_values = t_stats(permutated_data)
        clusters = find_clusters(t_values, critical_t)
        if len(clusters)>0:
            df_cluster = pd.DataFrame.from_dict(clusters).T
            largest_cluster = np.argmax(df_cluster['cluster_weight'])

            permutated_cluster_df.loc[rep, 'clusterID'] = df_cluster['cluster_id'][largest_cluster]
            permutated_cluster_df.loc[rep, 'nRep'] = rep
            permutated_cluster_df.loc[rep, 'value'] = df_cluster['cluster_weight'][largest_cluster]
    sorted_clusters = permutated_cluster_df['value'].values
    sorted_clusters.sort()
    percentile_cutoff = int((1-percentile) * n_reps)
    cutoff_value = sorted_clusters[percentile_cutoff]

    return permutated_cluster_df, cutoff_value

    # find cluster that maximizes measurement


def get_random_permutation_matrix(nrow, ncol, random_seed):
    """
    generate a matrix with random +1/-1 entries
    """
    np.random.seed(random_seed)
    random_permutation_matrix = pd.DataFrame(np.random.random([nrow, ncol]) - 0.5)
    random_permutation_matrix = np.sign(random_permutation_matrix)
    return random_permutation_matrix

def find_clusters(values_to_compare, critical_value, ignore_inf = True):
    """
    find a cluster of values above the critial threshold
    """
    all_clusters = {}
    over_critical = abs(values_to_compare) >= critical_value
    over_critical_positions, n_clusters = ndimage.label(over_critical)

    for cluster in range(n_clusters):
        cluster_id = cluster+1
        cluster_location = np.where(over_critical_positions == cluster_id)
        current_cluster = values_to_compare[cluster_location]
        if ignore_inf:
            current_cluster = current_cluster[abs(current_cluster)!=np.inf]
        all_clusters[cluster] = {}
        all_clusters[cluster]['cluster_id'] = cluster_id
        all_clusters[cluster]['cluster_size'] = len(current_cluster)
        all_clusters[cluster]['cluster_weight'] = abs(np.nansum(current_cluster))
        if all_clusters[cluster]['cluster_weight'] == np.inf or all_clusters[cluster]['cluster_weight'] == np.nan:
            raise ValueError(f"The cluster weight could not be computed. "
                             f"Cluster weight is {all_clusters[cluster]['cluster_weight']}")

        all_clusters[cluster]['cluster_location'] = cluster_location
    return all_clusters


def t_stats(values):
    """
    takes a data frame (values) and returns the t-statistic across rows
    """
    return np.nanmean(values, axis = 0)/(np.nanstd(values, axis = 0)/values.shape[0])
