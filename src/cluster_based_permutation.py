import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage

def cluster_based_permutation_test(condition_a, condition_b, critical_t, n_reps, percentile, random_seed = 22092023):
    """
    gets clusters above a critical t-value compares them to clusters arrising by chance
    condition_a: pandas data frame, with n (repetitions) rows and t (timepoints) columns.
    repetitions need to be in the same order (row 0 in condition a is from the same participant as row 0 in condition b)
    """
    # clusters in our data
    condition_difference = condition_a - condition_b
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
        permutated_data = data.mul(random_permutation_matrix[rep], axis = 0)
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

def find_clusters(values_to_compare, critical_value):
    """
    find a cluster of t-values above the critial threshold
    """
    all_clusters = {}
    over_critical = abs(values_to_compare) >= critical_value
    over_critical_positions, n_clusters = ndimage.label(over_critical)

    for cluster in range(n_clusters):
        cluster_id = cluster+1
        cluster_location = np.where(over_critical_positions == cluster_id)
        current_cluster = values_to_compare.values[cluster_location]
        all_clusters[cluster] = {}
        all_clusters[cluster]['cluster_id'] = cluster_id
        all_clusters[cluster]['cluster_size'] = len(current_cluster)
        all_clusters[cluster]['cluster_weight'] = abs(sum(current_cluster))
        all_clusters[cluster]['cluster_location'] = cluster_location
    return all_clusters


def t_stats(values):
    """
    takes a data frame (values) and returns the t-statistic across rows
    """
    return np.mean(values, axis = 0)/(np.std(values, axis = 0)/values.shape[0])
