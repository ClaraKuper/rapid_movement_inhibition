import src.movement_rates as mov
input_file = './tests/normalized_rates.csv'
baseline_name = 'flash- jump-'
t_value = 2.093
n_permutation = 1000
percentile_cutoff = 0.05
output_path = './tests/'
output_file = 'movement_cluster.csv'


mov.perform_cluster_based_permutation(input_file, baseline_name, t_value, n_permutation, percentile_cutoff,
                                      output_path, output_file, condition_column='condition')
