import numpy as np
import pandas as pd
import scipy.stats
import src.helper_funcs as helper
import src.cluster_based_permutation as cmp
from src.json_parsing import filter_data


def get_movement_rates_by_participant(input_file, output_path, output_file, onset_column, offset_column,
                                      participant_column,  order_column,
                                      analysis_parameter_dict, conditions_dict):
    data = pd.read_csv(input_file)

    participants = np.unique(data[participant_column])
    ref_scale = np.arange(analysis_parameter_dict['window_start'],
                          analysis_parameter_dict['window_end'], 1)

    column_names = ['participant', 'condition']
    [column_names.append(t) for t in ref_scale]
    movement_rates = pd.DataFrame(columns=column_names)
    row_idx = 0

    for p in participants:
        participant_data = data[data[participant_column] == p].reset_index(drop=True)

        for condition in conditions_dict:
            condition_dict = conditions_dict[condition]
            condition_data = filter_data(participant_data, condition_dict)
            movement_rate, scale = get_normalized_rates(condition_data, ref_scale,
                                                        onset_column, offset_column,
                                                        order_column, analysis_parameter_dict)

            # check if the scales are as they should be, break if they are not.
            assert np.all(ref_scale == scale)
            movement_rates.loc[row_idx, 'participant'] = p
            movement_rates.loc[row_idx, 'condition'] = condition
            movement_rates.loc[row_idx, scale] = movement_rate
            row_idx += 1

    # save movement rates as tables
    file_saved_at = helper.save_file(movement_rates, output_path, output_file)
    return file_saved_at


def mean_normalize_rates(input_file, output_path, output_file, baseline_name):
    data = pd.read_csv(input_file)
    normalized_rates = normalize_to_mean(data, baseline_name)

    file_saved_at = helper.save_file(normalized_rates, output_path, output_file)
    return file_saved_at


def normalize_to_mean(data, baseline, participant_column='participant', condition_column='condition'):
    """
    This function takes a dictionary of movement rates, and the key where the average should be recorded.
    Next, it computes the mean of all entries but the current. Returns a dictionary with these values
    """
    all_participants = np.unique(data[participant_column])

    for participant in all_participants:
        exclude_participant_data = data[data[participant_column] != participant]
        exclude_participant_baseline = exclude_participant_data[exclude_participant_data[condition_column] == baseline]
        baseline_average = exclude_participant_baseline.mean(axis=0, numeric_only=True)
        data.loc[data[participant_column] == participant, baseline_average.index] /= baseline_average.values

    return data


def save_single_condition_rate(input_file, condition_name, output_path, output_file, condition_column='condition'):
    data = pd.read_csv(input_file)
    filtered_data = data[data[condition_column] == condition_name]
    filtered_data = filtered_data.reset_index(drop=True)

    file_saved_at = helper.save_file(filtered_data, output_path, output_file)
    return file_saved_at


def perform_cluster_based_permutation(input_file, output_path, output_file, baseline_name,
                                      t_value, n_permutation, percentile_cutoff, condition_column='condition'):

    data = pd.read_csv(input_file)
    baseline_data = data[data[condition_column] == baseline_name].reset_index(drop=True)
    test_data = data[data[condition_column] != baseline_name]
    conditions = np.unique(test_data[condition_column])
    cluster_df = pd.DataFrame(columns=['condition', 'location', 'start_time', 'end_time',
                                       'center_location', 'center_value', 'center_location_idx',
                                       'cluster_weight', 'cutoff_weight', 'SEM'])
    ref_scale = baseline_data.select_dtypes(['number']).columns.values.astype(int)
    for condition in conditions:
        condition_data = test_data[test_data[condition_column] == condition].reset_index(drop=True)
        clusters, cutoff_value, \
            cluster_over_thresh = cmp.cluster_based_permutation_test(condition_data.select_dtypes(['number']),
                                                                     baseline_data.select_dtypes(['number']),
                                                                     t_value,
                                                                     n_permutation,
                                                                     percentile_cutoff)
        cluster_df = write_cluster_to_df(clusters, cluster_df, condition_data, ref_scale, cutoff_value, condition)

    file_saved_at = helper.save_file(cluster_df, output_path, output_file)
    return file_saved_at


def get_movement_rate_parameters(input_file,
                                 output_path,
                                 output_file,
                                 parameter_dict,
                                 participant_column='participant',
                                 condition_column='condition'):
    rate_data = pd.read_csv(input_file)
    participants = np.unique(rate_data[participant_column])
    conditions = np.unique(rate_data[condition_column])
    parameter_data = pd.DataFrame(columns=['participant', 'condition', 'stim_jumped', 'flash_shown', 'max_deviance',
                                           'latency'])
    row = 0
    for participant in participants:
        participant_data = rate_data[rate_data[participant_column] == participant]
        for condition in conditions:
            condition_data = participant_data[participant_data[condition_column] == condition]
            numeric_data = condition_data.select_dtypes(['number'])
            assert len(numeric_data.index) == 1
            search_data = numeric_data.loc[:, np.arange(parameter_dict['search_start'],
                                                        parameter_dict['search_end']).astype(str)]
            assert len(search_data.idxmax(axis=1).values) == 1
            parameter_data.loc[row, 'participant'] = participant
            parameter_data.loc[row, 'condition'] = condition
            parameter_data.loc[row, 'stim_jumped'] = int('jump+' in condition)
            parameter_data.loc[row, 'flash_shown'] = int('flash+' in condition)
            parameter_data.loc[row, 'max_deviance'] = search_data.max(axis=1).values[0]
            parameter_data.loc[row, 'latency'] = search_data.idxmax(axis=1).values[0]
            row += 1
    file_saved_at = helper.save_file(parameter_data, output_path, output_file)
    return file_saved_at


def write_cluster_to_df(clusters, df, condition_data, scale, cutoff_value, condition):
    cluster_row = len(df)
    for cluster_id in clusters:
        current_cluster = clusters[cluster_id]
        if len(current_cluster['cluster_location']) > 0:
            if not all(np.isnan(clusters[cluster_id]['cluster_values'])):
                # get the weighted center of the cluster
                value_loc_df = pd.DataFrame(np.array([current_cluster['cluster_location'],
                                                      current_cluster['cluster_values']]).T,
                                            columns=['location', 'values']).reset_index(drop=True)
                weighted_average_location = round(helper.get_weighted_average(value_loc_df,
                                                                              'location',
                                                                              'values'))
                # write everything to the data frame
                df.loc[cluster_row, 'condition'] = condition
                df.at[cluster_row, 'location'] = current_cluster['cluster_location']
                df.loc[cluster_row, 'start_time'] = scale[min(current_cluster['cluster_location'])]
                df.loc[cluster_row, 'end_time'] = scale[max(current_cluster['cluster_location'])]
                df.loc[cluster_row, 'center_location'] = scale[min(weighted_average_location,
                                                                   round(max(scale)))]
                df.loc[cluster_row, 'center_value'] = condition_data.mean(axis=0,
                                                                          numeric_only=True)[min(weighted_average_location,
                                                                                                 round(max(scale)))]
                #print(  df.loc[cluster_row, 'center_value'])
                df.loc[cluster_row, 'center_location_idx'] = weighted_average_location
                df.loc[cluster_row, 'cluster_weight'] = current_cluster['cluster_weight']
                df.loc[cluster_row, 'cutoff_weight'] = cutoff_value
                #print(len(scipy.stats.sem(condition_data.iloc[:, 2:], axis = 0)))
                #print(scipy.stats.sem(condition_data.iloc[:, 2:], axis = 0)[min(weighted_average_location, round(max(scale)))])
                df.loc[cluster_row, 'SEM'] = scipy.stats.sem(condition_data.iloc[:, 2:], axis = 0)[min(weighted_average_location, round(max(scale)))]
                cluster_row += 1
    return df


def get_highest_value_latency(rate, scale):
    max_value = max(abs(rate))
    idx = np.where(abs(rate) == max_value)
    latency = scale[idx]
    value = rate[idx]
    assert len(latency) == 1
    return value[0], latency[0]


def get_normalized_rates(data, scale, onset_column, offset_column,
                         order_column, analysis_parameter_dict,
                         n_trials=None):
    offsets = data[onset_column].dropna().astype(int).values
    first_touches = data[data[order_column] == analysis_parameter_dict['first_offset']][offset_column]
    last_touches = data[data[order_column] == analysis_parameter_dict['last_offset']][offset_column]
    smooth_distribution = get_uniform_cdf(min(first_touches), max(first_touches), min(last_touches), max(last_touches),
                                          scale)
    if not n_trials:
        n_trials = len(first_touches)
    smooth_distribution = smooth_distribution * n_trials
    movement_rate_raw, movement_rate, scale = causal_rate(offsets, analysis_parameter_dict['window_start'],
                                                          analysis_parameter_dict['window_end'], smooth_distribution,
                                                          analysis_parameter_dict['alpha'])

    return movement_rate, scale


def causal_rate(move_onset, lock_window_start, lock_window_end, n_trials, alpha):
    """
     analyse rate in causal time window

     input:    move_onset  - movement onset times
               lock_window_start  - window before lock
               lock_window_end  - window after lock
               n_trials      - number of trials

     output:   rate    - movement rate
               scale   - time axis

    12.12.2005 by Martin Rolfs
    21.06.2021 translated to python by Clara Kuper
    """
    scale = np.arange(lock_window_start, lock_window_end, 1)
    # check how many trials these values came from
    if type(n_trials) == int:
        n_trials = np.linspace(n_trials, n_trials, len(scale))
    elif len(n_trials) != len(scale):
        raise ValueError('n_trials must have the same as the length of lock_window_start:lock_window_end!'
                         f'But has length {len(n_trials)} instead of {len(scale)}')
    # alpha defines how much the distribution is shifted
    alpha = alpha
    # define empty arrays for scale and rate
    rate = []
    raw_rate = []

    # loop through all time windows
    for idx, t in enumerate(scale):
        # compute tau
        # here is a filter for all events BEFORE time point t
        tau = t - move_onset + 1 / alpha
        # filter tau as event 0/1
        tau = tau[tau > 0]
        # get the number of saccades in a given window
        causal = alpha ** 2 * tau * np.exp(-alpha * tau)
        # save the rate
        rate.append(sum(causal) * 1000 / n_trials[idx])
        raw_rate.append(sum(causal) * 1000)
    return raw_rate, rate, scale


def get_uniform_cdf(first_trial_start, last_trial_start, first_trial_end, last_trial_end, scale):
    uniform_dist = []
    for val in scale:
        if val < first_trial_start:
            uniform_dist.append(0.00001)
        elif first_trial_start <= val < last_trial_start:
            uniform_dist.append((val - first_trial_start) / (last_trial_start - first_trial_start))
        elif last_trial_start <= val < first_trial_end:
            uniform_dist.append(1)
        elif first_trial_end <= val < last_trial_end:
            uniform_dist.append(- 1 * ((val - last_trial_end) / (last_trial_end - first_trial_end)))
        elif val >= last_trial_end:
            uniform_dist.append(0.00001)
        else:
            raise ValueError(
                f'{val} is a very weird value. {first_trial_start, last_trial_start, first_trial_end, last_trial_end}')
    return np.array(uniform_dist)
