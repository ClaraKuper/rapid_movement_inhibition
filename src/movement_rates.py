import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import src.helper_funcs as helper
import src.cluster_based_permutation as cmp
from src.json_parsing import filter_data
from src.plotting import plot_single_participant_rates


def get_movement_rates_by_participant(data, onset_column, offset_column, participant_column, analysis_parameter_dict,
                                      order_column, conditions_dict, condition_color_dict, linestyle_dict, baseline_name,
                                      result_path, movement_rate_cluster_file, plot=False):
    participants = np.unique(data[participant_column])
    ref_scale = np.arange(analysis_parameter_dict['window_start'], analysis_parameter_dict['window_end'], 1)
    movement_rates = {}
    rate_parameters = {}
    for p in participants:
        movement_rates[p] = {}
        p_data = data[data[participant_column] == p].reset_index(drop=True)

        for condition in conditions_dict:
            condition_dict = conditions_dict[condition]
            c_data = filter_data(p_data, condition_dict)
            if len(c_data) > 0:
                movement_rate_raw, movement_rate, scale = get_normalized_rates(c_data, ref_scale, onset_column,
                                                                               offset_column, order_column,
                                                                               analysis_parameter_dict)
                assert np.all(ref_scale == scale)
                movement_rates[p][condition] = movement_rate
            else:
                movement_rates[p][condition] = np.zeros(len(ref_scale))

    movement_rate_mean, normalized_rates = normalize_to_mean(movement_rates, baseline_name)

    # cluster based permutation test
    baseline_data, flash_no_jump_data, no_flash_jump_data, flash_jump_data = helper.get_dataframe_per_condition(normalized_rates,
                                                                                                         ['flash- jump-',
                                                                                                          'flash+ jump-',
                                                                                                          'flash- jump+',
                                                                                                          'flash+ jump+'])
    save_data = flash_no_jump_data.copy(deep = True)
    save_data.columns = ref_scale
    save_data.to_csv(f'{result_path}/movement_rates_flash_only.csv', index = False)

    significant_clusters = {}
    cluster_row = 0
    significant_clusters_1d = pd.DataFrame(columns=['condition', 'location', 'start_time',
                                                    'end_time', 'center_location', 'center_value',
                                                    'center_location_idx', 'cluster_weight', 'weight_cutoff', 'SEM'])
    for condition, condition_name in zip([flash_no_jump_data, no_flash_jump_data, flash_jump_data], ['flash+ jump-', 'flash- jump+', 'flash+ jump+']):
        clusters, cutoff_value, cluster_over_thresh = cmp.cluster_based_permutation_test(condition,
                                                                                         baseline_data,
                                                                                         2.093,
                                                                                         1000,
                                                                                         0.05,
                                                                                         f'{result_path}/rate_cluster_{condition_name}.csv') #3.579,
        significant_clusters[condition_name] = cluster_over_thresh.cluster_location

        for cluster_id in clusters:
            if len(clusters[cluster_id]['cluster_location']) > 0:
                if not all(np.isnan(clusters[cluster_id]['cluster_values'])):
                    if abs(clusters[cluster_id]['cluster_weight']) >= cutoff_value:
                        significant_clusters_1d.loc[cluster_row, 'condition'] = condition_name
                        significant_clusters_1d.loc[cluster_row, 'location'] = clusters[cluster_id]['cluster_location']
                        significant_clusters_1d.loc[cluster_row, 'start_time'] = ref_scale[min(
                            clusters[cluster_id]['cluster_location'])]
                        significant_clusters_1d.loc[cluster_row, 'end_time'] = ref_scale[max(
                            clusters[cluster_id]['cluster_location'])]

                        value_loc_df = pd.DataFrame(np.array([clusters[cluster_id]['cluster_location'],
                                                              clusters[cluster_id]['cluster_values']]).T,
                                                    columns=['location', 'values']).reset_index(drop=True)
                        weighted_average_location = round(helper.get_weighted_average(value_loc_df,
                                                                                      'location',
                                                                                      'values'))
                        significant_clusters_1d.loc[cluster_row, 'center_location'] = ref_scale[min(
                            weighted_average_location,
                            round(max(ref_scale)))]
                        significant_clusters_1d.loc[cluster_row, 'center_value'] = np.mean(condition, axis = 0)[min(
                            weighted_average_location,
                            round(max(ref_scale)))]
                        significant_clusters_1d.loc[cluster_row, 'center_location_idx'] = weighted_average_location
                        significant_clusters_1d.loc[cluster_row, 'cluster_weight'] = clusters[cluster_id]['cluster_weight']
                        significant_clusters_1d.loc[cluster_row, 'weight_cutoff'] = cutoff_value
                        significant_clusters_1d.loc[cluster_row, 'SEM'] = scipy.stats.sem(condition, axis = 0)[min(
                            weighted_average_location,
                            round(max(ref_scale)))]
                        cluster_row += 1
    significant_clusters_1d.to_csv(movement_rate_cluster_file, index=False)

    for p in participants:
        rate_parameters[p] = {}
        for condition in conditions_dict:
            condition_dict = conditions_dict[condition]
            parameter_dict = get_rate_parameters(normalized_rates[p][condition],
                                                 scale,
                                                 np.ones(len(scale)),
                                                 analysis_parameter_dict)
            rate_parameters[p][condition] = parameter_dict
            try:
                rate_parameters[p][condition]['flash_shown'] = conditions_dict[condition]['flashShown']
                rate_parameters[p][condition]['stim_jumped'] = conditions_dict[condition]['stimJumped']
            except KeyError:
                rate_parameters[p][condition]['flash_shown'] = conditions_dict[condition]['flash']
                rate_parameters[p][condition]['stim_jumped'] = conditions_dict[condition]['shift']

    if plot:
        plot_single_participant_rates(normalized_rates, scale, rate_parameters, condition_color_dict)
    return normalized_rates, movement_rates, rate_parameters, scale, significant_clusters


def get_rate_parameters(rate, scale, base_rate, parameters):
    dictionary = {}
    search_scale, search_rate = filter_rate_and_scale(scale, rate, parameters['search_start'], parameters['search_end'])
    baseline = get_baseline(base_rate)
    value, latency = get_highest_value_latency(search_rate-baseline, search_scale)
    magnitude = value
    dictionary['minimum'] = value + baseline
    dictionary['latency'] = latency
    dictionary['baseline'] = baseline
    dictionary['magnitude'] = magnitude
    return dictionary


def filter_rate_and_scale(scale, rate, low, high):
    idx = np.where((low <= scale) & (high >= scale))[0]
    filter_rate = np.array(rate)[idx]
    filter_scale = np.array(scale)[idx]

    return filter_scale, filter_rate


def get_highest_value_latency(rate, scale):
    max_value = max(abs(rate))
    idx = np.where(abs(rate) == max_value)
    latency = scale[idx]
    value = rate[idx]
    try:
        assert len(latency) == 1
    except AssertionError:
        print(latency)
    return value[0], latency[0]


def get_baseline(rate):
    return np.mean(rate, axis = 0)


def get_normalized_rates(data, scale, onset_column, offset_column, order_column, analysis_parameter_dict, n_trials = None):
    onsets = data[onset_column].dropna().astype(int).values
    offsets = data[offset_column].dropna().astype(int).values

    #first_touches = data[data[order_column] == min(data[order_column])][onset_column]
    #last_touches = data[data[order_column] == max(data[order_column])][onset_column]
    #last_touches = data[data[order_column] == max(data[order_column])][onset_column]
    smooth_distribution = np.ones(len(scale)) #get_uniform_cdf(min(first_touches), max(first_touches), min(last_touches), max(last_touches),
                                          #scale)
    if not n_trials:
        n_trials = 400 #len(first_touches)
    smooth_distribution = smooth_distribution * n_trials
    movement_rate_raw, movement_rate, scale = causal_rate(offsets, analysis_parameter_dict['window_start'],
                                                          analysis_parameter_dict['window_end'], smooth_distribution, analysis_parameter_dict['alpha'])

    return movement_rate_raw, movement_rate, scale

def normalize_to_mean(dictionary, key):
    """
    This function takes a dictionary of movement rates, and the key where the average should be recorded.
    Next, it computes the mean of all entries but the current. Returns a dictionary with these values
    """
    all_values = [dictionary[k][key] for k in dictionary]
    df_all_dict_values = pd.DataFrame(all_values)
    df_all_dict_values['key'] = [k for k in dictionary.keys()]
    means_dict = {}
    normalized_dict = {}
    for outer_key in np.unique(df_all_dict_values['key']):
        mean_rates = df_all_dict_values[df_all_dict_values['key'] != outer_key]
        means_dict[outer_key] = mean_rates.mean(axis=0, numeric_only=True).values
        normalized_dict[outer_key] = {}
        for inner_key in dictionary[outer_key]:
            value = dictionary[outer_key][inner_key]
            denominator = means_dict[outer_key]

            if len(value) != len(denominator):
                shorten_to = min(len(value), len(denominator))
                value = value[:shorten_to]
                denominator = denominator[:shorten_to]

            normalized_dict[outer_key][inner_key] = value/denominator

    return means_dict, normalized_dict


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
        elif val >= first_trial_start and val < last_trial_start:
            uniform_dist.append((val - first_trial_start) / (last_trial_start - first_trial_start))
        elif val >= last_trial_start and val < first_trial_end:
            uniform_dist.append(1)
        elif val >= first_trial_end and val < last_trial_end:
            uniform_dist.append(- 1 * ((val - last_trial_end) / (last_trial_end - first_trial_end)))
        elif val >= last_trial_end:
            uniform_dist.append(0.00001)
        else:
            raise ValueError(
                f'{val} is a very weird value. {first_trial_start, last_trial_start, first_trial_end, last_trial_end}')
    return np.array(uniform_dist)
