import math
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import src.helper_funcs as helper
import src.cluster_based_permutation as cmp
import src.movement_rates as mr
from src.trial_by_trial_analysis import set_timings
from src.touch_position import get_fitted_responses
from src.json_parsing import set_data_type, filter_data
from src.plotting import make_figure_rates, plot_metrics, plot_average_participant_rates, \
    plot_average_participant_position, make_delay_figure
from statsmodels.stats.anova import AnovaRM
from scipy.stats import ttest_rel
from matplotlib.patches import Rectangle


def analysis_rates(raw_rates_input_file, results_output_path, raw_rates_output_filename,
                   mean_normalized_output_filename, rate_parameters_output_filename, permutest_output_filename,
                   onset_column, offset_column, participant_column, touch_order_column,
                   analysis_parameter_dict, permutation_dict, conditions_dict,
                   dependent_vars, independent_vars, baseline_condition_name):
    """
    Movement Rate Analysis
    - Step 1: Compute rates for individual participants
    - Step 2: Normalize rates to the mean baseline
    - Step 3: Get metrics from the rates
    - Step 4: Compare metrics statistically
    - Step 5: Compare rates with a cluster based permutation test

    This function calls all the above steps with parameters defined in the dictionaries (analysis_parameter_dict,
    permutation_dict, conditions_dict)

    raw_rates_input_file: path to the preprocessed data file (each touch is one row)
    results_ouput_path: path to the results folder
    raw_rates_output_filename: filename under which results from Step 1 are saved
    mean_normalized_output_filename: filename under which results from Step 2 are saved
    rate_parameters_output_filename: filename under which results from Step 3 are saved
    permutest_output_filename: filename under which results from Step 5 are saved

    Results from Step 4 are printed into the console.

    onset_column: the column name from raw_input_file that contains touch onsets relative to the change
    offset_column: the column name from raw_input_file that contains touch offsets relative to the change
    participant_column: the column name from raw_input_file that contains the participant id
    touch_order_column: the column name from raw_input_file that contains the order in which the taps were made

    analysis_parameter_dict: a dictionary defining the movement rate analysis. Must have the following keys:
    - window_start: start of the time scale (-700 in Kuper & Rolfs 2024)
    - window_end: end of the time scale (1500 in Kuper & Rolfs 2024)
    - alpha: the width of the moving causal kernel used to compute movement rates (1/50 in Kuper & Rolfs 2024)
    - search_start: start of the time window in which we look for movement parameters (0 in Kuper & Rolfs 2024)
    - search_end: end of the time window in which we look for movement parameters (700 in Kuper & Rolfs 2024)

    permutation_dict: a dictionary defining the permutation test. Must have the following keys:
    - baseline_name: name of the condition we want to compare to ('flash- jump-')
    - t_value: cutoff value for significant t-values (2.093)
    - n_permutations: the number of permutations to perform (1000)
    - percentile_cutoff: the upper percentile to consider (0.05)
    """

    # load the table with touch responses, save data file with movement rates
    raw_rates_output_path = mr.get_movement_rates_by_participant(raw_rates_input_file, results_output_path, raw_rates_output_filename,
                                                                 onset_column, offset_column, participant_column, touch_order_column,
                                                                 analysis_parameter_dict, conditions_dict)

    # load raw movement rates, save rates normalized to mean baseline and save
    normalized_rates_output_path = mr.mean_normalize_rates(raw_rates_output_path, results_output_path, mean_normalized_output_filename,
                                                           baseline_condition_name)

    # save parameters from the movement rates
    rate_parameters_output_path = mr.get_movement_rate_parameters(normalized_rates_output_path, results_output_path, rate_parameters_output_filename,
                                                                  analysis_parameter_dict)

    # run ANOVA on parameters
    run_anovas(rate_parameters_output_path, dependent_vars, independent_vars, 'participant')

    # perform a cluster-based permutation test to identify significant differences
    cluster_output_path = mr.perform_cluster_based_permutation(normalized_rates_output_path, results_output_path, permutest_output_filename,
                                                               permutation_dict['baseline_name'], permutation_dict['t_value'],
                                                               permutation_dict['n_permutations'], permutation_dict['percentile_cutoff'])


def analysis_position(data, x_col, y_col, target_x_col, target_y_col, x_full_length, y_full_length, pix2deg_by_name,
                      pix2deg_dictionary, participant_col, condition_dictionary, condition_color_dict,
                      condition_line_dict, time_col_name, params, dependent_vars, independent_vars_dict,
                      metrics_out_file, data_type_dict, results_path, figure_height=6):
    axs = make_figure_rates(figure_height, dependent_vars)
    data = set_data_type(data, data_type_dict)
    smoothed_response_positions, position_response_dictionary, \
        scale, data, cluster = get_fitted_responses(data, x_full_length, y_full_length, x_col, y_col,
                                                    target_x_col, target_y_col, pix2deg_by_name, pix2deg_dictionary,
                                                    participant_col, condition_dictionary, time_col_name, params,
                                                    helper.fit_sigmoid_func, results_path)
    metrics = helper.save_dict_as_table(position_response_dictionary, metrics_out_file, participant_col)
    helper.get_average_rates(smoothed_response_positions, scale, condition_dictionary, position_response_dictionary,
                             condition_color_dict, condition_line_dict, 0.95, axs['main'], cluster,
                             plot_average_participant_position)
    plot_metrics(metrics, dependent_vars, condition_color_dict, results_path)
    run_ttests(metrics, dependent_vars, independent_vars_dict)

    return data


def trial_by_trial_analysis(data, time_column, plot_column_dict, condition_dict, baseline_condition_dict, color_dict,
                            line_dict, participant_col, touch_on_col, touch_off_col, smooth_window_size, figure_name,
                            heatmap_parameters, heatmap_figure_path, cluster_1d_path, cluster_2d_path, result_path):
    participants = np.unique(data[participant_col])
    dictionary = {}
    heatmap_dictionary = {}
    test_data = set_timings(data, touch_on_col, touch_off_col)
    test_data = test_data[test_data.choiceOrder != 0]
    test_data = test_data.reset_index(drop=True)
    time = np.arange(min(test_data[test_data.stimJumped == 0][time_column]),
                     max(test_data[test_data.stimJumped == 0][time_column]))
    baseline_data = test_data.copy(deep=True)
    baseline_name = 'flash- jump-'

    for base_feat in baseline_condition_dict:
        baseline_data = baseline_data[baseline_data[base_feat] == baseline_condition_dict[base_feat]]
        baseline_data = baseline_data.reset_index(drop=True)

    for p in participants:
        p_data = test_data[test_data[participant_col] == p]
        p_base_data = baseline_data[baseline_data[participant_col] != p].reset_index(drop=True)
        p_base_heatmap = helper.make_heatmap(p_base_data, heatmap_parameters, time_column, 'flight_times').iloc[::-1]
        dictionary[p] = {}
        heatmap_dictionary[p] = {}

        for cond in condition_dict:
            dictionary[p][cond] = {}
            feature_dict = condition_dict[cond]
            feat_data = p_data.copy(deep=True)

            for feat in feature_dict:
                feat_data = feat_data[feat_data[feat] == feature_dict[feat]]
                feat_data = feat_data.reset_index(drop=True)

            for col in plot_column_dict:
                val = [helper.smooth_array(feat_data[plot_column_dict[col]], feat_data[time_column],
                                           smooth_window_size, t_val) for t_val in time]
                dictionary[p][cond][col] = val

            heatmap = helper.make_heatmap(feat_data, heatmap_parameters, time_column, 'flight_times').iloc[::-1]
            heatmap_dictionary[p][cond] = heatmap

    # cluster based permutation
    significant_clusters = {}
    significant_clusters_1d = pd.DataFrame(columns=['time_type', 'condition', 'location', 'start_time',
                                                    'end_time', 'center_location', 'center_value',
                                                    'cluster_weight', 'weight_cutoff'])
    cluster_row = 0
    time_length = []
    for col in plot_column_dict:
        significant_clusters[col] = {}
        base_data = pd.DataFrame([dictionary[p][baseline_name][col] for p in dictionary])
        time_length.append(base_data.dropna(axis=1).shape[1])
        for condition in condition_dict:
            if condition == baseline_name:
                continue
            else:
                warnings.simplefilter("ignore")
                contrast_data = pd.DataFrame([dictionary[p][condition][col] for p in dictionary])
                time_length.append(contrast_data.dropna(axis=1).shape[1])
                clusters, cutoff_value, \
                    cluster_over_thresh = cmp.cluster_based_permutation_test(contrast_data,
                                                                             base_data,
                                                                             2.093,#3.579,
                                                                             1000,
                                                                             0.05,
                                                                             f'{result_path}/{col}_time_{condition}.csv')
                significant_clusters[col][condition] = cluster_over_thresh.cluster_location
                for cluster_id in clusters:
                    if len(clusters[cluster_id]['cluster_location']) > 0:
                        if not all(np.isnan(clusters[cluster_id]['cluster_values'])):
                            significant_clusters_1d.loc[cluster_row, 'time_type'] = col
                            significant_clusters_1d.loc[cluster_row, 'condition'] = condition
                            significant_clusters_1d.loc[cluster_row, 'location'] = clusters[cluster_id]['cluster_location']
                            significant_clusters_1d.loc[cluster_row, 'start_time'] = time[min(
                                clusters[cluster_id]['cluster_location'])]
                            significant_clusters_1d.loc[cluster_row, 'end_time'] = time[max(
                                clusters[cluster_id]['cluster_location'])]

                            value_loc_df = pd.DataFrame(np.array([clusters[cluster_id]['cluster_location'],
                                                                  clusters[cluster_id]['cluster_values']]).T,
                                                        columns=['location', 'values']).reset_index(drop=True)
                            weighted_average_location = round(helper.get_weighted_average(value_loc_df,
                                                                                          'location',
                                                                                          'values'))
                            significant_clusters_1d.loc[cluster_row, 'center_location'] = time[min(
                                weighted_average_location,
                                round(max(time)))]
                            significant_clusters_1d.loc[cluster_row, 'center_value'] = np.mean(contrast_data)[min(
                                weighted_average_location,
                                round(max(time)))]
                            significant_clusters_1d.loc[cluster_row, 'cluster_weight'] = clusters[cluster_id]['cluster_weight']
                            significant_clusters_1d.loc[cluster_row, 'weight_cutoff'] = cutoff_value
                            cluster_row += 1

    significant_clusters_1d.to_csv(cluster_1d_path)

    significant_heatmap_clusters = {}
    significant_clusters_2d = pd.DataFrame(columns=['condition', 'location', 'start_time',
                                                    'end_time', 'start_latency', 'end_latency', 'center_time',
                                                    'center_latency', 'center_time_loc', 'center_latency_loc',
                                                    'center_value', 'cluster_weight', 'weight_cutoff'])
    row_idx = 0
    baseline_heatmap = np.array([heatmap_dictionary[x][baseline_name] for x in heatmap_dictionary])
    baseline_heatmap[np.where(np.isnan(baseline_heatmap))] = 0
    participants = [x for x in heatmap_dictionary]
    for cond in condition_dict:
        if cond == baseline_name:
            continue
        else:
            condition_heatmap = np.array([heatmap_dictionary[x][cond] for x in heatmap_dictionary])
            condition_heatmap[np.where(np.isnan(condition_heatmap))] = 0
            hm_clusters, hm_cutoff_value, \
                hm_cluster_over_thresh = cmp.cluster_based_permutation_test(condition_heatmap,
                                                                            baseline_heatmap,
                                                                            2.093,#3.579,
                                                                            1000,
                                                                            0.05,
                                                                            f'{result_path}/flight_heatmap_{condition}.csv',
                                                                            '2d')
            significant_heatmap_clusters[cond] = hm_cluster_over_thresh.cluster_location
            hm_time = heatmap_dictionary[participants[0]][baseline_name].columns.values
            hm_latency = heatmap_dictionary[participants[0]][baseline_name].index.values

            for cluster_id in hm_clusters:
                if len(hm_clusters[cluster_id]['cluster_location']) > 1:
                    if not all(np.isnan(hm_clusters[cluster_id]['cluster_values'])):
                        significant_clusters_2d.loc[row_idx, 'condition'] = cond
                        significant_clusters_2d.loc[row_idx, 'location'] = hm_clusters[cluster_id]['cluster_location']
                        significant_clusters_2d.loc[row_idx, 'start_time'] = hm_time[min(
                            hm_clusters[cluster_id]['cluster_location'][1])]
                        significant_clusters_2d.loc[row_idx, 'end_time'] = hm_time[max(
                            hm_clusters[cluster_id]['cluster_location'][1])]
                        significant_clusters_2d.loc[row_idx, 'start_latency'] = hm_latency[max(
                            hm_clusters[cluster_id]['cluster_location'][0])]
                        significant_clusters_2d.loc[row_idx, 'end_latency'] = hm_latency[min(
                            hm_clusters[cluster_id]['cluster_location'][0])]

                        value_loc_df_time = pd.DataFrame(np.array([hm_clusters[cluster_id]['cluster_location'][1],
                                                                   hm_clusters[cluster_id]['cluster_values']]).T,
                                                         columns=['location', 'values']).reset_index(drop=True)
                        weighted_average_location_time = round(helper.get_weighted_average(value_loc_df_time,
                                                                                           'location',
                                                                                           'values'))
                        significant_clusters_2d.loc[row_idx, 'center_time'] = hm_time[weighted_average_location_time]
                        significant_clusters_2d.loc[row_idx, 'center_time_loc'] = weighted_average_location_time

                        value_loc_df_latency = pd.DataFrame(np.array([hm_clusters[cluster_id]['cluster_location'][0],
                                                                      abs(hm_clusters[cluster_id]['cluster_values'])]).T,
                                                            columns=['location', 'values']).reset_index(drop=True)
                        weighted_average_location_latency = round(helper.get_weighted_average(value_loc_df_latency,
                                                                                              'location',
                                                                                              'values'))
                        significant_clusters_2d.loc[row_idx, 'center_latency'] = hm_latency[
                            weighted_average_location_latency]
                        significant_clusters_2d.loc[row_idx, 'center_latency_loc'] = weighted_average_location_latency
                        significant_clusters_2d.loc[row_idx, 'center_value'] = np.mean(condition_heatmap, axis=0)[
                            weighted_average_location_latency][weighted_average_location_time]
                        significant_clusters_2d.loc[row_idx, 'cluster_weight'] = hm_clusters[cluster_id]['cluster_weight']
                        significant_clusters_2d.loc[row_idx, 'weight_cutoff'] = hm_cutoff_value
                        row_idx += 1

    significant_clusters_2d.to_csv(cluster_2d_path, index=False)
    make_delay_figure(dictionary, test_data, condition_dict, plot_column_dict.keys(), time, significant_clusters,
                      significant_heatmap_clusters, color_dict, line_dict, figure_name,
                      heatmap_dictionary, heatmap_figure_path,
                      significant_clusters_2d[abs(significant_clusters_2d.cluster_weight) > significant_clusters_2d.weight_cutoff])

    return time, dictionary


def landing_position_analysis(data, x_touch, y_touch, x_dot_first, y_dot_first, x_dot_second, y_dot_second,
                              pix2deg_dict, analysis_parameter_dict, onset_column, offset_column, order_column,
                              participant_column, condition_color_dict, condition_line_dict, figure_path):
    data['mid_position_x'] = np.mean([data[x_dot_first], data[x_dot_second]], axis=0)
    data['mid_position_y'] = np.mean([data[y_dot_first], data[y_dot_second]], axis=0)

    data['distance_new'] = helper.compute_distance_pythagoras(data[x_touch], data[x_dot_second],
                                                              data[y_touch], data[y_dot_second])
    data['distance_new_dva'] = helper.scale_value_by_dict(data, 'distance_new', 'subject', pix2deg_dict)

    data['distance_old'] = helper.compute_distance_pythagoras(data[x_touch], data[x_dot_first],
                                                              data[y_touch], data[y_dot_first])
    data['distance_old_dva'] = helper.scale_value_by_dict(data, 'distance_old', 'subject', pix2deg_dict)

    data['distance_middle'] = helper.compute_distance_pythagoras(data[x_touch], data['mid_position_x'],
                                                                 data[y_touch], data['mid_position_y'])
    data['distance_middle_dva'] = helper.scale_value_by_dict(data, 'distance_middle', 'subject', pix2deg_dict)

    labels = ['distance_old_dva', 'distance_middle_dva', 'distance_new_dva']
    closest_col_name = 'closest_target'

    data = data.reset_index(drop=True)
    data = helper.add_min_label(data, labels, closest_col_name)

    participants = np.unique(data[participant_column])
    ref_scale = np.arange(analysis_parameter_dict['window_start'], analysis_parameter_dict['window_end'], 1)
    movement_rates = {}
    labels = np.unique(data[closest_col_name])

    for p in participants:
        movement_rates[p] = {}
        p_data = data[data[participant_column] == p].reset_index(drop=True)
        for idx, label in enumerate(labels):
            label_data = p_data[p_data[closest_col_name] == label]
            movement_rate_raw, movement_rate, scale = get_normalized_rates(label_data, ref_scale, onset_column,
                                                                           offset_column, order_column,
                                                                           analysis_parameter_dict, len(p_data) / 6)
            movement_rates[p][label] = movement_rate

    fig, axs = plt.subplots(1, 1)
    helper.get_average_rates(movement_rates,
                             scale,
                             labels,
                             {},
                             condition_color_dict,
                             condition_line_dict,
                             0.95, axs, [],
                             plot_average_participant_rates)

    axs.set_xlim([-500, 800])

    plt.savefig(figure_path)


def response_density_analysis(data, condition_dict,
                              window_center_name, touch_name,
                              position_name, origin_name,
                              dimensions, point_angle_deg,
                              column_names_to_align, participant_col,
                              parameters, x_value_col, ci, result_path,
                              cluster_table_path):
    center_name = 'centered'
    relative_name = 'relative'
    rotated_name = 'rotated'
    between_angle_name = 'angle_between_targets'
    transform_angle_name = 'transform_angle'
    scaled_touch_name = 'scaled_touch_distance'

    data = data.reset_index(drop=True)
    density_maps = {}
    for condition in condition_dict:
        condition_data, condition_index = filter_data(data, condition_dict[condition], return_index=True)

        # align x and y touch position to screen center
        for dim in dimensions:
            window_dim_name = f'{window_center_name}_{dim}'
            touch_dim_name = f'{touch_name}_{dim}'
            condition_data = helper.get_position_relative(condition_data, window_dim_name, touch_dim_name, center_name)

        # get angle in radians between the two dots
        condition_data[between_angle_name] = [(math.atan2(
            condition_data[f'{position_name}_y'].values[i] - condition_data[f'{origin_name}_y'].values[i],
            condition_data[f'{position_name}_x'].values[i] - condition_data[f'{origin_name}_x'].values[i])) for i in
            range(len(condition_data))]

        # compute how much we need to turn everything to align all dots to
        condition_data[transform_angle_name] = -1 * condition_data[between_angle_name] + math.radians(point_angle_deg)

        # get all positions relative to the origin
        for dim in dimensions:
            origin_col = f'{origin_name}_{dim}'
            column_dims_to_align = [f'{col}_{dim}' for col in column_names_to_align]
            for point_column in column_dims_to_align:
                condition_data = helper.get_position_relative(condition_data, origin_col, point_column, relative_name)

        # rotate the shifted dots
        rotate_column_names = [position_name, f'{center_name}_{touch_name}']
        for rotate_column in rotate_column_names:

            for idx in condition_data.index:
                origin = [condition_data[f'{relative_name}_{origin_name}_x'][idx],
                          condition_data[f'{relative_name}_{origin_name}_y'][idx]]
                point = [condition_data[f'{relative_name}_{rotate_column}_x'][idx],
                         condition_data[f'{relative_name}_{rotate_column}_y'][idx]]
                rad = condition_data[transform_angle_name][idx]
                rotx, roty = helper.rotate(origin, point, rad)
                condition_data.loc[idx, f'rotated_{rotate_column}_x'] = rotx
                condition_data.loc[idx, f'rotated_{rotate_column}_y'] = roty

        # scale x component of the touch by x component of the dots
        condition_data[scaled_touch_name] = condition_data[f'rotated_{center_name}_{touch_name}_x'] / condition_data[
            f'rotated_{position_name}_x']
        data.loc[condition_index, scaled_touch_name] = condition_data[scaled_touch_name].values

    # get the position density per participant
    y_value_col = scaled_touch_name
    for p in np.unique(data[participant_col]):
        p_data = data[data[participant_col] == p]
        density_maps[p] = {}

        for condition in condition_dict:
            condition_data = filter_data(p_data, condition_dict[condition])
            heatmap = helper.make_heatmap(condition_data, parameters, x_value_col, y_value_col)
            density_maps[p][condition] = heatmap
        keys = [x for x in condition_dict.keys()]
        density_maps[p]['diff'] = density_maps[p][keys[0]] - density_maps[p][keys[1]]

    fig, axs = plt.subplots(1, 3, figsize=(8.2, 2.5), sharex=True, sharey=True)
    cond_one = np.array([density_maps[p][keys[0]] for p in density_maps])
    cond_two = np.array([density_maps[p][keys[1]] for p in density_maps])
    cond_diff = np.array([density_maps[p]['diff'] for p in density_maps])

    # cluster-based permutation
    hm_clusters, hm_cutoff_value, hm_cluster_over_thresh = cmp.cluster_based_permutation_test(cond_one, cond_two,#3.579,
                                                                                              2.093, 1000, 0.05,
                                                                                              f'{result_path}/heatmap_tap.csv',
                                                                                              dimensions='2d')
    row_idx = 0
    significant_clusters_2d = pd.DataFrame(columns=['location', 'start_time', 'end_time', 'start_rotation',
                                                    'end_rotation', 'center_time', 'center_rotation', 'center_time_loc',
                                                    'center_rotation_loc', 'center_value', 'cluster_weight',
                                                    'weight_cutoff'])
    participants = [x for x in density_maps]
    hm_time = density_maps[participants[0]]['diff'].columns.values
    hm_rotation = density_maps[participants[0]]['diff'].index.values

    for cluster_id in hm_clusters:
        if len(hm_clusters[cluster_id]['cluster_location'][0]) > 1:
            if not all(np.isnan(hm_clusters[cluster_id]['cluster_values'])):
                #print(np.isnanhm_clusters[cluster_id]['cluster_values'] == np.nan)
                significant_clusters_2d.loc[row_idx, 'location'] = hm_clusters[cluster_id]['cluster_location']
                significant_clusters_2d.loc[row_idx, 'start_time'] = hm_time[min(
                    hm_clusters[cluster_id]['cluster_location'][1])]
                significant_clusters_2d.loc[row_idx, 'end_time'] = hm_time[max(
                    hm_clusters[cluster_id]['cluster_location'][1])]
                significant_clusters_2d.loc[row_idx, 'start_rotation'] = hm_rotation[min(
                    hm_clusters[cluster_id]['cluster_location'][0])]
                significant_clusters_2d.loc[row_idx, 'end_rotation'] = hm_rotation[max(
                    hm_clusters[cluster_id]['cluster_location'][0])]

                value_loc_df_time = pd.DataFrame(np.array([hm_clusters[cluster_id]['cluster_location'][1],
                                                           abs(hm_clusters[cluster_id]['cluster_values'])]).T,
                                                 columns=['location', 'values']).reset_index(drop=True)
                weighted_average_location_time = round(helper.get_weighted_average(value_loc_df_time,
                                                                                   'location',
                                                                                   'values'))
                significant_clusters_2d.loc[row_idx, 'center_time'] = hm_time[weighted_average_location_time]
                significant_clusters_2d.loc[row_idx, 'center_time_loc'] = weighted_average_location_time

                value_loc_df_latency = pd.DataFrame(np.array([hm_clusters[cluster_id]['cluster_location'][0],
                                                              abs(hm_clusters[cluster_id]['cluster_values'])]).T,
                                                    columns=['location', 'values']).reset_index(drop=True)
                weighted_average_location_latency = round(helper.get_weighted_average(value_loc_df_latency,
                                                                                      'location',
                                                                                      'values'))
                significant_clusters_2d.loc[row_idx, 'center_rotation'] = hm_rotation[
                    weighted_average_location_latency]
                significant_clusters_2d.loc[row_idx, 'center_rotation_loc'] = weighted_average_location_latency
                significant_clusters_2d.loc[row_idx, 'center_value'] = np.mean(cond_diff, axis=0)[
                    weighted_average_location_latency][weighted_average_location_time]
                significant_clusters_2d.loc[row_idx, 'cluster_weight'] = hm_clusters[cluster_id]['cluster_weight']
                significant_clusters_2d.loc[row_idx, 'weight_cutoff'] = hm_cutoff_value
                row_idx += 1
    cluster_df = significant_clusters_2d[abs(significant_clusters_2d.cluster_weight) > significant_clusters_2d.weight_cutoff]
    significant_clusters_2d.to_csv(cluster_table_path)

    f1 = sns.heatmap(data=pd.DataFrame(np.mean(cond_one, axis=0),
                                       columns=heatmap.columns,
                                       index=heatmap.index),
                     ax=axs[0], vmin=0, vmax=0.004, cbar=True, cmap='Greys')
    f2 = sns.heatmap(data=pd.DataFrame(np.mean(cond_two, axis=0),
                                       columns=heatmap.columns, index=heatmap.index),
                     ax=axs[1], vmin=0, vmax=0.004, cbar=True, cmap='Greys')
    f3 = sns.heatmap(data=pd.DataFrame(np.mean(cond_diff, axis=0),
                                       columns=heatmap.columns, index=heatmap.index),
                     ax=axs[2], vmin=-0.002, vmax=0.002, cbar=True, cmap='vlag')
    for idx in cluster_df.index:
        axs[2].scatter(cluster_df.loc[idx, 'center_time_loc'],
                       cluster_df.loc[idx, 'center_rotation_loc'],
                       marker='+', color='black')
    plt.tight_layout()
    for cluster in hm_cluster_over_thresh.cluster_location:
        for array in np.array(cluster).T:
            f3.add_patch(Rectangle((array[1], array[0]), 1, 1, fill=True, alpha=0.2, edgecolor='none'))

    fig.savefig(f'{result_path}/heatmap_response_densities.svg')
    return density_maps


def run_anovas(input_file, dependent_vars, independent_vars, group):
    data = pd.read_csv(input_file)
    for metric in dependent_vars:
        anova = AnovaRM(data=data,
                        depvar=metric,
                        subject=group,
                        within=independent_vars)
        fitted_anova = anova.fit()

        print(f'Results from ANOVA with '
              f'\nDEPENDENT variable: {metric.upper()}; '
              f'\nINDEPENDENT variables: {[x.upper() for x in independent_vars]}, '
              f'\nGROUPED by: {group.upper()}\n\n')
        print(fitted_anova)


def run_ttests(data, dependent_vars, independent_vars_dict):
    for dep_var in dependent_vars:
        for key in independent_vars_dict:
            assert (len(independent_vars_dict[key]) == 2)
            set_one = data[data[key] == independent_vars_dict[key][0]][dep_var].values
            set_two = data[data[key] == independent_vars_dict[key][1]][dep_var].values
            results = ttest_rel(set_one, set_two, nan_policy='omit')

            print(f'Results from paired T-Test with '
                  f'\nDEPENDENT variable: {dep_var} ; '
                  f'\nINDEPENDENT variables: {key} at levels {independent_vars_dict[key]}')
            print(results, '\n\n')
