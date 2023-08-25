import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import src.helper_funcs as helper
from src.movement_rates import get_movement_rates_by_participant, get_normalized_rates
from src.trial_by_trial_analysis import set_timings
from src.touch_position import get_fitted_responses
from src.json_parsing import set_data_type, filter_data
from src.plotting import make_figure_rates, plot_metrics,  plot_average_participant_rates, \
    plot_average_participant_position, make_delay_figure, make_latency_heatmaps
from statsmodels.stats.anova import AnovaRM
from scipy.stats import ttest_rel, t, sem



def analysis_rates(data, onset_column, offset_column, participant_column, analysis_parameter_dict, order_column,
                   conditions_dict, condition_color_dict, metrics_out_file, metrics_figure_file, dependent_vars,
                   independent_vars, figure_height=6):
    '''
    Movement Rate Analysis
    - Step 1: Compute Rates for Individual Participants
    - Step 2: Get metrics from the rates
    - Step 3: compare metrics statistically
    - Step 4: Plot Rates
    - Step 5: Plot metrics
    '''

    # make figure
    axs = make_figure_rates(figure_height, dependent_vars)
    rates, rate_metrics, scale = get_movement_rates_by_participant(data,
                                                                   onset_column,
                                                                   offset_column,
                                                                   participant_column,
                                                                   analysis_parameter_dict,
                                                                   order_column,
                                                                   conditions_dict,
                                                                   condition_color_dict)
    metrics = helper.save_dict_as_table(rate_metrics, metrics_out_file, participant_column)
    helper.get_average_rates(rates,
                             scale,
                             conditions_dict,
                             rate_metrics,
                             condition_color_dict,
                             0.95,
                             axs['main'],
                             plot_average_participant_rates)

    plot_metrics(metrics, dependent_vars, axs, condition_color_dict)
    plt.savefig(metrics_figure_file)

    run_anovas(dependent_vars, independent_vars, metrics, participant_column)


def analysis_position(data, x_col, y_col, target_x_col, target_y_col, x_full_length, y_full_length, pix2deg_by_name,
                      pix2deg_dictionary, participant_col, condition_dictionary, condition_color_dict, time_col_name,
                      params, dependent_vars, independent_vars_dict, metrics_out_file, data_type_dict, figure_height=6):
    axs = make_figure_rates(figure_height, dependent_vars)
    data = set_data_type(data, data_type_dict)
    smoothed_response_positions, position_response_dictionary, scale, data = get_fitted_responses(data, x_full_length,
                                                                                            y_full_length, x_col, y_col,
                                                                                            target_x_col, target_y_col,
                                                                                            pix2deg_by_name,
                                                                                            pix2deg_dictionary,
                                                                                            participant_col,
                                                                                            condition_dictionary,
                                                                                            time_col_name, params,
                                                                                            helper.fit_sigmoid_func)

    metrics = helper.save_dict_as_table(position_response_dictionary, metrics_out_file, participant_col)
    helper.get_average_rates(smoothed_response_positions, scale, condition_dictionary, position_response_dictionary,
                             condition_color_dict, 0.95, axs['main'], plot_average_participant_position)
    plot_metrics(metrics, dependent_vars, axs, condition_color_dict)
    run_ttests(metrics, dependent_vars, independent_vars_dict)

    return data


def trial_by_trial_analysis(data, time_column, plot_column_dict, condition_dict, baseline_condition_dict, color_dict,
                            participant_col, touch_on_col, touch_off_col, smooth_window_size, figure_name,
                            heatmap_parameters, heatmap_figure_path, heatmap_individual_figure_path):
    participants = np.unique(data[participant_col])
    dictionary = {}
    heatmap_dictionary = {}
    test_data = set_timings(data, touch_on_col, touch_off_col)
    test_data = test_data[test_data.choiceOrder != 0]
    test_data = test_data.reset_index(drop=True)
    time = np.arange(min(test_data[time_column]), max(test_data[time_column]))
    all_base_heatmaps = []
    baseline_data = test_data.copy(deep=True)
    for base_feat in baseline_condition_dict:
        baseline_data = baseline_data[baseline_data[base_feat] == baseline_condition_dict[base_feat]]
        baseline_data = baseline_data.reset_index(drop=True)

    for p in participants:
        p_data = test_data[test_data[participant_col] == p]
        p_base_data = baseline_data[baseline_data[participant_col] == p].reset_index(drop=True)
        p_base_heatmap = helper.make_heatmap(p_base_data, heatmap_parameters, time_column, 'flight_times')
        dictionary[p] = {}
        heatmap_dictionary[p] = {}
        heatmap_dictionary[p]['flash- jump-'] = p_base_heatmap

        for cond in condition_dict:
            dictionary[p][cond] = {}
            feature_dict = condition_dict[cond]
            feat_data = p_data

            for feat in feature_dict:
                feat_data = feat_data[feat_data[feat] == feature_dict[feat]]
                feat_data = feat_data.reset_index(drop=True)

            for col in plot_column_dict:
                val = [helper.smooth_array(feat_data[plot_column_dict[col]], feat_data[time_column], smooth_window_size, t) for
                       t in time]
                base_val = [
                    helper.smooth_array(p_base_data[plot_column_dict[col]], p_base_data[time_column], smooth_window_size, t)
                    for t in time]
                dictionary[p][cond][col] = val
                dictionary[p][cond][f'{col}_diff'] = np.array(val) - np.array(base_val)

            heatmap = helper.make_heatmap(feat_data, heatmap_parameters, time_column, 'flight_times')
            heatmap_dictionary[p][cond] = heatmap

    make_delay_figure(dictionary, test_data, condition_dict, plot_column_dict.keys(), time, color_dict, figure_name,
                      heatmap_dictionary, heatmap_figure_path, participant_col)

    make_latency_heatmaps(heatmap_dictionary, condition_dict.keys(), heatmap_individual_figure_path)

    return time, dictionary


def landing_position_analysis(data, x_touch, y_touch, x_dot_first, y_dot_first, x_dot_second, y_dot_second, pix2deg_dict,
                              analysis_parameter_dict, onset_column, offset_column, order_column, participant_column,
                              condition_color_dict, figure_path):

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
                             0.95, axs,
                             plot_average_participant_rates)

    axs.set_xlim([-500, 800])

    plt.savefig(figure_path)


def response_density_analysis(data, condition_dict,
                              window_center_name, touch_name,
                              position_name, origin_name,
                              dimensions, point_angle_deg,
                              column_names_to_align, participant_col,
                              parameters, x_value_col, ci):

    center_name = 'centered'
    relative_name = 'relative'
    rotated_name = 'rotated'
    between_angle_name = 'angle_between_targets'
    transform_angle_name = 'transform_angle'
    scaled_touch_name = 'scaled_touch_distance'

    data = data.reset_index(drop = True)
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

    fig, axs = plt.subplots(1, 4, figsize = (25, 5), sharex=True, sharey=True)
    cond_one = np.array([density_maps[p][keys[0]] for p in density_maps])
    cond_two = np.array([density_maps[p][keys[1]] for p in density_maps])
    cond_diff = np.array([density_maps[p]['diff'] for p in density_maps])

    # confidence bounds
    lower, upper = t.interval(confidence=ci, df=len(cond_diff) - 1, loc=np.mean(cond_diff, axis=0),
                                 scale=sem(cond_diff, axis=0))
    # mask
    mask = pd.DataFrame(np.sign(lower) + np.sign(upper),
                        columns=heatmap.columns, index=heatmap.index)

    f1 = sns.heatmap(data = pd.DataFrame(np.mean(cond_one, axis = 0),
                                         columns=heatmap.columns, index=heatmap.index),
                     ax=axs[0], vmin = 0, vmax = 0.01)
    f2 = sns.heatmap(data = pd.DataFrame(np.mean(cond_two, axis = 0),
                                         columns=heatmap.columns, index=heatmap.index),
                     ax=axs[1], vmin = 0, vmax = 0.01)
    f3 = sns.heatmap(data = pd.DataFrame(np.mean(cond_diff, axis = 0),
                                         columns=heatmap.columns, index=heatmap.index),
                     ax=axs[2], vmin = -0.005, vmax = 0.005)

    f4 = sns.heatmap(data = mask, ax=axs[3], vmin = -2, vmax = 2)

    return density_maps


def run_anovas(dependent_vars, independent_vars, data, group):
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
    # save anovas to csv?


def run_ttests(data, dependent_vars, independent_vars_dict):
    for dep_var in dependent_vars:
        for key in independent_vars_dict:
            assert(len(independent_vars_dict[key]) == 2)
            set_one = data[data[key] == independent_vars_dict[key][0]][dep_var].values
            set_two = data[data[key] == independent_vars_dict[key][1]][dep_var].values
            results = ttest_rel(set_one, set_two, nan_policy='omit')

            print(f'Results from paired T-Test with '
                  f'\nDEPENDENT variable: {dep_var} ; '
                  f'\nINDEPENDENT variables: {key} at levels {independent_vars_dict[key]}')
            print(results, '\n\n')

