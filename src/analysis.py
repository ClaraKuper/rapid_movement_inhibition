import matplotlib.pyplot as plt
from src.helper_funcs import save_dict_as_table, fit_sigmoid_func, get_average_rates, smooth_array, \
    compute_distance_pythagoras, scale_value_by_dict, add_min_label
from src.movement_rates import get_movement_rates_by_participant, get_normalized_rates
from src.trial_by_trial_analysis import set_timings
from src.touch_position import get_fitted_responses
from src.json_parsing import set_data_type
from src.plotting import make_figure_rates, plot_metrics,  plot_average_participant_rates, \
    plot_average_participant_position, make_delay_figure
from statsmodels.stats.anova import AnovaRM
from scipy.stats import ttest_rel
import numpy as np


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
    metrics = save_dict_as_table(rate_metrics, metrics_out_file, participant_column)
    get_average_rates(rates,
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
                                                                                            fit_sigmoid_func)

    metrics = save_dict_as_table(position_response_dictionary, metrics_out_file, participant_col)
    get_average_rates(smoothed_response_positions, scale, condition_dictionary, position_response_dictionary,
                      condition_color_dict, 0.95, axs['main'], plot_average_participant_position)
    plot_metrics(metrics, dependent_vars, axs, condition_color_dict)
    run_ttests(metrics, dependent_vars, independent_vars_dict)

    return data


def trial_by_trial_analysis(data, time_column, plot_column_dict, condition_dict, baseline_condition_dict, color_dict,
                            participant_col, touch_on_col, touch_off_col, smooth_window_size, figure_name):
    participants = np.unique(data[participant_col])
    dictionary = {}
    test_data = set_timings(data, touch_on_col, touch_off_col)
    test_data = test_data[test_data.choiceOrder != 0]
    test_data = test_data.reset_index(drop=True)
    time = np.arange(min(test_data[time_column]), max(test_data[time_column]))

    baseline_data = test_data.copy(deep=True)
    for base_feat in baseline_condition_dict:
        baseline_data = baseline_data[baseline_data[base_feat] == baseline_condition_dict[base_feat]]
        baseline_data = baseline_data.reset_index(drop=True)

    for p in participants:
        p_data = test_data[test_data[participant_col] == p]
        p_base_data = baseline_data[baseline_data[participant_col] == p].reset_index(drop=True)
        dictionary[p] = {}

        for cond in condition_dict:
            dictionary[p][cond] = {}
            feature_dict = condition_dict[cond]
            feat_data = p_data

            for feat in feature_dict:
                feat_data = feat_data[feat_data[feat] == feature_dict[feat]]
                feat_data = feat_data.reset_index(drop=True)

            for col in plot_column_dict:
                val = [smooth_array(feat_data[plot_column_dict[col]], feat_data[time_column], smooth_window_size, t) for
                       t in time]
                base_val = [
                    smooth_array(p_base_data[plot_column_dict[col]], p_base_data[time_column], smooth_window_size, t)
                    for t in time]
                dictionary[p][cond][col] = val
                dictionary[p][cond][f'{col}_diff'] = np.array(val) - np.array(base_val)

    make_delay_figure(dictionary, test_data, condition_dict, plot_column_dict.keys(), time, color_dict, figure_name)
    return time, dictionary


def landing_position_analysis(data, x_touch, y_touch, x_dot_first, y_dot_first, x_dot_second, y_dot_second, pix2deg_dict,
                              analysis_parameter_dict, onset_column, offset_column, order_column, participant_column,
                              condition_color_dict, figure_path):

    data['mid_position_x'] = np.mean([data[x_dot_first], data[x_dot_second]], axis=0)
    data['mid_position_y'] = np.mean([data[y_dot_first], data[y_dot_second]], axis=0)

    data['distance_new'] = compute_distance_pythagoras(data[x_touch], data[x_dot_second],
                                                       data[y_touch], data[y_dot_second])
    data['distance_new_dva'] = scale_value_by_dict(data, 'distance_new', 'subject', pix2deg_dict)

    data['distance_old'] = compute_distance_pythagoras(data[x_touch], data[x_dot_first],
                                                       data[y_touch], data[y_dot_first])
    data['distance_old_dva'] = scale_value_by_dict(data, 'distance_old', 'subject', pix2deg_dict)

    data['distance_middle'] = compute_distance_pythagoras(data[x_touch], data['mid_position_x'],
                                                          data[y_touch], data['mid_position_y'])
    data['distance_middle_dva'] = scale_value_by_dict(data, 'distance_middle', 'subject', pix2deg_dict)

    labels = ['distance_old_dva', 'distance_middle_dva', 'distance_new_dva']
    closest_col_name = 'closest_target'

    data = data.reset_index(drop=True)
    data = add_min_label(data, labels, closest_col_name)

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

    get_average_rates(movement_rates,
                      scale,
                      labels,
                      {},
                      condition_color_dict,
                      0.95, axs,
                      plot_average_participant_rates)

    axs.set_xlim([-500, 800])

    plt.savefig(figure_path)


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

