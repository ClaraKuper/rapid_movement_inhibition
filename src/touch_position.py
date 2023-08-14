import numpy as np
import src.helper_funcs as hf
from src.json_parsing import filter_data


def get_fitted_responses(data, x_full_length, y_full_length, x_col, y_col, target_x_col, target_y_col,
                         pix2deg_by_name, pix2deg_dictionary, participant_id_col, condition_dictionary,
                         time_col_name, params, sigmoid_func):

    #data = data.dropna(axis=1)
    #data = data.reset_index(drop=True)

    # get screen center
    x_center_name = 'x_center'
    y_center_name = 'y_center'
    data[x_center_name] = data[x_full_length]/2
    data[y_center_name] = data[y_full_length]/2

    # align touches
    relative_key = '_relative'
    data = hf.set_values_relative(data, x_col, [x_center_name], relative_key)
    data = hf.set_values_relative(data, y_col, [y_center_name], relative_key)

    # explicitly set the type of the columns to enable later operations
    data[f'{x_col}{relative_key}'] = data[f'{x_col}{relative_key}'].astype(float)
    data[f'{y_col}{relative_key}'] = data[f'{y_col}{relative_key}'].astype(float)

    data[target_x_col] = data[target_x_col].astype(float)
    data[target_y_col] = data[target_y_col].astype(float)

    # get distance between touch and target
    distance_name = 'distance'
    data[distance_name] = hf.compute_distance_pythagoras(data[f'{x_col}{relative_key}'], data[target_x_col],
                                                         data[f'{y_col}{relative_key}'], data[target_y_col])
    data[distance_name] = data[distance_name].astype(float)
    # get relative distance
    distance_dva_name = f'{distance_name}_dva'
    data[distance_dva_name] = hf.scale_value_by_dict(data, distance_name, pix2deg_by_name, pix2deg_dictionary)
    data[distance_dva_name] = data[distance_dva_name].astype(float)
    # get smoothed response positions
    smoothed_response_positions, scale = get_smoothed_responses(data, participant_id_col, condition_dictionary,
                                                                distance_dva_name, time_col_name, params)

    # fit by person and condition:
    position_response_dictionary = fit_by_id_and_condition(data,
                                                           participant_id_col,
                                                           condition_dictionary,
                                                           sigmoid_func,
                                                           time_col_name,
                                                           distance_dva_name)

    return smoothed_response_positions, position_response_dictionary, scale


def fit_by_id_and_condition(data, id_col_name, condition_dictionary, fit_function, x_name, y_name):
    dictionary = {}
    for p in np.unique(data[id_col_name]):
        dictionary[p] = {}
        p_data = data[data[id_col_name] == p]
        for condition in condition_dictionary:
            dictionary[p][condition] = {}
            cond_data = filter_data(p_data, condition_dictionary[condition])
            popt, pcov = fit_function(cond_data[x_name], cond_data[y_name])
            dictionary[p][condition]['L'] = popt[0]
            dictionary[p][condition]['x0'] = popt[1]
            dictionary[p][condition]['k'] = popt[2]
            dictionary[p][condition]['b'] = popt[3]
            dictionary[p][condition]['flash_shown'] = condition_dictionary[condition]['flashShown']
            dictionary[p][condition]['stim_jumped'] = condition_dictionary[condition]['stimJumped']
    return dictionary


def get_smoothed_responses(data, id_col_name, condition_dictionary, distance_col_name, time_col_name, params):
    dictionary = {}
    scale = np.arange(params['start_smooth'], params['end_smooth'])
    for p in np.unique(data[id_col_name]):
        dictionary[p] = {}
        p_data = data[data[id_col_name] == p]
        for condition in condition_dictionary:
            cond_data = filter_data(p_data, condition_dictionary[condition])
            # continue here!
            smoothed_positions = [hf.smooth_array(cond_data[distance_col_name], cond_data[time_col_name], params['smooth'], time) for time in scale]
            dictionary[p][condition] = smoothed_positions
    return dictionary, scale
