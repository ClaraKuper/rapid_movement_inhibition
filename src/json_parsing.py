import src.helper_funcs as hf
import numpy as np
import os
import pandas as pd


def format_and_save_data_multiple_participants(input_path, output_path, filter_dict,
                                               length_dict, explode_lists, set_time_cols,
                                               relative_to_cols, remove_false_cols, outlier_dict,
                                               format_dict, incomplete_session_paths,
                                               extra_session_path = '',
                                               force_save_new = False):
    if os.path.isfile(output_path) and not force_save_new:
        data = pd.read_csv(output_path)
    else:
        # load the main session
        data = save_json_as_csv(input_path, output_path)

        # load incomplete data
        for session_file in incomplete_session_paths:
            extra_data = load_json(session_file)
            data = pd.concat([data, extra_data])
            data = data.reset_index(drop=True)
            # print(f'participant:{np.unique(extra_data.prolific_id)}\n '
            #      f'session: {np.unique(extra_data.session_number)}\n '
            #      f'session code: {session}\n ')
        # load extra data
        if not extra_session_path == '':
            extra_data = load_json(extra_session_path)
            data = pd.concat([data, extra_data])
            data = data.reset_index(drop=True)

        data = filter_data(data, filter_dict)
        data = adjust_format(data, length_dict=length_dict, expand_lists=explode_lists)
        relative_key = '_relative'
        for c in set_time_cols:
            data = hf.set_values_relative(data, c, relative_to_cols, relative_key)
        data = remove_false(data, remove_false_cols)
        data = remove_outlier(data, outlier_dict)
        data = set_data_type(data, format_dict)
        data.to_csv(output_path, index=False)
    return data


def parse_screen_information(input_path, output_path, filter_dict, outlier_dict, session_column, pixel_column):
    if os.path.isfile(output_path):
        data = pd.read_csv(output_path)
    else:
        data = save_json_as_csv(input_path, output_path)
        data = filter_data(data, filter_dict)
        data = remove_outlier(data, outlier_dict)
        data.to_csv(output_path, index=False)

    px2deg_dict = parse_values_to_dictionary(data, session_column, pixel_column)
    return px2deg_dict


def parse_values_to_dictionary(data, get_by, get_value):
    dictionary = {}
    for idx in data.index:
        dictionary[data.loc[idx, get_by]] = data.loc[idx, get_value]
    return dictionary


def save_json_as_csv(input_path, output_path):
    data = load_json(input_path)
    data.to_csv(output_path, index=False)
    return data


def load_json(file):
    data = pd.DataFrame()
    with open(file) as f:
        # unpack the data
        for jf in f:
            # save it to a data frame
            df = pd.read_json(jf)
            data = pd.concat([data, df], axis=0)
    return data


def filter_data(data, dictionary, return_index = False):
    for key in dictionary:
        data = data[data[key] == dictionary[key]]
    data = data.dropna(axis=1, how='all')
    index = data.index
    data = data.reset_index(drop=True)
    if return_index:
        return data, index
    else:
        return data


def adjust_format(data, length_dict, expand_lists):
    data = match_entry_length(data, length_dict)

    for expand_list in expand_lists:
        data = data.explode(expand_list)

    return data


def match_entry_length(data, dictionary):
    for key in dictionary:
        for idx in data.index:
            try:
                current_length = len(data.loc[idx, key])
            except TypeError:
                data.loc[idx, key] = [data.loc[idx, key]]
                current_length = len(data.loc[idx, key])
            desired_length = dictionary[key]
            if current_length < desired_length:
                nan_pad = [np.nan] * (desired_length - current_length)
                data.loc[[idx], key] = pd.Series([np.concatenate([data.loc[idx, key], nan_pad])],
                                                 index=data.index[[idx]])
            elif current_length > desired_length:
                data.loc[[idx], key] = pd.Series([data.loc[idx, key][:desired_length]], index=data.index[[idx]])
    return data


def remove_false(data, remove_false_cols):
    for c in remove_false_cols:
        data = data[data[c] == True]
    return data


def remove_outlier(data, dictionary):
    for key in dictionary:
        for entry in dictionary[key]:
            data = data[data[key] != entry]
    return data


def test_file_equality(new_file, test_file):
    new = pd.read_csv(new_file)
    test = pd.read_csv(test_file)

    if not new.shape == test.shape:
        raise ValueError(f'The new file does not have the right format. Expected shape {test.shape}, '
                         f'got {new.shape} instead. \nHave there been any changes to the input data or to the script?')

    col_missmatch = []
    for col in test.columns:
        if not all(new[col].dropna().values == test[col].dropna().values):
            col_missmatch.append(col)

    if len(col_missmatch) > 0:
        raise ValueError(f'columns {col_missmatch} have changed')

    print('Test passed without errors.')


def set_data_type(data, dictionary):
    for key in dictionary:
        data[key] = data[key].astype(dictionary[key])
    return data

def add_participant_id(data, dictionary, col_name, condition_letter, id_col_name, ses_col_name, participant_letter = 'P'):
    for s in dictionary:
        p_number = dictionary[s].upper().split(condition_letter)[0]
        try:
            s_number = dictionary[s].upper().split(condition_letter)[1]
        except IndexError:
            s_number = 0

        try:
            final_number = p_number.split(participant_letter)[1]
        except IndexError:
            final_number = p_number.split(participant_letter)[0]

        data.loc[data[col_name] == s, id_col_name] = int(final_number)
        data.loc[data[col_name] == s, ses_col_name] = int(s_number)
    return data
