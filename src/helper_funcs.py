import math
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import curve_fit


def save_dict_as_table(dictionary, out_file, key_name):
    data = pd.DataFrame()
    for key in dictionary:
        df = pd.DataFrame.from_dict(dictionary[key], orient='index')
        df[key_name] = key
        data = pd.concat([data, df], axis=0)
    data = data.reset_index(drop=False, names='condition')
    data.to_csv(out_file, index=False)
    return data


def scale_value_by_dict(data, scale_value, scale_by, scale_dict):
    scaled_values = data[scale_value].copy(deep = True)
    for key in scale_dict:
        scaled_values[data[scale_by] == key] /= scale_dict[key]
    return scaled_values


def set_values_relative(data, set_value_col, relative_to_cols, key):
    aligned_value = data[set_value_col]
    for col in relative_to_cols:
        aligned_value -= data[col]
    data[f'{set_value_col}{key}'] = aligned_value
    return data


def smooth_array(array, time_array, smooth, time):
    start = time - smooth
    end = time + smooth
    idx_1 = np.where(time_array >= start)
    idx_2 = np.where(time_array < end)
    mean_value = np.mean(array[np.intersect1d(idx_1, idx_2)])

    return mean_value


def sigmoid(x, L, x0, k, b):
    y = L/(1 + np.exp(-k*(x-x0))) + b
    return y


def fit_sigmoid_func(x_data, y_data):
    # initial guess
    try:
        p0 = [max(y_data), np.median(x_data), 0.001, min(y_data)]
        popt, pcov = curve_fit(sigmoid, x_data, y_data, p0, method='lm')
    except ValueError:
        finite_lim = np.where([not np.isfinite(y) for y in y_data])[0].min()
        y_data = y_data[:finite_lim]
        x_data = x_data[:finite_lim]

        p0 = [max(y_data), np.median(x_data), 0.001, min(y_data)]
        popt, pcov = curve_fit(sigmoid, x_data, y_data, p0, method='lm')

    return popt, pcov


def compute_distance_pythagoras(x1, x2, y1, y2):
    x_val = x1 - x2
    y_val = y1 - y2

    return pythagoras(x_val, y_val)


def pythagoras(x_value, y_value):
    """
    takes two values and computes the distance between them
    """
    return np.sqrt(x_value**2 + y_value**2)


def get_average_rates(rates, scale, conditions, parameters, condition_color_dict, linestyle_dict, ci, axs, cluster, plotting_func):
    rates_dict = {}
    ci_dict = {}
    for cond in conditions:
        ci_dict[cond] = {}
        condition_rates = [rates[p][cond] for p in rates]
        average_rate = np.mean(condition_rates, axis=0)
        rates_dict[cond] = average_rate
        ci_upper, ci_lower = st.t.interval(alpha=ci, df=len(rates) - 1,
                                           loc=average_rate,
                                           scale=st.sem(condition_rates))
        ci_dict[cond]['ci_upper'] = ci_upper
        ci_dict[cond]['ci_lower'] = ci_lower

    plotting_func(rates_dict, ci_dict, scale, parameters, cluster, condition_color_dict, linestyle_dict, axs)
    # return rates_dict, scale


def add_min_label(data, labels, new_col_name):
    closest = []
    for idx in data.index:
        position_min = np.argmin(data.loc[idx, labels])
        closest.append(labels[position_min])
    data[new_col_name] = closest
    return data

def rotate(origin, point, angle):
    # angle in radians
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def get_position_relative(data, origin, point, name_appendix):
    data[f'{name_appendix}_{point}'] = data[point] - data[origin]
    return data

def scramble_columns_piecewise(data, scramble_column, piecewise, name_scrambled, session_name):
    data = data.reset_index(drop = True)
    for s in np.unique(data[session_name]):
        s_data = data[data[session_name] == s]
        for piece in np.unique(data[piecewise]):
            values_to_scramble = s_data[s_data[piecewise] == piece][scramble_column].values
            idx = s_data[s_data[piecewise] == piece].index
            np.random.shuffle(values_to_scramble)
            data.loc[idx, name_scrambled] = values_to_scramble

    return data

def make_heatmap(data, parameters, x_value_col, y_value_col):
    heatmap_df = pd.DataFrame()
    total_x = abs(parameters['x_min']) + abs(parameters['x_max'])
    window_width_x = total_x / parameters['n_col']

    total_y = abs(parameters['y_min']) + abs(parameters['y_max'])
    window_width_y = total_y / parameters['n_row']

    for n_c in np.arange(parameters['x_min'], parameters['x_max'], window_width_x): #range(parameters['n_col']):
        for n_r in np.arange(parameters['y_min'], parameters['y_max'], window_width_y): #range(parameters['n_row']):
            x_val = n_c #parameters['x_min'] + n_c * window_width_x
            y_val = n_r #parameters['y_min'] + n_r * window_width_y

            x_filtered = data[data[x_value_col].between(x_val, x_val + window_width_x)]
            y_filtered = x_filtered[x_filtered[y_value_col].between(y_val, y_val + window_width_y)]
            try:
                heatmap_df.loc[round(y_val, 5), round(x_val)] = len(y_filtered) / len(data)
            except ZeroDivisionError:
                heatmap_df.loc[y_val, x_val] = 0
    #heatmap_df.columns =
    #heatmap_df.index =

    return heatmap_df

def get_dataframe_per_condition(data, conditions):
    dataframes = []
    for condition in conditions:
        dataframes.append(pd.DataFrame([data[x][condition] for x in data]))
    return dataframes
