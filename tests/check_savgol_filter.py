import src.helper_functions as helper
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def show_data_filtered(filename, col_name, time_name, limit):
    df = pd.read_csv(filename)
    df_high_confidence = helper.threshold_setnan_entries_from_df(df, col_name, 'confidence', 0.9)
    x = df_high_confidence[col_name].values[limit]
    time = df_high_confidence[time_name].values[limit]

    plt.plot(time, x, color='green')
    x_filtered = savgol_filter(x, 3, 2)
    x_smoothed = smooth_positions(x, 5)
    plt.plot(time, x_filtered, color='limegreen')
    plt.plot(time[2:-2], x_smoothed, color='grey')

    plt.show()


def smooth_positions(pos, n_smooth):
    if n_smooth % 2 != 0:
        n_smooth = n_smooth-1
        print('non-even smoothing, will assume that the smoothing function is centered')
    new_series = []
    for entry in range(int(n_smooth/2), len(pos)-int(n_smooth/2)):
        new_series.append(np.mean(pos[entry-int(n_smooth/2):entry+int(n_smooth/2)]))
    return new_series


def threshold_setnan_entries_from_df(df, set_nan_cols, threshold_col, threshold_value):
    set_nan_df = df.copy()
    for col in set_nan_cols:
        set_nan_df.loc[set_nan_cols[threshold_col] < threshold_value, col] = np.nan
    return set_nan_df


if __name__ == '__main__':
    show_data_filtered('../data/test_data/test_gaze_data.csv',
                       'y_norm',
                       'world_timestamp',
                       np.arange(1000, 3000))
