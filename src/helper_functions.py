import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def threshold_setnan_entries_from_df(df, set_nan_cols, threshold_col, threshold_value):
    set_nan_df = df.copy()
    for col in set_nan_cols:
        set_nan_df.loc[set_nan_df[threshold_col] < threshold_value, col] = np.nan
    return set_nan_df


def smooth_columns_in_df(df, columns_to_smooth, smooth_window=3, smooth_polynomial=2):
    smooth_df = df.copy()
    for col in columns_to_smooth:
        values = smooth_df[col].values
        smooth_df.loc[:, col] = savgol_filter(values, smooth_window, smooth_polynomial)
    return smooth_df
