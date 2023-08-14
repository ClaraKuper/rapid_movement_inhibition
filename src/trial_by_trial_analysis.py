import numpy as np


def set_timings(data, touch_on_col, touch_off_col):
    data['touchOff_shifted'] = np.concatenate([[0], data[touch_off_col].iloc[0:-1].values])
    data['rest_times'] = data[touch_off_col] - data[touch_on_col]
    data['flight_times'] = data[touch_on_col] - data['touchOff_shifted']
    data['cycle_times'] = data['rest_times'] + data['flight_times']
    return data
