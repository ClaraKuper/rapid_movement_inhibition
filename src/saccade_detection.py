import src.helper_functions as helper
import numpy as np
import os
import pandas as pd


def write_saccades_to_file(raw_data_name, save_as_name):
    if os.path.isfile(save_as_name):
        raise FileExistsError(f'File {save_as_name} exists.')

    saccade_df = pd.DataFrame(columns=['onset_eyetracker', 'offset_eyetracker', 'amplitude', 'duration', 'peak_velocity'])
    df = pd.read_csv(raw_data_name)
    amplitudes, peak_velocities, saccade_durations, onsets, offsets = detect_saccades(data=df,
                                                                                      x_name='x_norm',
                                                                                      y_name='y_norm',
                                                                                      time_name='gaze_timestamp',
                                                                                      confidence_name='confidence',
                                                                                      confidence_threshold=0.9)
    saccade_df['onset_eyetracker'] = onsets
    saccade_df['offset_eyetracker'] = offsets
    saccade_df['amplitude'] = amplitudes
    saccade_df['duration'] = saccade_durations
    saccade_df['peak_velocity'] = peak_velocities

    saccade_df.to_csv(save_as_name)


def detect_saccades(data, x_name, y_name, time_name, confidence_name, confidence_threshold,
                    velocity_threshold_factor=3):
    df_high_confidence = helper.threshold_setnan_entries_from_df(data,
                                                                 [x_name, y_name],
                                                                 confidence_name,
                                                                 confidence_threshold)
    df_high_confidence_smoothed = helper.smooth_columns_in_df(df_high_confidence, [x_name, y_name])
    x_pos = df_high_confidence_smoothed[x_name]
    y_pos = df_high_confidence_smoothed[y_name]
    time = df_high_confidence_smoothed[time_name]
    frequency = 1 / np.median(np.diff(time))
    return engbert_mergenthaler(x_pos, y_pos, time, velocity_threshold_factor, frequency)


def engbert_mergenthaler(x, y, time, thresh_factor, freq):
    """
    saccade detection algorithm after engbert and mergenthaler/ engbert and kliegl:
    https://www.sciencedirect.com/science/article/pii/S0042698903000841

    the velocity of saccades is being compared against a median std of the average saccade
    """
    # initialize saccade parameters to detect
    amplitudes = []
    onsets = []
    offsets = []
    pvels = []
    saccdurs = []

    # combine x and y velocities
    # get distance travelled
    dist = np.sqrt(np.array(x) ** 2 + np.array(y) ** 2)
    dist_vel = np.diff(dist) * freq
    # find acceleration
    dist_acc = np.diff(dist_vel)

    # compute threshold and apply
    adaptive_threshold = np.nanmedian(abs(dist_vel)) + thresh_factor * np.nanstd(abs(dist_vel))
    dist_thresh = np.where(abs(dist_vel) > adaptive_threshold)[0]

    # split into individual events
    split_dist_index = np.split(dist_thresh, np.where(np.diff(dist_thresh) > 4)[0] + 1)
    for s_idx in split_dist_index:
        # check direction of saccade - forward or backward:
        start_sign, end_sign = get_direction(dist_vel[s_idx])
        start_position = get_saccade_onset(dist_acc, s_idx, start_sign)
        end_position = get_saccade_offset(dist_acc, s_idx, end_sign)

        vel_samples = dist_vel[start_position:end_position]
        time_samples = time[start_position:end_position]

        # filter the x and y position that belong to the saccade
        dist_x_saccade = np.array(x)[start_position:end_position]
        dist_y_saccade = np.array(y)[start_position:end_position]

        try:
            onsets.append(time_samples.iloc[0])
            offsets.append(time_samples.iloc[-1])

            # get parameters
            amplitudes.append(get_amplitude(dist_x_saccade, dist_y_saccade))
            pvels.append(np.max(abs(vel_samples)))
            saccdurs.append(time_samples.iloc[-1] - time_samples.iloc[0])
        except IndexError as e:
            pass
    return amplitudes, pvels, saccdurs, onsets, offsets


def get_direction(velocity):
    if np.sign(np.mean(velocity)) == np.sign(1):
        start_sign = 1
    else:
        start_sign = -1
    end_sign = start_sign * -1
    return start_sign, end_sign


def get_amplitude(dist_x, dist_y):
    return np.sqrt((dist_x[0] - dist_x[-1]) ** 2 + (dist_y[0] - dist_y[-1]) ** 2)


def get_saccade_onset(acceleration, index, start_sign):
    """
    # look for the start of the acceleration (deceleration)
    # detect beginning of the saccade - go backwards till the sign flips
    :param acceleration:
    :param index:
    :param start_sign:
    :return:
    """
    start_search = max(index[0] - 1, 0)
    while np.sign(start_sign) == np.sign(acceleration[start_search]):
        start_search -= 1
        if start_search == 0:
            break
    return start_search


def get_saccade_offset(acceleration, index, sign):
    """
    # look for the end of the acceleration (deceleration)
    # detect end of the saccade - go forwards till the sign flips
    :param acceleration:
    :param index:
    :param sign:
    :return:
    """

    end_search = min(index[-1] + 1, len(acceleration) - 1)
    while np.sign(sign) == np.sign(acceleration[end_search]):
        end_search += 1
        if end_search == len(acceleration):
            break
    return end_search + 1
