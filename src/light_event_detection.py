import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.helper_funcs import add_block_structure


def detect_change_signal(time_series, light_series, threshold=10):
    n_samples = len(time_series)
    df = pd.DataFrame(columns=['start_time', 'end_time', 'start_index', 'end_index', 'duration', 'mean_value'])
    index = 0
    start_time = 0
    end_time = 0
    assert n_samples == len(light_series)
    light_on = 0
    idx = 0
    while idx < n_samples:
        if light_series[idx] > threshold and not light_on:
            find_id = idx
            while np.diff(light_series)[find_id] > 0 & find_id > 0:
                find_id -= 1
            start_idx = find_id + 1
            start_time = time_series[start_idx]
            light_on = 1

        elif light_series[idx] < threshold and light_on:
            find_id = idx
            while np.diff(light_series)[find_id] < 0:
                find_id += 1
            idx = find_id
            end_idx = find_id + 1
            end_time = time_series[end_idx]
            df.loc[index, 'start_time'] = start_time
            df.loc[index, 'end_time'] = end_time
            df.loc[index, 'start_index'] = start_idx
            df.loc[index, 'end_index'] = end_idx
            df.loc[index, 'duration'] = end_time - start_time
            df.loc[index, 'mean_value'] = np.mean(light_series[start_idx:end_idx])
            index += 1

            light_on = 0
        idx += 1

    return df


def get_block_onsets(time_series, duration_series, block_list, n_blocks=10, threshold=60):
    df = pd.DataFrame(columns=['block_number', 'onset'])
    time_series = time_series[np.where((0.8 < duration_series) & (duration_series < 1.2))[0]]
    offset_series = time_series + duration_series[np.where((0.8 < duration_series) & (duration_series < 1.2))[0]]
    block_onsets = np.split(time_series, np.where(np.diff(time_series) > threshold)[0] + 1)
    block_onsets = [max(x) for x in block_onsets]
    #block_onsets = [max(x) for x in block_onsets]
    try:
        assert len(block_onsets) == len(block_list)
    except AssertionError:
        print(f"detected blocks: {len(block_onsets)}")
        print(f"expected blocks: {len(block_list)}")


    #elif len(block_onsets) > n_blocks:
    #    long_blocks = []
    #    for block in block_onsets:
    #        if max(time_series) - block > 60:
    #            long_blocks.append(block)
    #    block_onsets = long_blocks[len(long_blocks) - n_blocks:]

    df.onset = block_onsets
    df.block_number = block_list
    return df


def get_block_onsets_alt(time_series, duration_series, value_series, n_blocks=10, event_threshold=20, threshold=5):
    long_events = np.where(duration_series > 0.06)[0]
    long_event_times = time_series[long_events]
    long_event_values = value_series[long_events]
    distance_between_long = np.diff(long_event_times)
    block_times = np.split(long_event_times, np.where(distance_between_long > 5)[0]+1)
    block_values = np.split(long_event_values, np.where(distance_between_long > 5)[0]+1)
    for t, v in zip(block_times, block_values):
        if len(t) > 4:
            plt.scatter(t, v)
            plt.show()
    #diode_on = np.where(value_series >= event_threshold)[0]
    #time_on = time_series[diode_on]
    #value_on = value_series[diode_on]
    #diffs = np.diff(time_on)
    #x = np.split(inside_frames.index, np.where(diffs > 100)[0])
    #blocks = []
    #for y in x:
    #    # print(marker_data.world_index.values[y])
    #    plt.plot(marker_data.world_index.values[y][:-1], np.diff(marker_data.world_index[y]))
    #    plt.show()
    #    start_markers = marker_data.world_index.values[y][np.where(np.diff(marker_data.world_index[y]) > 20)[0] + 1]
    #    blocks.append(start_markers[-1])
    #return (blocks[::-1][:n_blocks][::-1])


def get_event_onsets(df, block_onsets):
    # filter down
    df = df[df.duration < 1]
    df = df.reset_index(drop=True)
    # find values inside the blocks
    df = add_block_structure(df, block_onsets)

    return df


