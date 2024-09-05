import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.saccade_detection import engbert_mergenthaler
from src.helper_functions import threshold_setnan_entries_from_df

if __name__ == '__main__':
    gaze = pd.read_csv('../data/test_data/test_gaze_data.csv')
    gaze = threshold_setnan_entries_from_df(gaze, ['x_norm', 'y_norm'], 'confidence', 0.9)
    gaze = gaze
    x = gaze['x_norm']
    y = gaze['y_norm']

    frequency = 1 / np.median(np.diff(gaze['gaze_timestamp']))
    dist = np.sqrt(np.array(x) ** 2 + np.array(y) ** 2)
    velocity = np.diff(dist) * frequency
    amplitudes, pvels, saccdurs, onsets, offsets = engbert_mergenthaler(gaze['x_norm'],
                                                                        gaze['y_norm'],
                                                                        gaze['gaze_timestamp'],
                                                                        1.5, frequency)
    print(velocity)
    plt.plot(gaze['gaze_timestamp'][1:], velocity, alpha=0.2)
    plt.vlines(onsets, 0, 1, color='black')
    plt.vlines(offsets, 0, 1, color='grey')
    # plt.xlim(3450, 3500)
    plt.ylim(-5, 5)
    plt.show()


