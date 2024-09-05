import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter
from scipy.signal import freqz


def filter_events(sound_series, lowcut, highcut, fs, order=2, threshold=2):
    y = butter_bandpass_filter(sound_series, lowcut, highcut, fs, order=order)
    y_filter = y > threshold
    y_new = y * y_filter
    return y_new, y


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter(order, [lowcut, highcut], fs=fs, btype='band')
    y = lfilter(b, a, data)
    return y


def detect_touch_events(time_series, sound_series, merge_threshold=0.04):
    df = pd.DataFrame(columns=['start_time', 'event_id'])
    event = 0
    count = 0
    last_sample = time_series[0]
    time_on = sound_series[0]
    for sample, time in zip(sound_series, time_series):
        if sample > 0 and not event:
            event = 1
            time_on = time
            last_sample = time
        elif sample > 0 and event:
            last_sample = time
        elif sample == 0 and event:
            if time - last_sample >= merge_threshold:
                df.loc[count, 'start_time'] = time_on
                df.loc[count, 'event_id'] = count

                event = 0
                count += 1
    return df
