import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def engbert_mergenthaler(x, y, time, thresh_factor, freq=500, plot=False):
    """
    saccade detection algorithm after engbert and mergenthaler/ engbert and kliegl: 
    https://www.sciencedirect.com/science/article/pii/S0042698903000841
    
    the velocity of saccades is being compared against a median std of the average saccade
    """
    # initialize saccade parameters to detect
    amplitudes = []
    onsets     = []
    offsets    = []
    pvels      = []
    saccdurs   = []
    first_sacc_sample = 0
    n_sample_around = 5
    
    # smooth x and y positions
    x = smooth_positions(x, 2)
    # same for the y position of the gaze
    y = smooth_positions(y, 2)
    
    time_smooth = np.array(time)[:-3]
    
    # combine x and y velocities!!
    # get distance travelled
    dist = np.sqrt(np.array(x)**2 + np.array(y)**2)
    dist_vel = np.diff(dist) * freq
    # find acceleration
    dist_acc = np.diff(dist_vel)

    # dist_noise = np.sqrt(np.median((dist_vel - np.median(dist_vel))**2))
    # compute threshold and apply
    adaptive_threshold = np.nanmedian(abs(dist_vel)) + thresh_factor * np.nanstd(abs(dist_vel))
    dist_thresh = np.where(abs(dist_vel) > adaptive_threshold)[0]

    if plot:
        plt.plot(time_smooth, dist_vel)
    # check if there are multiple saccades
    if any(np.diff(dist_thresh) > 4):
        # split into individual events
        split_dist_index = np.split(dist_thresh, np.where(np.diff(dist_thresh) > 4)[0]+1)
        for s_idx in split_dist_index:
            # check direction of saccade - forward or backward:
            if np.sign(np.mean(dist_vel[s_idx])) == np.sign(1):
                start_sign = 1
            else:
                start_sign = -1
            end_sign = start_sign*-1

            # look for the start of the acceleration (deceleration)
            # detect beginning of the saccade - go backwards till the sign flips
            start_search = max(s_idx[0] - 1, 0)
            while np.sign(start_sign) == np.sign(dist_acc[start_search]):
                start_search -= 1

                if start_search == 0:
                    break
            start_position = start_search

            # look for the start of the acceleration (deceleration)
            # detect beginning of the saccade - go forwards till the sign flips
            end_search = min(s_idx[-1]+1, len(dist_acc)-1)
            while np.sign(end_sign) == np.sign(dist_acc[end_search]):
                end_search += 1

                if end_search == len(dist_acc):
                    break
            end_position = end_search + 1

            vel_samples = dist_vel[start_position:end_position]
            time_samples = time_smooth[start_position:end_position]

            # filter the x and y position that belong to the saccade
            dist_x_saccade = np.array(x)[start_position:end_position]
            dist_y_saccade = np.array(y)[start_position:end_position]

            if plot:
                plt.plot(time_samples, vel_samples)

            try:
                onsets.append(time_samples[0])
                offsets.append(time_samples[-1])

                # get parameters
                amplitudes.append(np.sqrt((dist_x_saccade[0]-dist_x_saccade[-1])**2 +
                                      (dist_y_saccade[0]-dist_y_saccade[-1])**2))
                pvels.append(np.max(abs(vel_samples)))
                saccdurs.append(time_samples[-1] - time_samples[0])
            except IndexError as e:
                pass
    if plot:
        plt.show()
    return amplitudes, pvels, saccdurs, onsets, offsets


def get_intercept(vel_samples, time_samples, all_times, intercept=0):
        reg = LinearRegression().fit(vel_samples.reshape(-1, 1), time_samples.reshape(-1, 1))
        sample = reg.predict([[intercept]])[0][0]
        relative_time = all_times - sample
        position = np.argmin(abs(relative_time))
        return sample, position


def smooth_positions(pos, n_smooth):
    if n_smooth % 2 != 0:
        n_smooth = n_smooth-1
        print('non-even smoothing, will assume that the smoothing function is centered')
    new_series = []
    for entry in range(int(n_smooth/2), len(pos)-int(n_smooth/2)):
        new_series.append(np.mean(pos.iloc[entry-int(n_smooth/2):entry+int(n_smooth/2)]))
    return new_series


def get_longest_consecutive(seq):
    all_sequences = np.split(seq, np.where(np.diff(seq) > 1)[0]+1)
    val = [len(x) for x in all_sequences]
    return all_sequences[np.argmax(val)]


def scale_gaze_position(gaze_df, calibration_df):
    x_scaling_factor = calibration_df['win_width_deg'] * calibration_df['px2deg']
    y_scaling_factor = calibration_df['win_width_deg'] * calibration_df['px2deg']

    gaze_df.x_scaled = gaze_df.x_scaled * x_scaling_factor.iloc[0] # calibration_df['win_width_deg'].iloc[0]
    gaze_df.y_scaled = gaze_df.y_scaled * y_scaling_factor.iloc[0] # calibration_df['win_height_deg'].iloc[0]
    return gaze_df.x_scaled.values, gaze_df.y_scaled.values


def align_gaze_data_to_block_onset(gaze_df, marker_df, inside_marker, path, include_blocks):
    block_df = pd.DataFrame(columns=['onset_frame', 'block_id'])
    block_correct = 0
    save = True

    #try:
    #    block_frames = pd.read_csv(f'{path}/block_onsets_pupil.csv')
    #    block_df = block_frames.copy()
    #    block_frames = block_frames.onset_frame.values
    #    #block_frames = get_block_frames_alt(marker_df, inside_marker, n_blocks)
    #    save = False
    #except FileNotFoundError:
    block_frames = get_block_frames_alt(marker_df, inside_marker, len(include_blocks))
    gaze_df.loc[:, 'gaze_timestamp_block'] = np.nan
    gaze_df.loc[:, 'world_timestamp_block'] = np.nan

    for idx in range(len(block_frames)):
        # block_len defines an offset in case the first surface shown in the recording does not correspond to the first block
        block = include_blocks[idx]
        try:
            in_block = gaze_df[gaze_df.world_index.between(block_frames[idx], block_frames[idx+1], inclusive='left')].index
        except IndexError:
            in_block = gaze_df[gaze_df.world_index > block_frames[idx]].index
        try:
            if in_block[-1] - in_block[0] < 2800:
                block_correct -= 1
                print(in_block[-1] - in_block[0])
            else:
                gaze_df.loc[in_block, 'block_id'] = block
                block_df.loc[block-1, 'onset_frame'] = block_frames[idx]
                block_df.loc[block-1, 'block_id'] = block
                try:
                    gaze_df.loc[in_block, 'gaze_timestamp_block'] = gaze_df.loc[in_block, 'gaze_timestamp'] - gaze_df.loc[in_block[0], 'gaze_timestamp']
                    gaze_df.loc[in_block, 'world_timestamp_block'] = gaze_df.loc[in_block, 'world_timestamp'] - gaze_df.loc[in_block[0], 'world_timestamp']
                except IndexError:
                    print("ALERT ALIGNMENT")
                    print(gaze_df.loc[in_block, 'gaze_timestamp'])
                    # gaze_df.loc[in_block, 'gaze_timestamp'] -= gaze_df.loc[in_block, 'gaze_timestamp']
                    # gaze_df.loc[in_block, 'world_timestamp'] -= gaze_df.loc[in_block, 'world_timestamp']
        except IndexError:
            print(block)
            print(block_frames)
            print(idx)
            pass
        if save:
            block_df.to_csv(f'{path}/block_onsets_pupil.csv', index=False)
    return gaze_df.block_id.values, gaze_df.gaze_timestamp_block.values, gaze_df.world_timestamp_block.values


def get_block_frames(marker_data, inside_marker, n_blocks):
    inside_frames = marker_data[marker_data.marker_uid.isin(inside_marker)]
    diffs = inside_frames.world_index.diff()
    x = inside_frames.world_index.iloc[np.where(diffs > 100)[0]-1]
    x = np.append(x.index, max(inside_frames.index))[::-1][:n_blocks][::-1]
    world_frames = inside_frames.world_index[x]
    return world_frames.values


def get_block_frames_alt(marker_data, inside_marker, n_blocks):
    inside_frames = marker_data[marker_data.marker_uid.isin(inside_marker)]
    diffs = inside_frames.world_index.diff()

    x = np.split(inside_frames.index, np.where(diffs > 100)[0])
    blocks = []
    for y in x:
        # print(marker_data.world_index.values[y])
        #plt.plot(marker_data.world_index.values[y][:-1], np.diff(marker_data.world_index[y]))
        #plt.show()
        start_markers = marker_data.world_index.values[y][np.where(np.diff(marker_data.world_index[y]) > 20)[0]+1]
        # start_markers =
        blocks.append(start_markers[-1])

    return(blocks[::-1][:n_blocks][::-1])

def align_gaze_data_to_trial_onset(gaze_data, trial_data):
    gaze_data.loc[:, 'trial_id'] = np.nan
    gaze_data.loc[:, 'gaze_timestamp_trial'] = np.nan
    gaze_data.loc[:, 'world_timestamp_trial'] = np.nan

    for b in np.unique(gaze_data.block_id):
        if not np.isnan(b):
            gaze_in_block = gaze_data[gaze_data.block_id == b]
            block_trials = trial_data[trial_data.block_id == b]
            for t in np.unique(block_trials.trial_n):
                block_trial = block_trials[block_trials.trial_n == t]
                in_trial = gaze_in_block[gaze_in_block.gaze_timestamp_block.between(block_trial.start_trial_in_block.values[0],
                                                                                    block_trial.end_trial_in_block.values[0])]

                if len(in_trial) > 0:

                    gaze_data.loc[in_trial.index, 'trial_n'] = t
                    gaze_data.loc[in_trial.index, 'gaze_timestamp_trial'] = in_trial.loc[:, 'gaze_timestamp_block'] - in_trial.iloc[0]['gaze_timestamp_block']
                    gaze_data.loc[in_trial.index, 'world_timestamp_trial'] = in_trial.loc[:, 'world_timestamp_block'] - in_trial.iloc[0]['world_timestamp_block']

    return gaze_data['trial_n'].values, gaze_data['gaze_timestamp_trial'].values, gaze_data['world_timestamp_trial'].values

def detect_saccades(trial_df, gaze_df, light_data, block_len, block_id):
    saccade_df = pd.DataFrame(columns=['onset', 'onset_corrected', 'block', 'trial', 'flash', 'shift', 'onset_relative',
                                       'offset_relative', 'onset_trial', 'offset_trial', 'success', 'low_confidence'])
    idx = 0
    for b in range(int(max(block_len, 0)), int(np.nanmax(block_id))):
        block_trials = trial_df[trial_df.block_id == b + 1]
        block_gaze = gaze_df[gaze_df.block_id == b + 1]
        block_light = light_data[light_data.block_id == b + 1]

        for t_idx in block_trials.index:
            trial_gaze = block_gaze[block_gaze.gaze_timestamp.between(block_trials.start_trial_in_block[t_idx],
                                                                      block_trials.end_trial_in_block[t_idx], inclusive='both')]
            trial_light = block_light[
                block_light.start_event_in_block.between(block_trials.start_trial_in_block[t_idx],
                                                         block_trials.end_trial_in_block[t_idx])]
            saccade_df.loc[idx, 'onset_relative'] = block_trials['start_trial_in_block'][t_idx] - \
                                               block_trials['start_event_in_block'][t_idx]
            saccade_df.loc[idx, 'offset_relative'] = block_trials['end_trial_in_block'][t_idx] - \
                                                block_trials['start_event_in_block'][t_idx]
            try:
                saccade_df.loc[idx, 'onset_trial'] = trial_gaze.gaze_timestamp.values[0]
                saccade_df.loc[idx, 'offset_trial'] = trial_gaze.gaze_timestamp.values[-1]
            except IndexError:
                print(b)
                #print(block_trials.head())
                #print(block_gaze.head())
                #print(block_light.head())
                print(min(block_gaze.gaze_timestamp))
                print(block_trials.start_trial_in_block[t_idx])
                print(max(block_gaze.gaze_timestamp))
                print('==================================================================')
            frequency = 1 / np.median(np.diff(trial_gaze.gaze_timestamp))
            # run saccade detection on trial
            amplitude, pvel, saccdur, onset, offset = engbert_mergenthaler(trial_gaze.x_scaled,
                                                                           trial_gaze.y_scaled,
                                                                           trial_gaze.gaze_timestamp, 2,
                                                                           freq=frequency)
            for o, sd, a, pv in zip(onset, saccdur, amplitude, pvel):
                saccade_df.loc[idx, 'block'] = b+1
                saccade_df.loc[idx, 'trial'] = t_idx
                saccade_df.loc[idx, 'flash'] = block_trials['flash'][t_idx]
                saccade_df.loc[idx, 'shift'] = block_trials['shift'][t_idx]
                saccade_df.loc[idx, 'onset'] = o - block_trials['start_event_in_block'][t_idx]
                saccade_df.loc[idx, 'onset_corrected'] = o - block_trials['photodiode_onset_in_block'][t_idx]

                saccade_df.loc[idx, 'duration'] = sd
                saccade_df.loc[idx, 'amplitude'] = a
                saccade_df.loc[idx, 'pvel'] = pv

                saccade_df.loc[idx, 'success'] = block_trials['success'][t_idx]
                saccade_df.loc[idx, 'low_confidence'] = sum(np.isnan(trial_gaze.x_scaled)) / len(trial_gaze)

                idx += 1
    return saccade_df
