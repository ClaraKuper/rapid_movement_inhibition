import numpy as np
from src.json_parsing import filter_data
from src.plotting import plot_single_participant_rates


def get_movement_rates_by_participant(data, onset_column, offset_column, participant_column, analysis_parameter_dict,
                                      order_column, conditions_dict, condition_color_dict, plot=False):
    participants = np.unique(data[participant_column])
    ref_scale = np.arange(analysis_parameter_dict['window_start'], analysis_parameter_dict['window_end'], 1)
    movement_rates = {}
    rate_parameters = {}
    for p in participants:
        movement_rates[p] = {}
        rate_parameters[p] = {}
        p_data = data[data[participant_column] == p].reset_index(drop=True)

        for condition in conditions_dict:
            condition_dict = conditions_dict[condition]
            c_data = filter_data(p_data, condition_dict)
            movement_rate_raw, movement_rate, scale = get_normalized_rates(c_data, ref_scale, onset_column,
                                                                           offset_column, order_column,
                                                                           analysis_parameter_dict)
            assert np.all(ref_scale == scale)
            parameter_dict = get_rate_parameters(movement_rate, scale, analysis_parameter_dict)

            movement_rates[p][condition] = movement_rate
            rate_parameters[p][condition] = parameter_dict
            rate_parameters[p][condition]['flash_shown'] = conditions_dict[condition]['flashShown']
            rate_parameters[p][condition]['stim_jumped'] = conditions_dict[condition]['stimJumped']

    if plot:
        plot_single_participant_rates(movement_rates, scale, rate_parameters, condition_color_dict)
    return movement_rates, rate_parameters, scale


def get_rate_parameters(rate, scale, parameters):
    dictionary = {}
    search_scale, search_rate = filter_rate_and_scale(scale, rate, parameters['search_start'], parameters['search_end'])
    minimum, latency = get_min_latency(search_rate, search_scale)

    base_scale, base_rate = filter_rate_and_scale(scale, rate, parameters['baseline_start'], parameters['baseline_end'])
    baseline = get_baseline(base_rate)
    magnitude = baseline - minimum
    dictionary['minimum'] = minimum
    dictionary['latency'] = latency
    dictionary['baseline'] = baseline
    dictionary['magnitude'] = magnitude
    return dictionary


def filter_rate_and_scale(scale, rate, low, high):
    idx = np.where((low <= scale) & (high >= scale))[0]
    filter_rate = np.array(rate)[idx]
    filter_scale = np.array(scale)[idx]

    return filter_scale, filter_rate


def get_min_latency(rate, scale):
    minimum = min(rate)
    latency = scale[np.where(rate == minimum)]
    assert len(latency) == 1
    return minimum, latency[0]


def get_baseline(rate):
    return np.mean(rate)


def get_normalized_rates(data, scale, onset_column, offset_column, order_column, analysis_parameter_dict, n_trials = None):
    onsets = data[onset_column].dropna().astype(int).values
    offsets = data[offset_column].dropna().astype(int).values

    first_touches = data[data[order_column] == min(data[order_column])][onset_column]
    last_touches = data[data[order_column] == max(data[order_column])][onset_column]
    last_touches = data[data[order_column] == max(data[order_column])][onset_column]
    smooth_distribution = get_uniform_cdf(min(first_touches), max(first_touches), min(last_touches), max(last_touches),
                                          scale)
    if not n_trials:
        n_trials = len(first_touches)
    smooth_distribution = smooth_distribution * n_trials
    movement_rate_raw, movement_rate, scale = causal_rate(offsets, analysis_parameter_dict['window_start'],
                                                          analysis_parameter_dict['window_end'], smooth_distribution)
    return movement_rate_raw, movement_rate, scale


def causal_rate(move_onset, lock_window_start, lock_window_end, n_trials, alpha=1 / 50):
    """
     analyse rate in causal time window

     input:    move_onset  - movement onset times
               lock_window_start  - window before lock
               lock_window_end  - window after lock
               n_trials      - number of trials

     output:   rate    - movement rate
               scale   - time axis

    12.12.2005 by Martin Rolfs
    21.06.2021 translated to python by Clara Kuper
    """
    scale = np.arange(lock_window_start, lock_window_end, 1)
    # check how many trials these values came from
    if type(n_trials) == int:
        n_trials = np.linspace(n_trials, n_trials, len(scale))
    elif len(n_trials) != len(scale):
        raise ValueError('n_trials must have the same as the length of lock_window_start:lock_window_end!'
                         f'But has length {len(n_trials)} instead of {len(scale)}')
    # alpha defines how much the distribution is shifted
    alpha = alpha
    # define empty arrays for scale and rate
    rate = []
    raw_rate = []

    # loop through all time windows
    for idx, t in enumerate(scale):
        # compute tau
        # here is a filter for all events BEFORE time point t
        tau = t - move_onset + 1 / alpha
        # filter tau as event 0/1
        tau = tau[tau > 0]
        # get the number of saccades in a given window
        causal = alpha ** 2 * tau * np.exp(-alpha * tau)
        # save the rate
        rate.append(sum(causal) * 1000 / n_trials[idx])
        raw_rate.append(sum(causal) * 1000)
    return raw_rate, rate, scale


def get_uniform_cdf(first_trial_start, last_trial_start, first_trial_end, last_trial_end, scale):
    uniform_dist = []
    for val in scale:
        if val < first_trial_start:
            uniform_dist.append(0.00001)
        elif val >= first_trial_start and val < last_trial_start:
            uniform_dist.append((val - first_trial_start) / (last_trial_start - first_trial_start))
        elif val >= last_trial_start and val < first_trial_end:
            uniform_dist.append(1)
        elif val >= first_trial_end and val < last_trial_end:
            uniform_dist.append(- 1 * ((val - last_trial_end) / (last_trial_end - first_trial_end)))
        elif val >= last_trial_end:
            uniform_dist.append(0.00001)
        else:
            raise ValueError(
                f'{val} is a very weird value. {first_trial_start, last_trial_start, first_trial_end, last_trial_end}')
    return np.array(uniform_dist)
