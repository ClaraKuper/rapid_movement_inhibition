import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as st


mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False


def plot_single_participant_rates(movement_rates, scale, parameters, color_dict, cols=4, panel_size=4):
    rows = len(movement_rates) / cols
    rows = int(np.ceil(rows))
    fig, axs = plt.subplots(rows, cols, figsize=(panel_size * cols, panel_size * rows))
    for idx, participant in enumerate(movement_rates):
        for condition in movement_rates[participant]:
            axs.flatten()[idx].plot(scale, movement_rates[participant][condition],
                                    label=condition,
                                    color=color_dict[condition])
            axs.flatten()[idx].scatter(parameters[participant][condition]['latency'],
                                       parameters[participant][condition]['minimum'],
                                       color=color_dict[condition])
            axs.flatten()[idx].hlines(parameters[participant][condition]['baseline'],
                                      min(scale),
                                      max(scale),
                                      color=color_dict[condition])

        axs.flatten()[idx].set_xlim([-600, 850])
        axs.flatten()[idx].set_ylim([-0.1, 9])
        axs.flatten()[idx].set_title(participant[:8])
        if idx == len(movement_rates) - 1:
            axs.flatten()[idx].legend()
    plt.tight_layout()


def plot_average_participant_rates(movement_rates, ci_dict, scale, parameters, color_dict, axs):
    for condition in movement_rates:
        axs.plot(scale,
                 movement_rates[condition],
                 color=color_dict[condition],
                 label=condition,
                 linewidth=3)
        axs.fill_between(scale,
                         ci_dict[condition]['ci_upper'],
                         ci_dict[condition]['ci_lower'],
                         alpha=0.3,
                         color=color_dict[condition])
        latencies = [parameters[p][condition]['latency'] for p in parameters]
        minimum = [parameters[p][condition]['minimum'] for p in parameters]

        axs.scatter(latencies, minimum, color=color_dict[condition], alpha=1, s=3)
    axs.set_ylim([-0.1, 9])
    axs.set_xlim([-600, 850])
    axs.set_xlabel('time [ms] since event')
    axs.set_ylabel('movement rates [onsets/s]')
    axs.legend()


def make_figure_rates(height, sides):
    main_width = height
    side_height = height / len(sides)
    side_width = side_height

    ratio = int(np.floor(main_width / side_width))

    fig = plt.figure(layout="constrained", figsize=(main_width + side_width, height))
    mosaic = [np.concatenate([["main"] * ratio, [x]]) for x in sides]
    ax_dict = fig.subplot_mosaic(mosaic)
    return ax_dict


def plot_average_participant_position(positions, ci_dict, scale, params, color_dict, axs):
    # fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    for condition in positions:
        axs.plot(scale,
                 positions[condition],
                 color=color_dict[condition],
                 label=condition,
                 linewidth=3)
        axs.fill_between(scale,
                         ci_dict[condition]['ci_upper'],
                         ci_dict[condition]['ci_lower'],
                         alpha=0.3,
                         color=color_dict[condition])
    axs.set_ylim([-0.1, 1.5])
    axs.set_xlim([-600, 850])
    axs.set_xlabel('time [ms] since event')
    axs.set_ylabel('average position error [dva]')
    axs.legend()


def plot_metrics(metrics, which_to_plot, axs, condition_color):
    for p in which_to_plot:
        sns.stripplot(data=metrics,
                      x='condition',
                      y=p,
                      hue='condition',
                      ax=axs[p],
                      palette=condition_color,
                      legend=False)
        axs[p].set_xticks([])


def hex_to_rgb(hexadec):
    rgb = []
    for i in (0, 2, 4):
        decimal = int(hexadec[i:i + 2], 16)
        rgb.append(decimal)

    return tuple(rgb)


def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def create_double_color_cols(data, col_1, col_2, base_colors, drop_per_id):
    dict_colors = {}
    data = data.sort_values([col_1, col_2])
    data = data.reset_index(drop=True, inplace=False)
    new_col_name = f'{col_1}_{col_2}'
    data[new_col_name] = [f'{data.loc[idx, col_1]}_{data.loc[idx, col_2]}' for idx in data.index]

    for c2_id, n_c2 in enumerate(np.unique(data[col_2])):
        for c1_id, n_c1 in enumerate(np.unique(data[col_1])):
            base_color = base_colors[c1_id]
            rgb_val = hex_to_rgb(base_color)
            rgb_val = [max(0, int(x - drop_per_id * c2_id * x)) for x in rgb_val]
            new_hex = rgb_to_hex(rgb_val[0], rgb_val[1], rgb_val[2])
            dict_colors[f'{n_c1}_{n_c2}'] = new_hex
    return data, new_col_name, dict_colors


def make_delay_figure(delay_dict, data, conditions, measured_times, time, color_dict, figure_name):
    condition_name = conditions.keys()
    figure, axs = plt.subplots(2, 3, sharex='all', sharey='row', figsize=(9, 4))
    axs[0, int(len(conditions) / 2)].set_xlabel('movement onset since event [ms]')

    current_axis = 0

    for cond in condition_name:
        cond_data = data.copy(deep=True)
        feat = conditions[cond]

        for f in feat:
            cond_data = cond_data[cond_data[f] == feat[f]]

        for t in measured_times:
            vals = [delay_dict[p][cond][f'{t}_diff'] for p in delay_dict]
            mean_vals = np.mean(vals, axis=0)
            ci_lower, ci_upper = st.t.interval(confidence=0.95, df=len(vals) - 1,
                                               loc=mean_vals,
                                               scale=st.sem(vals))
            axs[0, current_axis].plot(time, mean_vals, color=color_dict[t], label=cond)
            axs[0, current_axis].fill_between(time, ci_lower, ci_upper, color=color_dict[t], alpha=0.2)
            axs[0, current_axis].scatter(time[np.where(ci_lower > 0)],
                                         np.ones(len(np.where(ci_lower > 0)[0])) * [190, 170][t == 'rest'],
                                         color=color_dict[t])

        axs[0, current_axis].set_title(cond)

        trial_copy, new_col_name, color_dict_hist = create_double_color_cols(cond_data.copy(deep=True),
                                                                             'choiceOrder',
                                                                             'prolific_id',
                                                                             ['AE43C7', 'FF3CC7', 'F0F600',
                                                                              '00E5E8', '007C77', '007C77'],
                                                                             0.05)
        sns.histplot(data=trial_copy,
                     x='touchOff_relative',
                     hue=new_col_name,
                     multiple='stack',
                     palette=color_dict_hist,
                     legend=False,
                     edgecolor=None,
                     ax=axs[1, current_axis])

        current_axis += 1

    plt.tight_layout()
    plt.savefig(figure_name)
    plt.show()
