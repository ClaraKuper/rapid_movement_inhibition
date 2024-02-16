import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
from matplotlib.patches import Rectangle


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


def plot_average_participant_rates(movement_rates, ci_dict, scale, parameters, cluster, color_dict,
                                   line_dict, axs, upper_y = 1.5):
    for condition in movement_rates:
        upper_lim = []
        axs.plot(scale,
                 movement_rates[condition],
                 color=color_dict[condition],
                 linestyle = line_dict[condition],
                 label=condition,
                 linewidth=1)
        axs.fill_between(scale,
                         ci_dict[condition]['ci_upper'],
                         ci_dict[condition]['ci_lower'],
                         alpha=0.3,
                         color=color_dict[condition])
        latencies = [parameters[p][condition]['latency'] for p in parameters]
        minimum = [parameters[p][condition]['minimum'] for p in parameters]
        upper_lim.append(min(6, max(ci_dict[condition]['ci_upper'])))

        axs.scatter(latencies, minimum, color=[color_dict[condition], "White"][int(line_dict[condition] == '--')], edgecolors=color_dict[condition], alpha=1, s=3)
    sig_line = 0

    for cond in cluster:
        for c in cluster[cond]:
            axs.hlines(sig_line, scale[min(c)], scale[max(c)], colors=color_dict[cond], linestyles=line_dict[cond])
        sig_line += 0.02
    axs.set_ylim([-0.1, upper_y])
    axs.set_xlim([-200, 850])
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


def plot_average_participant_position(positions, ci_dict, scale, params, cluster, color_dict, line_dict, axs):
    for condition in positions:
        axs.plot(scale,
                 positions[condition],
                 color=color_dict[condition],
                 linestyle= line_dict[condition],
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


def plot_metrics(metrics, which_to_plot, condition_color, figure_path):

    for p in which_to_plot:
        gx = sns.stripplot(data=metrics,
                           x='condition',
                           y=p,
                           hue='condition',
                           palette=condition_color,
                           legend=False)
        gx.get_figure().savefig(f'{figure_path}/{p}.svg')
        plt.show()


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


def make_delay_figure(delay_dict, data, conditions, measured_times, time, significant_clusters, heatmap_clusters,
                      color_dict, line_dict, figure_name, heatmap_dict, figure_heatmap_name, cluster_df):
    condition_name = conditions.keys()
    figure, axs = plt.subplots(2, 1, sharex='all', figsize=(2.5, 2))
    axs[1].set_xlabel('movement onset since event [ms]')
    axs[0].set_xlim([-200, 850])

    for cond in condition_name:
        cond_data = data.copy(deep=True)
        feat = conditions[cond]

        for f in feat:
            cond_data = cond_data[cond_data[f] == feat[f]]

        for t in measured_times:
            vals = [delay_dict[p][cond][t] for p in delay_dict]
            print(f'{t}, {cond} has a standard deviation of {np.nanstd(vals)}')
            mean_vals = np.mean(vals, axis=0)
            ci_lower, ci_upper = st.t.interval(confidence=0.95, df=len(vals) - 1,
                                               loc=mean_vals,
                                               scale=st.sem(vals))
            axs[int(t == 'rest')].plot(time, mean_vals, color=color_dict[cond], label=cond, linestyle=line_dict[cond])
            axs[int(t == 'rest')].fill_between(time, ci_lower, ci_upper, color=color_dict[cond], alpha=0.2)

    for col in significant_clusters:
        sig_line = [100, 60][int(col == 'rest')]
        for cond in significant_clusters[col]:
            for c in significant_clusters[col][cond]:
                axs[int(col == 'rest')].hlines(sig_line, time[min(c)], time[max(c)],
                                               color=color_dict[cond], linestyle=line_dict[cond])
                sig_line += [10, 2][int(col == 'rest')]
                if col == 'rest':
                    axs[1].set_ylim(55, 85)
                else:
                    axs[0].set_ylim(95, 320)
    plt.tight_layout()
    plt.savefig(figure_name)
    plt.show()

    current_axis = 0
    heatmap_fig, heatmap_axs = plt.subplots(2, 2, sharex='all', sharey='all', figsize=(3, 2.5))
    participants = [p for p in heatmap_dict]
    jumpflash = pd.DataFrame(np.mean([heatmap_dict[p]['flash+ jump+'] for p in participants], axis=0),
                             columns=heatmap_dict[participants[0]]['flash+ jump+'].columns,
                             index=heatmap_dict[participants[0]]['flash+ jump+'].index)
    jumpnoflash = pd.DataFrame(np.mean([heatmap_dict[p]['flash- jump+'] for p in participants], axis=0),
                               columns=heatmap_dict[participants[0]]['flash- jump+'].columns,
                               index=heatmap_dict[participants[0]]['flash- jump+'].index)
    nojumpflash = pd.DataFrame(np.mean([heatmap_dict[p]['flash+ jump-'] for p in participants], axis=0),
                               columns=heatmap_dict[participants[0]]['flash+ jump-'].columns,
                               index=heatmap_dict[participants[0]]['flash+ jump-'].index)
    nojumpnoflash = pd.DataFrame(np.mean([heatmap_dict[p]['flash- jump-'] for p in participants], axis=0),
                                 columns=heatmap_dict[participants[0]]['flash- jump-'].columns,
                                 index=heatmap_dict[participants[0]]['flash- jump-'].index)

    a = sns.heatmap(data=nojumpnoflash,
                    ax=heatmap_axs[0, current_axis],
                    vmin=0, vmax=0.01,
                    cmap='Greys', cbar=False)

    b = sns.heatmap(data=nojumpflash,
                    ax=heatmap_axs[1, current_axis], vmin=0, vmax=0.01,
                    cmap='Greys', cbar=False)

    for cluster in heatmap_clusters['flash+ jump-']:
        for array in np.array(cluster).T:
            b.add_patch(Rectangle((array[1], array[0]), 1, 1, fill=True, alpha=0.5, edgecolor='none'))

    for idx in cluster_df[cluster_df.condition == 'flash+ jump-'].index:
        heatmap_axs[1, current_axis].scatter(cluster_df.loc[idx, 'center_time_loc'],
                                             cluster_df.loc[idx, 'center_latency_loc'],
                                             marker='+', color='black')

    current_axis += 1

    c = sns.heatmap(data=jumpnoflash,
                    ax=heatmap_axs[0, current_axis], vmin=0, vmax=0.01,
                    cmap='Greys', cbar=False)

    for cluster in heatmap_clusters['flash- jump+']:
        for array in np.array(cluster).T:
            c.add_patch(Rectangle((array[1], array[0]), 1, 1, fill=True, alpha=0.5, edgecolor='none'))

    for idx in cluster_df[cluster_df.condition == 'flash- jump+'].index:
        heatmap_axs[0, current_axis].scatter(cluster_df.loc[idx, 'center_time_loc'],
                                             cluster_df.loc[idx, 'center_latency_loc'],
                                             marker='+', color='black')
    d = sns.heatmap(data=jumpflash,
                    ax=heatmap_axs[1, current_axis], vmin=0, vmax=0.01,
                    cmap='Greys', cbar=False)

    for cluster in heatmap_clusters['flash+ jump+']:
        for array in np.array(cluster).T:
            d.add_patch(Rectangle((array[1], array[0]), 1, 1, fill=True, alpha=0.5, edgecolor='none'))

    for idx in cluster_df[cluster_df.condition == 'flash+ jump+'].index:
        heatmap_axs[1, current_axis].scatter(cluster_df.loc[idx, 'center_time_loc'],
                                             cluster_df.loc[idx, 'center_latency_loc'],
                                             marker='+', color='black')
    plt.tight_layout()
    plt.savefig(figure_heatmap_name)


def make_latency_heatmaps(heatmap_dict, conditions, save_path=None):
    fig, axs = plt.subplots(nrows=len(heatmap_dict), ncols=len(conditions), figsize=(15, 30), sharex=True, sharey=True)
    for row_idx, p in enumerate(heatmap_dict):
        for col_idx, cond in enumerate(conditions):
            hm = sns.heatmap(heatmap_dict[p][cond], vmin=.01, vmax=0.025, ax=axs[row_idx,col_idx])
    if save_path:
        plt.savefig(save_path)
