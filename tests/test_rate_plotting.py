from src.plotting import plot_average_participant_rates

input_rate_file = './tests/movement_rates_raw.csv'
input_cluster_file = './tests/movement_cluster.csv'
color_dict = {'flash+ jump+': '#3DA7B8',
              'flash+ jump-': '#3DA7B8',
              'flash- jump+': '#333333',
              'flash- jump-': '#333333'
              }
line_dict = {'flash+ jump+': '-',
             'flash+ jump-': '--',
             'flash- jump+': '-',
             'flash- jump-': '--'
             }
figure_params = {
    'width': 3,
    'height': 3,
    'plot_only_significant_cluster': True,
    'linewidth': 1,
    'sem_alpha': 0.3,
    'ci': 0.95,
    'start_cluster_lines_at': 0,
    'space_cluster_lines_by': 0.5,
    'lower_y': -0.1,
    'upper_y': 7,
    'lower_x': -700,
    'upper_x': 1500,
    'y_title': 'movement rates [onsets/s]',
    'x_title': 'time [ms] since event'
}
output_file = './tests/test_rate_figure.pdf'

plot_average_participant_rates(input_rate_file, input_cluster_file, color_dict, line_dict, figure_params, output_file)