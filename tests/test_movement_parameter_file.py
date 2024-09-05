import src.movement_rates as mov

input_path = './tests/normalized_rates.csv'
output_path = './tests/'
output_file = 'save_rate_params.csv'

params = {'search_start': 0,
          'search_end': 700}

mov.get_movement_rate_parameters(input_path,
                                 output_path,
                                 output_file,
                                 params)