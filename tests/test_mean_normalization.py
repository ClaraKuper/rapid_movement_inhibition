import src.movement_rates as mov

raw_rates = './tests/movement_rates_raw.csv'
baseline = 'flash- jump-'
out_path = './tests/'
out_file = 'normalized_rates.csv'

mov.mean_normalize_rates(raw_rates, baseline, out_path, out_file)
