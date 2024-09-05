import src.movement_rates as mov


def main():
    input_file = './data/remote/data_background/preprocessed/clean_remote_data.csv'
    move_offset_col = 'touchOn_relative'
    move_onset_col = 'touchOff_relative'
    participant_col = 'prolific_id'
    touch_order_col = 'choiceOrder'

    rate_analysis_parameters = {'window_start': -700,
                                'window_end': 1500,
                                'search_start': 0,
                                'search_end': 700,
                                'baseline_start': -100,
                                'baseline_end': 0,
                                'first_offset': 0,
                                'last_offset': 5,
                                'alpha': 1 / 50}

    conditions_rates = {'flash+ jump+': {'flashShown': 1,
                                         'stimJumped': 1},
                        'flash+ jump-': {'flashShown': 1,
                                         'stimJumped': 0},
                        'flash- jump+': {'flashShown': 0,
                                         'stimJumped': 1},
                        'flash- jump-': {'flashShown': 0,
                                         'stimJumped': 0}}

    out_path = './tests/'
    out_file = 'movement_rates_raw.csv'

    mov.get_movement_rates_by_participant(input_file, move_onset_col, move_offset_col, participant_col,
                                          touch_order_col, rate_analysis_parameters,
                                          conditions_rates, out_path, out_file)


if __name__ == '__main__':
    main()
