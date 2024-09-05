import shutil
import pandas as pd
import numpy as np
import os


# Loop through each participant folder from '01' to '24'
def main(file_path):
    all_saccades = pd.DataFrame()
    all_events = pd.DataFrame()
    for i in range(1,25):
        for subfolder_name, prefix in zip(['target', 'background'], ['A', 'B']):
            calibration = pd.read_csv(f'{file_path}/{subfolder_name}/raw/calibration.csv')
             # Loop through each of the subfolders 'gaze' and 'move'
            for session in [1, 2]:
                    for surf_folder in ['surfaces', 'surfaces2']:
                        subfolder = f'{file_path}/{subfolder_name}/raw/P{str(i).zfill(2)}/{prefix}{session}'
                        saccade_folder = f'{subfolder}/saccades'

                        if not os.path.isdir(saccade_folder):
                            os.mkdir(saccade_folder)

                        if os.path.exists(subfolder):
                            filename = 'normalized_position_with_blocks.csv'
                            event_filename = 'trial_events_jatos.csv'
                            print(f'processing ppt {i}, condition {subfolder_name}, surface_folder {surf_folder}')
                            total_saccades = 0
                            try:
                                calib = calibration[calibration['participant_id'] == i]
                                calib = calib[calib['session_number'] == session]

                                df = pd.read_csv(f'{subfolder}/{surf_folder}/{filename}')
                                event_df = pd.read_csv(f'{subfolder}/{event_filename}')

                                block_ids = np.unique(df.block_id)
                                block_ids = block_ids[~np.isnan(block_ids)]

                                for b in block_ids:
                                    block_df = df[df.block_id == b]

                                    trial_ids = np.unique(block_df.trial_n)
                                    trial_ids = trial_ids[~np.isnan(trial_ids)]
                                    for t in trial_ids:

                                        trial_df = block_df[block_df.trial_n == t].copy(deep=True).reset_index(drop=True)

                                        # Convert TSV
                                        tsv_filename = filename.replace('.csv', f'_{b}_{t}_remodnav.tsv')
                                        # Compute gaze position in mm and save only x and y columns
                                        convert_csv_to_tsv(trial_df, f'{subfolder}/{surf_folder}/{tsv_filename}')
                                        # get the mm to dva transformation

                                        sample_length = np.median(np.diff(trial_df['gaze_timestamp']))
                                        hz = np.round(1/sample_length, 3)
                                        savgol_length = np.round(3.2/hz, 3)
                                        min_sac_dur = np.round(1.5*sample_length, 3)
                                        scaling_factor = 1/calib['px2deg'].iloc[0]
                                        trial_df['gaze_timestamp_remodnav'] = np.arange(0, len(trial_df) * sample_length, sample_length)
                                        file_target = filename.replace(".csv", f'_{b}_{t}_events.tsv')

                                        # Run remodnav command with scaling factor
                                        os.system(f"remodnav {subfolder}/{surf_folder}/{tsv_filename} {saccade_folder}/{file_target} {scaling_factor} {hz} --savgol-length {savgol_length} --min-saccade-duration {min_sac_dur} --noise-factor 3 --velthresh-startvelocity 100")
                                        # print(f'Processed file: {i} {subfolder_name} {session} {surf_folder} {tsv_filename}')

                                        # Convert TSV to CSV
                                        tsv_file_path = f'{saccade_folder}/{file_target}'
                                        csv_file_name = filename.replace(".csv", f"_{b}_{t}_events.csv")
                                        sac_file_name = filename.replace(".csv", f"_{b}_{t}_saccades.csv")

                                        df_tsv = pd.read_csv(tsv_file_path, sep='\t')
                                        df_tsv['participant'] = i
                                        df_tsv['condition'] = prefix
                                        df_tsv['session'] = session
                                        df_tsv['block'] = np.nan
                                        df_tsv['trial_n'] = np.nan
                                        df_tsv['trial_on_time'] = np.nan
                                        df_tsv['flash_shown'] = np.nan
                                        df_tsv['jump_shown'] = np.nan
                                        df_tsv['event_time'] = np.nan
                                        df_tsv['relative_eye_movement_time'] = np.nan

                                        for idx in df_tsv.index:
                                            time_diffs = trial_df['gaze_timestamp_remodnav'] - df_tsv.loc[idx, 'onset']
                                            info_line = trial_df.iloc[np.argmin(abs(time_diffs)), :]

                                            df_tsv.loc[idx, 'block']         = info_line['block_id']
                                            df_tsv.loc[idx, 'trial_n']       = info_line['trial_n']
                                            df_tsv.loc[idx, 'trial_on_time'] = info_line['gaze_timestamp_trial']

                                            if not np.isnan(info_line['trial_n']):
                                                event_block = event_df[event_df.block_id == info_line['block_id']]
                                                event_trial = event_block[event_block.trial_n == info_line['trial_n']]
                                                # print(event_trial)
                                                df_tsv.loc[idx, 'flash_shown'] = event_trial['flash'].values[0]
                                                df_tsv.loc[idx, 'jump_shown']  = event_trial['shift'].values[0]
                                                df_tsv.loc[idx, 'event_time']  = event_trial['start_event_in_trial'].values[0]
                                                df_tsv.loc[idx, 'relative_eye_movement_time'] = df_tsv.loc[idx, 'trial_on_time'] - df_tsv.loc[idx, 'event_time']

                                        # df_tsv.to_csv(f'{saccade_folder}/{csv_file_name}', index=False)
                                        # print(f'Converted TSV to CSV: {csv_file_name}')

                                        sac_df = df_tsv[((df_tsv.label == 'SACC') | (df_tsv.label == 'ISAC'))]
                                        sac_df = sac_df.dropna(subset = ['trial_n']).reset_index(drop = True)
                                        # sac_df.to_csv(f'{saccade_folder}/{sac_file_name}', index = False)

                                        if len(sac_df) == 0:
                                            print(f'\nParticipant {i}, Condition {subfolder_name}, Session {session}, \n'
                                                          f'Surface {surf_folder}, Block {b}, Trial {t}: \n'
                                                          f'NO SACCADES DETECTED!')

                                        all_events = pd.concat([all_events, df_tsv]).reset_index(drop = True)
                                        all_saccades = pd.concat([all_saccades, sac_df]).reset_index(drop = True)

                                        # save the combined df
                                        all_events.to_csv('../results/inlab/all_eye_movement_events.csv', index = False)
                                        all_saccades.to_csv('../results/inlab/all_saccades.csv', index = False)
                                        total_saccades += len(all_saccades)


                                        # delete unnessecary stuff
                                        os.remove(tsv_file_path)
                                        os.remove(f'{subfolder}/{surf_folder}/{tsv_filename}')
                                        # print(f'Saved all Saccades')

                            except FileNotFoundError as e:
                                print(e)
                                print(f'File not found: {filename}')

                        print(f'An average of {total_saccades/len(event_df)} '
                              f'saccades was detected across {len(event_df)} trials')

                        shutil.rmtree(saccade_folder)


def convert_csv_to_tsv(file, filename):
    new_file = file[['x_scaled', 'y_scaled']].copy(deep=True)
    new_file.to_csv(filename, index=False, header=False, sep='\t')


if __name__ == '__main__':
    file_path = '../data/inlab'
    main(file_path)
