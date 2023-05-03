import json
import pandas as pd
import os


def main(get_subjects, file_dir):
    for subject in get_subjects:
        s_count = 0
        s_trials = []
        # loop over files till we found one that matches the subject
        for component_dir in os.listdir(file_dir):
            full_dir = file_dir + '/' + component_dir
            if os.path.isdir(full_dir):
                with open(full_dir + '/trialData.json') as trial_data:
                    completed_trials = json.load(trial_data)
                    if not completed_trials[0]['prolific_id'] == subject:
                        continue
                with open(full_dir + '/allTimelineVariables.json') as trial_data:
                    all_trials = json.load(trial_data)

                completed_ids = []
                for x in completed_trials:
                    try:
                        if x['test_part'] == 'trial':
                            if x['success'] == 1:
                                completed_ids.append(x['trialID'])
                    except KeyError:
                        pass
                for trial in all_trials:
                    if trial['trialID'] in completed_ids:
                        pass
                    else:
                        trial['session_number'] = completed_trials[0]['session_number']
                        trial['session_id'] = completed_trials[0]['session_id']
                        trial['study_id'] = completed_trials[0]['study_id']
                        trial['unique_id'] = str(trial['trialID'])+trial['session_id']
                        s_trials.append(trial)
                        s_count += 1
        print(f'{subject} got {s_count} extra trials')
        with open(file_dir + '/' + subject + '_extra_trials.json', 'w') as extra_file:
            json.dump(s_trials, extra_file)


if __name__ == '__main__':
    main(['5c2279c061685000011d9d5b',
          '5fcea6e0350f1f14e9e09bd5',
          '607b623e240cc6eebb28aa64',
          '60f0102ca3ecb53656f7197d',
          '6155cbd24be7bcccea214120',
          '6007287ed787d81a04d1410c',
          '611d580c8ca031dceaf5bce5'
          ], '../data/study_data_multiple_sessions/extra_trials')
