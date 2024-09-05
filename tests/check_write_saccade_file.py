from src.saccade_detection import get_saccades_by_block


def main():
    file_name = '../data/test_data/test_gaze_data.csv'
    get_saccades_by_block(file_name)


if __name__ == '__main__':
    main()
