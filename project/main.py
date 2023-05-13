import argparse
from eda import Mixes, Tracks

def parse_arguments():
    parser = argparse.ArgumentParser(description='Feature Export')
    parser.add_argument(
        '--json_file',
        type=str,
        default='../data/djmix-dataset.json',
        help='Path to JSON file'
        )
    parser.add_argument(
        '--min_files',
        type=int,
        default=5,
        help='Minimum number of files'
        )
    parser.add_argument(
        '--max_files',
        type=int,
        default=20,
        help='Maximum number of files'
        )
    parser.add_argument(
        '--tracks_path',
        type=str,
        default='../data/tracks/',
        help='Path to tracks folder')
    # parser.add_argument(
    #     '--feature_type',
    #     type=str,
    #     default='spectrogram',
    #     help='Type of feature to export'
    #     )
    return parser.parse_args()

def main():
    args = parse_arguments()

    mixes = Mixes(args.json_file, args.min_files, args.max_files, args.tracks_path)
    train_tracks = Tracks(mixes)
    train_tracks.export_spectrograms()

if __name__ == '__main__':
    main()