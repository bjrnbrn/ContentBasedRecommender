import os
import argparse
import concurrent.futures
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pathlib import Path
base = Path('__file__').resolve().parent

from utils.mixscrap import MixesDbGenres, GenreSets, Tracklist 
from utils.eda import Mixes, Tracks, load_train_data, load_test_data
from utils.models import CNNModel


def parse_arguments():
    parser = argparse.ArgumentParser(description='Feature Export')
    parser.add_argument(
        '--json_file',
        type=str,
        default=base/'data'/'djmix-dataset.json',
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
        default=base/'data'/'tracks',
        help='Path to tracks folder')
    parser.add_argument(
        '--feature_type',
        type=str,
        default='spectrograms',
        help='Type of feature to extract'
        )
    parser.add_argument(
        '--label_type',
        type=str,
        default='genre',
        help='Type of label to predict'
        )
    return parser.parse_args()


def process_batch(batch, i, j):
    pickle_path = base / 'data' / 'pkls'
    os.makedirs(pickle_path, exist_ok=True)
    filename = pickle_path / 'train_features_batch{}.pkl'.format(i + 1)
    if os.path.exists(filename):
        print(f"File '{filename}' already exists. Skipping for batch {i + 1} of {j}.")
        return
    
    batch.load_audio_data()
    batch.extract_audio_features()
    batch.to_pickle(filename)
    print(f'Extraction of audio features for batch {i + 1} of {j} successfull')
    batch.export_spectrograms()
    print(f'Export of spectrogram imagery for batch {i + 1} of {j} successfull')




def main():

    args = parse_arguments()
    print('\n\n')
    print('___________________________________________________________________')
    print('-------------------------------------------------------------------')
    print(' A. WEB SCRAPING FOR MIXDB DATA')
    print('-------------------------------------------------------------------')
    # List all Genres and Subgenres of the MixesDb Website
    genres = MixesDbGenres()
    genres_mixes = []
    for genre, links in genres.items():
        if links[1] == 'genre':
            genre_mixes = GenreSets(links[0])
            genres_mixes.append(genre_mixes)

    for i in genres_mixes:
        print(i)

            


    print('-------------------------------------------------------------------')
    print('\n\n')
'''    
    load_genres = ['Acid', 'Dub', 'Ghetto', 'Pop', 'Soul']
    # TODO: Write as a function
    for genre in load_genres:
        genre_link = genres[genre][0]
        genre_sets = GenreSets(genre_link)
        i = 0
        
        for key, mix_link in genre_sets.items():
            # if i < 20:
            mix_id = f'{genre.lower()}_{key}'
            print('-------------------------------------')
            print(f'mix id:   {mix_id}')
            print('-------------------------------------')
            Tracklist(mix_link)
            i += 1
            print('-------------------------------------------------------------------\n')
        print('-------------------------------------------------------------------\n\n')


    print('\n')
    print('-------------------------------------------------------------------')
    print(' B. EXTRACT - TRANSFORM - LOAD')
    print('-------------------------------------------------------------------')
    
    mixes = Mixes(
        args.json_file,
        args.min_files,
        args.max_files,
        args.tracks_path
        )
    mixes.summary() 

    tracks = Tracks(mixes)
    
    # train_batches = tracks.split_df()
    # print('NUMBER OF BATCHES: ', len(train_batches))


    # j = len(train_batches)
    # with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
    #     print('AVAILABLE CORES:   ', os.cpu_count())
    #     print('USED CORES:        ', executor._max_workers)
    #     # Process each batch in parallel using multi-threading
    #     futures = [
    #         executor.submit(
    #             process_batch(batch_data, i, j)
    #             ) for i, batch_data in enumerate(train_batches)
    #         ]
    #     concurrent.futures.wait(futures)
    
    

    X, y, NUM_CLASSES = load_train_data(
        tracks,
        args.feature_type,
        args.label_type
        )



    print('\n')
    print('-------------------------------------------------------------------')
    print(' C. CNN MODEL & PREDICTIONS')
    print('-------------------------------------------------------------------')
    
    cnn_model = CNNModel(num_classes=NUM_CLASSES)
    cnn_model.train(X, y)

    X_TEST, NAMES = load_test_data(args.feature_type)
    

    predictions = cnn_model.predict(X_TEST)
    for i, prediction in enumerate(predictions):
        print(NAMES[i], prediction)



    print('\n')
    print('-------------------------------------------------------------------')
    print(' D. MODEL EVALUATION')
    print('-------------------------------------------------------------------')
    eval = ''
    

    print('\n')
    print('-------------------------------------------------------------------')
    print(' E. EXPORT OR FRONTEND CONNECTION')
    print('-------------------------------------------------------------------')
    export = ''


    print('\n')
'''



if __name__ == '__main__':
    
    main()
    