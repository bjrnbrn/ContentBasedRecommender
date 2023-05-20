import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import cv2

from pathlib import Path
BASE = Path('__file__').resolve().parent

data_path = BASE / 'data'
tracks_path = data_path / 'tracks'
specs_path = data_path / 'spectrograms'




class Mixes(pd.DataFrame):

    def __init__(self, json_file, min_files=None, max_files=None, tracks_path=tracks_path):
        mixdata = self._load_json(json_file)
        super().__init__(mixdata)
        self._clean_data()
        self._files_available(min_files, max_files, tracks_path)

    def summarize(self):
        print(self.__str__())
        self._print_genre_summary()

    def __str__(self):
        mix_id_count = len(self['mix_id'])
        files_count_sum = self['files_count'].sum()
        min_files = self['files_count'].min()
        max_files = self['files_count'].max()
        genre_counts = len(self['genre'].unique())
        return (
            f'\nNUMBER OF MIXES:  {mix_id_count}' #pass to model
            f'\nAVAILABLE FILES:  {files_count_sum}'
            f'\nFILES IN MIXES:   {min_files} TO {max_files}'
            f'\nGENRES IN MIXES:  {genre_counts}') #pass to model

    def _print_genre_summary(self):
        genre_counts = self['genre'].value_counts().sort_values(ascending=False)
        genre_summary = genre_counts.head(5).append(genre_counts.tail(5))
        print('\nGENRE SUMMARY:')
        print('-------------------------------------------------------------------')
        for genre, count in genre_summary.items():
            files_count = self.loc[self['genre'] == genre, 'files_count'].sum()
            print(f'total of {count} mix(es), {files_count} file(s) in {genre.upper()}')
        print('-------------------------------------------------------------------', '\n')


    def _load_json(self, json_file):
        with open(json_file) as f:
            mixdata = pd.read_json(f)
        return mixdata


    def _clean_data(self):
        self.columns = [
            'mix_id', 'mix_title', 'url', 'audio_source',
            'audio_url', 'tracks_identified', 'tracks_total',
            'transitions', 'timestamps', 'tracklist', 'genre']
        
        self['genre'] = self['genre'].apply(
            lambda x: x[-1]['key'].replace('Category:', ''))
        self['year'] = self['mix_title'].str[:4].apply(
            lambda x: int(x) if x.isdigit() else None)
        
        self['year'] = self['year'].ffill().astype(int)
        self['tracklist'] = self['tracklist'].apply(
            lambda x: [track['id'] for track in x]
            )
        
        self.drop(['mix_title', 'url', 'audio_source',
                'audio_url', 'transitions', 'timestamps'],
                axis=1,
                inplace=True)


    def _files_available(self, min_files, max_files, tracks_path):
        file_names = [os.path.splitext(f)[0] for f in os.listdir(tracks_path)]
        
        self['available_files'] = self['tracklist'].apply(
            lambda x: [str(track_id) for track_id in x if str(track_id) in file_names] or np.nan)
        
        self.dropna(subset=['available_files'], inplace=True)

        self['files_count'] = self['available_files'].apply(
            lambda x: len(x))
        
        if min_files is not None:
            self.drop(self[self['files_count'] < min_files].index, inplace=True)
        if max_files is not None:
            self.drop(self[self['files_count'] > max_files].index, inplace=True)
        self.reset_index(drop=True, inplace=True)




class Tracks(pd.DataFrame):

    def __init__(self, source):

        if isinstance(source, Mixes):
            exploded = source.explode('available_files')[
                ['available_files', 'genre', 'mix_id']]
            
            exploded['input_path'] = exploded.apply(
                lambda row: f'{tracks_path}/{row["available_files"]}.mp3',
                axis=1)
            exploded['output_path'] = exploded.apply(
                lambda row: f'{specs_path}/train/{row["mix_id"]}/',
                axis=1)
            exploded = exploded.reset_index(drop=True)
            super().__init__(exploded)
            self.columns = [
                'track_id',
                'genre',
                'mix_id',
                'input_path',
                'output_path']

        elif isinstance(source, pd.DataFrame):
            data = {
                'track_id': source['track_id'],
                'genre': source['genre'],
                'mix_id': source['mix_id'],
                'input_path': source['input_path'],
                'output_path': source['output_path']
                }
            super().__init__(data)
        
        elif isinstance(source, str) and source.endswith('.pkl'):
            with open(source, 'rb') as file:
                data = pd.read_pickle(file)
            super().__init__(data)

        elif isinstance(source, str):
            extensions = ['.wav', '.mp3', '.flac', '.aif']
            file_names = [
                f for f in os.listdir(source) if any(f.endswith(ext) for ext in extensions)]
            track_ids = [os.path.splitext(f)[0] for f in file_names]
            
            data = {
                'track_id': track_ids,
                'genre': [np.nan] * len(file_names),
                'mix_id': 'test',
                'input_path': [os.path.join(source, f) for f in file_names],
                'output_path': f'{specs_path}/test/'
                }
            super().__init__(data)
        
        else:
            raise ValueError("Invalid input data type. Expected Mixes object or string path.")
        print(f'NUMBER OF NON-UNIQUE TRACKS: {len(self)}')


    def split_df(self, batch_size=50):
        num_batches = (len(self) // batch_size) + 1
        batches = np.array_split(self, num_batches)
        tracks_batches = []
        for batch in batches:
            tracks_batch = Tracks(batch)
            tracks_batches.append(tracks_batch)
        return tracks_batches  
    

# TODO: Script gets killed when loading too many audio files.
    def load_audio_data(self):
        def transform(x):
            try:
                y, rate = librosa.load(x, sr=None)
                return (y, rate)
            except:
                print(f'Failed to load {x}')
                return (None, None)
        transformed = self['input_path'].apply(lambda x: transform(x))
        self['audio_data'] = transformed.apply(lambda x: x[0])
        self['sample_rate'] = transformed.apply(lambda x: x[1])
        self = self.dropna(subset=['audio_data'])
        # return self  


# TODO: extract CHROMAGRAM, maybe more features
    def extract_audio_features(self):
        y = self['audio_data']
        rate = self['sample_rate']

        self['duration'] = y.apply(
            lambda x: librosa.get_duration(
            y=x, sr=rate.iloc[0]))
        self['num_channels'] = y.apply(
            lambda x: x.shape[1] if len(x.shape) > 1 else 1)
        self['amplitude_range'] = y.apply(
            lambda x: (x.min(), x.max()))
        # return self


    def export_spectrograms(self):
        for index, row in self.iterrows():
            track_id = row['track_id']
            output_path = row['output_path']

            y = row['audio_data']
            sr = row['sample_rate']
            data = librosa.feature.melspectrogram(y=y, sr=sr)

            plt.figure(figsize=(4,2.5))
            plt.axis('off')
            data_db = librosa.power_to_db(S=data, ref=np.max)
            librosa.display.specshow(
                data_db, sr=sr,
                x_axis='time',
                y_axis='mel',
                cmap='viridis'
                )#, vmin=0, vmax=1)
            plt.tight_layout()

            os.makedirs(output_path, exist_ok=True)
            plt.savefig(f'{output_path}{track_id}.png')
            plt.close()
        plt.close()




# UTILITY FUNCTIONS OF MODULE
def load_train_data(tracks, feature_type, label_type):
    X = list()
    y = list()
    file_path = BASE / 'data' / feature_type / 'train'
    for label in os.listdir(file_path):
        label_dir = os.path.join(file_path, label)
        if os.path.isdir(label_dir):
            
            for feature_file in os.listdir(label_dir):
                feature_path = os.path.join(label_dir, feature_file)
                feature_id = feature_file.split('.')[0]
                track_ids = tracks['track_id'].tolist()
                if feature_id in track_ids:

                    feature = cv2.imread(feature_path)
                    genre = tracks.loc[tracks['track_id'] == feature_id, 'genre'].values[0]
                    mix = label
                    X.append(feature)
                    if label_type == 'mix':   
                        y.append(mix)
                    elif label_type == 'genre':
                        y.append(genre)
    X = np.array(X)
    num_classes = len(set(y))
    print(
        '\nTRAINING DATA:'
        f'\nNo. of Images:             {len(X)}'
        f'\nNo. of labeled Images:     {len(y)}'
        f'\nNo. of unique Labels:      {num_classes}'
        )
    return X, y, num_classes


def load_test_data(feature_type):
    X = list()
    filenames = list()
    file_path = BASE / 'data' / feature_type / 'test'
    files = os.listdir(file_path)
    for file in files:
        filenames.append(file.split('.')[0])
        feature = os.path.join(file_path, file)
        feature = cv2.imread(feature)
        X.append(feature)
    X = np.array(X)
    return X, filenames

