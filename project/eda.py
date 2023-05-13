import os
import gc
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display




class Mixes(pd.DataFrame):

    def __init__(self, json_file, min_files=None, max_files=None, tracks_path='../data/tracks/'):
        mixdata = self._load_json(json_file)
        super().__init__(mixdata)
        self._clean_data()
        self._files_available(min_files, max_files, tracks_path)


    def _load_json(self, json_file):
        with open(json_file) as f:
            mixdata = pd.read_json(f)
        return mixdata

    def _clean_data(self):
        self.columns = [
            'mix_id',
            'mix_title',
            'url',
            'audio_source',
            'audio_url',
            'tracks_identified',
            'tracks_total',
            'transitions',
            'timestamps',
            'tracklist',
            'genre'
            ]
        # Extract genre from tags column
        self['genre'] = self['genre'].apply(
            lambda x: x[-1]['key'].replace('Category:', '')
            )
        # Extract year from title column
        self['year'] = self['mix_title'].str[:4].apply(
            lambda x: int(x) if x.isdigit() else None
            )
        self['year'] = self['year'].ffill().astype(int)
        # Extract track ids from tracklist column
        self['tracklist'] = self['tracklist'].apply(
            lambda x: [track['id'] for track in x]
            )
        self.drop([
            'mix_title',
            'url',
            'audio_source',
            'audio_url',
            'transitions',
            'timestamps'
            ],
            axis=1,
            inplace=True
            )

    def _files_available(self, min_files, max_files, tracks_path):
        # Check for file availability and drop NaNs
        file_names = [
            os.path.splitext(f)[0] for f in os.listdir(tracks_path)
            ]
        self['available_files'] = self['tracklist'].apply(
            lambda x: [str(track_id) for track_id in x if str(track_id) in file_names] or np.nan
            )
        self.dropna(subset=['available_files'], inplace=True)

        # Filter according to args
        self['files_count'] = self['available_files'].apply(
            lambda x: len(x)
            )
        if min_files is not None:
            self.drop(self[self['files_count'] < min_files].index, inplace=True)
        if max_files is not None:
            self.drop(self[self['files_count'] > max_files].index, inplace=True)
        self.reset_index(drop=True, inplace=True)



class Tracks(pd.DataFrame):

    def __init__(self, source):

        if isinstance(source, Mixes):
            # Explode available files column of Mixes object (track_id not unique)
            exploded = source.explode('available_files')[[
                'available_files',
                'genre'
                ]]
            exploded['path'] = exploded['available_files'].apply(
                lambda x: f'../data/tracks/{x}.mp3'
                )
            exploded = exploded.reset_index(drop=True)
            super().__init__(exploded)
            self.columns = ['track_id', 'genre', 'file_path']

        elif isinstance(source, str):
            # Check path for accepted audio files
            extensions = ['.wav', '.mp3', '.flac', '.aif']
            file_names = [
                f for f in os.listdir(source) if any(f.endswith(ext) for ext in extensions)
                ]
            track_ids = [os.path.splitext(f)[0] for f in file_names]
            data = {
                'track_id': track_ids,
                'genre': [np.nan] * len(file_names),
                'file_path': [os.path.join(source, f) for f in file_names]
                }
            super().__init__(data)

        else:
            raise ValueError("Invalid input data type. Expected Mixes object or string path.")


    def export_spectrograms(self, output_folder='../data/spectrograms/', batch_size=100):
        os.makedirs(output_folder, exist_ok=True)
        existing_files = os.listdir(output_folder)

        # Iterate over the tracks in batches
        for i in range(0, len(self), batch_size):
            batch = self.iloc[i:i+batch_size]

            for index, row in batch.iterrows():
                file_path = row['file_path']
                track_id = row['track_id']
                output_file = os.path.join(output_folder, track_id + '.png')
                if track_id + '.png' in existing_files:
                    continue

                audio_data, sr = librosa.load(file_path)
                spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr)
                spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

                plt.figure(figsize=(10, 4))
                librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
                plt.axis('off')
                plt.savefig(output_file, bbox_inches='tight', pad_inches=0)
                plt.close()
                del audio_data
                del sr
                del spectrogram
                del spectrogram_db
                gc.collect()
