import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
from yt_dlp import YoutubeDL

from pathlib import Path
BASE = Path('__file__').resolve().parent

_throttle_count = 0


def download_track(track_id, out_path=os.path.join(BASE, 'data', 'tracks')):

    url = f'https://www.youtube.com/watch?v={track_id}'
    track_path = ''.join([out_path, f'{track_id}.mp3'])
    
    download_audio(
        url=url,
        path=track_path,
    )


def download_audio(url, path, max_throttle=100):
    if os.path.isfile(path):
        return f'{path} already exists. Skip downloading.'
  
    def throttle_detector(d):
        global _throttle_count
        
        if d['status'] == 'downloading' and d['speed'] is not None:
            speed_kbs = d['speed'] / 1024  # downloading speed in KiB/s
            if speed_kbs < 100:
                _throttle_count += 1
            else:
                _throttle_count = 0
        
            if _throttle_count > max_throttle:
                raise Exception(f'The download speed is throttled more than {max_throttle} times. Aborting.')
  
    missed = []
    params = {
        'format': 'bestaudio',
        'outtmpl': path[:-4],
        'postprocessors': [{  # Extract audio using ffmpeg
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            }],
        'progress_hooks': [throttle_detector],
        }
    with YoutubeDL(params) as ydl:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        try:
            ydl.download(url)
        except:
            missed.append(url)
        return missed
