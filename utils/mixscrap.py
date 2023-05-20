import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import bs4
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

from pathlib import Path
BASE = Path('__file__').resolve().parent

from utils.tubeload import download_track



# GLOBAL
base_url = 'https://www.mixesdb.com'
mixdbtxt = BASE / 'data' / 'mixesdb.txt'
y_folder = BASE / 'data' / 'tracks'



class MixesDbGenres(dict):
    """
    Dictionary of all Genres and their
    Subgenres and their links
    keys: Name of Genre
    values: [http-link, genre level]
    TODO: make subgenres values of genres
    """
    def __init__(self, html=mixdbtxt):
        with open(html, 'r') as file:
            html_content = file.read()
        self.parse_html(html_content)
        print(self)


    def parse_html(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        for a in soup.find_all('a', href=True):
            prefix = '/w/Category:'
            
            if a['href'].startswith(prefix):
                link = f"{base_url}{a['href']}"
                name = a['href'].removeprefix(prefix)
                if a.find_previous_sibling('br'):
                    self[name] = [link, 'subgenre']
                else:
                    self[name] = [link, 'genre']

    def __str__(self):
        genre_count = sum(1 for value in self.values() if value[1] == 'genre')
        subgr_count = sum(1 for value in self.values() if value[1] == 'subgenre')
        return (
            '\nSUCCESSFULLY PARSED MIXDB GENRE SITE:'
            f'\nNo. of Genres:             {genre_count}'
            f'\nNo. of Subgenres:          {subgr_count}'
            f'\nTotal Links to Genres:     {len(self)}')



class GenreSets(dict):
    """
    Takes a link to a specific Genre page
    and returns links to all the sets of 
    that genre and their links
    """
    def __init__(self, genre_url: str, start_id=1):
        self.parse_html(genre_url, start_id)

    def parse_html(self, genre_url, start_id):
        r = requests.get(genre_url, headers={'User-agent': 'surfacer 1.3'})
        content = r.content
        soup = BeautifulSoup(content, 'html.parser')
        mixes_list = soup.find(
            'h2', {'id': 'Mixes'}
            ).find_next(
            'ul', {'id': 'catMixesList'}
            )
        list_objects = mixes_list.find_all('li')
        genre = genre_url.split(":")[-1]
        print(f"FOUND {len(list_objects)} MIXES FOR GENRE  '{genre}'")

        i = start_id
        for li_tag in list_objects:
            a_tag = li_tag.find('a')
            href_value = a_tag['href']
            mix_id = f'mix_{i}'
            link = f"{base_url}{href_value}"
            self[mix_id] = link
            i += 1

    def filter(self, genre, year, min_set_links):
        pass



class Tracklist(dict):
    """
    Takes a link to a specific set page
    and returns all YouTube links provided
    for that set.
    """
    def __init__(self, set_url):
        options = Options()
        options.add_argument('--headless')
        service = Service(BASE / 'archive' / 'chromedriver' / 'chromedriver')
        self.driver = webdriver.Chrome(options=options, service=service)
        self.driver.get(set_url)
        
        self.mix_tracklist = self.get_tracklist()
        self.driver.quit()

    def get_tracklist(self):
        try:
            mix = self.driver.title.split(" | ")[0]
            tracklist = self.driver.find_elements(By.XPATH, '//ol/li/span')
            youtube_ids = {}
            
            for track in tracklist:
                artist = track.get_attribute("data-keywordsartist")
                title = track.get_attribute("data-keywordstitle")
                youtube_id = track.get_attribute("data-youtubeid")
                youtube_ids[youtube_id] = [artist, title]
                print('{0: <13}'.format(youtube_id), ' - ', artist, ' - ', title)
            
            return {mix: youtube_ids}
        
        except:
            print('No YouTube IDs found on the page:', self.driver.current_url)
            return None


    def filter(self, min_tracks=5, max_tracks=20):
        pass


    def download(self, output_folder=y_folder):
        
        for keys, values in self.items():
            print(keys, values)
            
            for ytid in values.keys():
                
                if len(ytid) == 11:
                    download_track(ytid)



