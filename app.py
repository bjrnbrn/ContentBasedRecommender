import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
base = Path('__file__').resolve().parent

from utils.mixscrap import Tracklist, GenreSets, MixesDbGenres
from utils.eda import Mixes, Tracks, load_train_data, load_test_data
from utils.models import CNNModel

# Global vars
json_file = base / 'data' / 'djmix-dataset.json'
pickle_path = base / 'data' / 'pkls'
tracks_path = base / 'data' / 'tracks'




# Define markdown summaries for each section
home_summary = """
#### Project Goal:

*Build a Model to categorize Songs according to some Similarity Features.*

#### Initial Approach:

*Find out which Songs fit to each other by their appearance in professional DJ Sets.*
"""

section_a_summary = """
#### Dataset:

*Making use of the MixesDb Website which archives DJ Mixes (e.g. from Soundcloud) and links to Youtube Videos of songs of the Mix if available.*

#### Process:

***First:** Selenium scraped all Genres of MixesDb*

***Then:** Fetch various DJ Mixes for each Genre*

***Finally:** Obtain all available Youtube Links for each Set*

"""

section_b_summary = """
#### Summary of the Dataset:

*After scraping MixesDB, the Data was then cleaned and the Audio was acquired.*

*The available Audio Files were transformed into Mel-Spectrograms with the Librosa python package for the CNN Model analysis.*

"""

section_c_summary = """
I used a Convolutional Neural Network for Song Categorization by either:

- Occurence in a Mix
- Genre of the Mix

*The Model takes as Input all the sorted Spectrograms of the audio files from MixesDB*

*The Model then outputs Predictions to categorize my own Song Library.*
"""

conclusion_summary = """
*Predictions are not valid yet due to the uneven Distribution of the available Genres*

**Next Steps:** 

- Scrape MixesDB again for a more evenly distributed Genre base.
- Extract more relevant features except the Spectrograms for further Similarity Analyis.
- Build another, more customized Model to the task?
- Make Predictions on the basis of a songs occurence in a DJ Mix.


"""




def home():
    st.markdown(home_summary)


def find_audio_file(file_name, audio_file_path):
    # Check if the file with the same name (regardless of extension) exists in the folder
    file_base_name = os.path.splitext(file_name)[0]
    for file in os.listdir(audio_file_path):
        if os.path.splitext(file)[0] == file_base_name:
            audio_file_path = os.path.join(audio_file_path, file)
            return audio_file_path
    return None


def genre_summary(mixes):
    genre_counts = mixes['genre'].value_counts().sort_values(ascending=False)
    length = 10
    genre_summary = genre_counts.head(length).append(genre_counts.tail(length))
    summary_text = '-------------------------------------------------------------------\n'
    for i, (genre, count) in enumerate(genre_summary.items()):
        files_count = mixes.loc[mixes['genre'] == genre, 'files_count'].sum()
        summary_text += f'total of {count} mix(es), {files_count} file(s) in {genre.upper()}\n'
        if i == length - 1:  # Last item of the head
            summary_text += '...\n'
    summary_text += '-------------------------------------------------------------------\n\n'
    return summary_text


def get_the_data(min_files, max_files):
    mixes = Mixes(
        json_file,
        min_files,
        max_files,
        tracks_path
    )
    tracks = Tracks(mixes)
    return mixes, tracks


def train_and_predict_data(tracks, feature_type, label_type):
    X, y, NUM_CLASSES = load_train_data(
        tracks,
        feature_type,
        label_type
    )
    cnn_model = CNNModel(num_classes=NUM_CLASSES)
    cnn_model.train(X, y)

    X_TEST, NAMES = load_test_data(feature_type)
    predictions = cnn_model.predict(X_TEST)
    return cnn_model, NAMES, predictions


def section_a():
    st.markdown('### A. WEB SCRAPING FOR MIXDB DATA')
    st.markdown(section_a_summary)
    # List all Genres and Subgenres of the MixesDb Website
    genres = MixesDbGenres()

    # Split the genres into multiple columns
    num_columns = 3  # Number of columns to display the genres
    genre_columns = st.columns(num_columns)

    for i, genre in enumerate(genres):
        # Get the genre name, link, and type
        genre_name = genre
        link = genres[genre][0]
        genre_type = genres[genre][1]

        # Generate the link with the desired style
        link_html = f'<a href="{link}" style="color: #FF4500; text-decoration: underline;" target="_blank">{genre_name}</a>'

        # Determine the column to place the link based on index
        column_index = i % num_columns

        # Display the link in the corresponding column
        with genre_columns[column_index]:
            if genre_type == 'genre':
                st.markdown(link_html, unsafe_allow_html=True)
            else:
                st.markdown(f'- {link_html}', unsafe_allow_html=True)


def section_b(mixes, tracks):
    st.markdown('### B. EXTRACT - TRANSFORM - LOAD')
    st.markdown(section_b_summary)
    # Display the summary of the mixes dataframe
    st.text('\n')
    st.text('\n')
    st.markdown('#### Mixes Dataframe Summary:')
    
    mixes_summary = mixes.__str__().title()
    st.text('-------------------------------------------------------------------\n')
    st.text(mixes_summary)
    st.text('-------------------------------------------------------------------\n')
    st.text('\n')
    st.text('\n')
    # Display the genre summary of the mixes dataframe
    st.markdown('#### Mixes Genre Summary:')
    genre_summary_text = genre_summary(mixes)
    st.text(genre_summary_text)


def section_c(feature_type, label_type, tracks):
    st.markdown('### CNN MODEL & PREDICTIONS')
    st.markdown(section_c_summary)

    with st.spinner("Training the model..."):
        _ , NAMES, predictions = train_and_predict_data(tracks, feature_type, label_type)
    st.success("Model trained successfully!")
    
    # List the Predictions
    for i, prediction in enumerate(predictions):
        st.write(NAMES[i], f'<span style="color: #FF4500;">{prediction}</span>', unsafe_allow_html=True)




def conclusion(mixes):
    st.markdown('### CONCLUSION')
    st.markdown(conclusion_summary)
    
    genre_counts = mixes['genre'].value_counts().sort_values(ascending=False)
    genres = genre_counts.index
    mix_counts = genre_counts.values
    files_counts = [mixes.loc[mixes['genre'] == genre, 'files_count'].sum() for genre in genres]

    # Sort genres and counts in descending order
    genres = genres[::-1]
    mix_counts = mix_counts[::-1]
    files_counts = files_counts[::-1]

    # Set dark theme style
    plt.style.use('dark_background')

    # Plotting number of mixes
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    ax1.barh(genres, mix_counts, color='#FF4500', edgecolor='black')
    ax1.set_xlabel('Number of Mixes')
    ax1.set_title('Number of Mixes per Genre', fontsize=16)

    # Plotting number of tracks
    fig2, ax2 = plt.subplots(figsize=(10, 10))
    ax2.barh(genres, files_counts, color='#FF4500', edgecolor='black')
    ax2.set_xlabel('Number of Tracks')
    ax2.set_title('Number of Tracks per Genre', fontsize=16)

    # Display the plots
    st.pyplot(fig1)
    st.pyplot(fig2)


def playit():
    # Spectrogram File Selection
    file_path = base / 'data' / 'spectrograms' / 'test'
    files = os.listdir(file_path)
    files.insert(0, '')  # Insert an empty value at the beginning
    file_name = st.selectbox("Select a Song", files)
    
    if file_name:
        image = Image.open(file_path / file_name)
        st.image(image)

        # Audio Files
        audio_file_path = base / 'audio'
        audio_file = find_audio_file(file_name, audio_file_path)
        if audio_file:
            st.audio(audio_file, format='audio/mp3')
        else:
            st.warning('Audio file not found.')







def main():
    st.title("A Content based Approach to Music (Playlist) Recommendations")
    
    
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Overview", "Web Scraping", "Data", "Model Predictions", "Conclusion", "Play"])
    
    
    # User input section
    feature_type = st.sidebar.selectbox(
        "Type of feature to extract",
        ["Spectrograms", "Chromagrams"]).lower()
    
    label_type = st.sidebar.selectbox(
        "Type of label to predict",
        ["Genre", "Mix"]).lower()
    
    min_files = st.sidebar.slider(
        "Minimum Files", 1, 20, 5)
    
    max_files = st.sidebar.slider(
        "Maximum Files", min_files, 50, 20)
    

    # Get the Mixes and Tracks Dataframes
    mixes, tracks = get_the_data(min_files, max_files)


    # Page Selections
    if selection == "Overview":
        home()
    elif selection == "Web Scraping":
        section_a()
    elif selection == "Data":
        section_b(mixes, tracks)
    elif selection == "Model Predictions":
        section_c(feature_type, label_type, tracks)
    elif selection == "Conclusion":
        conclusion(mixes)
    elif selection == "Play":
        playit()





if __name__ == '__main__':
    main()
