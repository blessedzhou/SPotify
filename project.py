import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import nltk
from nltk.stem.porter import PorterStemmer

# Download the punkt tokenizer for nltk
nltk.download('punkt')

# Initialize the PorterStemmer
stemmer = PorterStemmer()

# Set up Spotify API credentials
SPOTIPY_CLIENT_ID = 'b9e325a9a66a48659521be74222dff28'
SPOTIPY_CLIENT_SECRET = '6ed5906966e8454086c5b54043e55ba9'

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("spotify_millsongdata.csv")
    listen_counts = pd.read_csv("test.csv")
    data = data.sample(10000).reset_index(drop=True)
    data.drop('link', axis=1, inplace=True)
    data['text'] = data['text'].str.lower().replace(r'\n', ' ', regex=True)
    data['text'] = data['text'].apply(tokenization)
    song_id = listen_counts.sample(n=57650).reset_index(drop=True)
    data['userId'] = song_id['userId']
    data['songId'] = song_id['songId']
    data['count'] = song_id['count']
    return data

def tokenization(txt):
    tokens = nltk.word_tokenize(txt)
    stemming = [stemmer.stem(w) for w in tokens]
    return " ".join(stemming)

data = load_data()
matrix = data.pivot_table(index='songId', columns='userId', values='count').fillna(0)
matrix_norm = matrix.subtract(matrix.mean(axis=1), axis=0)

# Prepare song feature matrix and decode map
df_songs_features = csr_matrix(matrix_norm.values)
df_unique_songs = data.drop_duplicates(subset=['songId']).reset_index(drop=True)[['songId', 'song']]
decode_id_song = {song: i for i, song in enumerate(list(df_unique_songs.set_index('songId').loc[matrix.index].song))}

# Recommender class
class Recommender:
    def __init__(self, metric, algorithm, k, data, decode_id_song):
        self.metric = metric
        self.algorithm = algorithm
        self.k = k
        self.data = data
        self.decode_id_song = decode_id_song
        self.model = self._recommender().fit(data)

    def make_recommendation(self, new_song, n_recommendations):
        recommended = self._recommend(new_song=new_song, n_recommendations=n_recommendations)
        return recommended

    def _recommender(self):
        return NearestNeighbors(metric=self.metric, algorithm=self.algorithm, n_neighbors=self.k, n_jobs=-1)

    def _recommend(self, new_song, n_recommendations):
        recommendations = []
        recommendation_ids = self._get_recommendations(new_song=new_song, n_recommendations=n_recommendations)
        recommendations_map = self._map_indeces_to_song_title(recommendation_ids)
        for i, (idx, dist) in enumerate(recommendation_ids):
            recommendations.append(recommendations_map[idx])
        return recommendations

    def _get_recommendations(self, new_song, n_recommendations):
        recom_song_id = self._fuzzy_matching(song=new_song)
        distances, indices = self.model.kneighbors(self.data[recom_song_id], n_neighbors=n_recommendations+1)
        return sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]

    def _map_indeces_to_song_title(self, recommendation_ids):
        return {song_id: song_title for song_title, song_id in self.decode_id_song.items()}

    def _fuzzy_matching(self, song):
        match_tuple = []
        for title, idx in self.decode_id_song.items():
            ratio = fuzz.ratio(title.lower(), song.lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            st.write(f"The recommendation system could not find a match for {song}")
            return
        return match_tuple[0][1]

model = Recommender(metric='cosine', algorithm='brute', k=20, data=df_songs_features, decode_id_song=decode_id_song)

# Streamlit App
st.title("Spotify Song Recommender")

song = st.text_input("Enter a song title:")
n_recommendations = st.slider("Number of recommendations:", 1, 10, 5)

if st.button("Recommend"):
    new_recommendations = model.make_recommendation(new_song=song, n_recommendations=n_recommendations)
    st.write(f"The recommendations for '{song}' are:")
    for recommendation in new_recommendations:
        st.write(recommendation)
        result = sp.search(q=recommendation, type='track', limit=1)
        if result['tracks']['items']:
            track = result['tracks']['items'][0]
            st.write(f"**Track**: {track['name']}")
            st.write(f"**Artist**: {track['artists'][0]['name']}")
            st.write(f"[Open in Spotify]({track['external_urls']['spotify']})")
            st.image(track['album']['images'][0]['url'])
