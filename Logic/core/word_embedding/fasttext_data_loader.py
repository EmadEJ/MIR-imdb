import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import json

class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """
    def __init__(self, file_path):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path

    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres).
        """
        with open(self.file_path, "r") as FILE:
            data = json.load(FILE)
        df = pd.DataFrame(columns=['title', 'synopsis', 'summaries', 'reviews', 'genres'])
        for idx, movie in enumerate(data):
            df.loc[idx] = [movie['title'], ' '.join(movie['synopsis']), ' '.join(movie['summaries']), ' '.join([review[0] for review in movie['reviews']]), (movie['genres'][0] if len(movie['genres']) > 0 else '')]
        return df

    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        df = self.read_data_to_df()
        X = []
        y = []
        for _, row in df.iterrows():
            X.append(' '.join([row['title'], row['synopsis'], row['summaries'], row['reviews']]))
            y.append(row['genres'])
        
        return X, y


