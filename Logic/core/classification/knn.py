import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

from .basic_classifier import BasicClassifier
from .data_loader import ReviewLoader


class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        super().__init__()
        self.k = n_neighbors

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        self
            Returns self as a classifier
        """
        self.points = x
        self.labels = y
        return self

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        preds = []
        for doc in tqdm(x, "knn"):
            distances = np.array([np.linalg.norm(doc - point) for point in self.points])
            indexes = np.argsort(distances)
            cnt = 0
            for idx in indexes[:self.k]:
                if self.labels[idx] == 1:
                    cnt += 1
            if cnt > self.k / 2:
                preds.append(1)
            else:
                preds.append(-1)
        return np.array(preds)

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        preds = self.predict(x)
        return classification_report(y, preds)


# F1 Accuracy : 70%
# my F1 Score: 76%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    loader = ReviewLoader('IMDB Dataset.csv')
    loader.load_data()
    loader.get_embeddings()
    X_train, X_test, y_train, y_test = loader.split_data()
    knn = KnnClassifier(n_neighbors=11)
    knn.fit(X_train, y_train)
    print(knn.prediction_report(X_test, y_test))
