import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from .basic_classifier import BasicClassifier
from .data_loader import ReviewLoader


class SVMClassifier(BasicClassifier):
    def __init__(self):
        super().__init__()
        self.model = SVC(verbose=1)

    def fit(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc
        """
        self.model.fit(x, y)

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
        return self.model.predict(x)

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
        pred = self.predict(x)
        return classification_report(y, pred)


# F1 accuracy : 78%
# my F1 Score : 83%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    loader = ReviewLoader('IMDB Dataset.csv')
    loader.load_data()
    loader.get_embeddings()
    X_train, X_test, y_train, y_test = loader.split_data()
    svm = SVMClassifier()
    svm.fit(X_train, y_train)
    print(svm.prediction_report(X_test, y_test))