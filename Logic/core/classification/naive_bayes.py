import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from .basic_classifier import BasicClassifier
from .data_loader import ReviewLoader


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        super().__init__()
        self.cv = count_vectorizer
        self.num_classes = 2
        self.classes = [1, -1]
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = []
        self.feature_probabilities = None
        self.log_probs = []
        self.alpha = alpha

    def fit(self, X, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

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
        X = np.array(X)
        y = np.array(y)
        X = self.cv.fit_transform(X)
        self.number_of_samples = X.shape[0]
        self.number_of_features = X.shape[1]
        for c in self.classes:
            c_X = X[y == c]
            # prior
            self.prior.append(c_X.shape[0] / X.shape[0])
            # feature
            count = np.sum(c_X, axis=0) + self.alpha
            total_sum = np.sum(count, axis=1)
            count = count.flatten()
            probs = np.divide(count, total_sum)
            log_probs = np.log(probs)
            self.log_probs.append(log_probs)

    def predict(self, X):
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
        X = np.array(X)
        X = self.cv.transform(X)
        scores = []
        for i in range(self.num_classes):
            score = X.dot(self.log_probs[i].T)
            score = score + np.log(self.prior[i])
            scores.append(score)
        
        pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            if scores[0][i] > scores[1][i]:
                pred[i] = self.classes[0]
            else:
                pred[i] = self.classes[1]

        return pred

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

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        pred = self.predict(sentences)
        pos_cnt = len([x for x in range(pred) if x == 1])
        return pos_cnt / len(pred)

# F1 Accuracy : 85%
# my F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the reviews using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    loader = ReviewLoader('IMDB Dataset.csv')
    loader.load_data()
    X_train, X_test, y_train, y_test = loader.split_data()
    vectorizer = CountVectorizer()
    nb = NaiveBayes(vectorizer)
    nb.fit(X_train, y_train)
    print(nb.prediction_report(X_test, y_test))
