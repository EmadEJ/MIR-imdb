import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from .data_loader import ReviewLoader
from .basic_classifier import BasicClassifier


class ReviewDataSet(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(np.array(embeddings))
        self.labels = torch.LongTensor(np.array(labels))

        if len(self.embeddings) != len(self.labels):
            raise Exception("Embddings and Labels must have the same length")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.embeddings[i], self.labels[i]


class MLPModel(nn.Module):
    def __init__(self, in_features=100, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, xb):
        return self.network(xb)


class DeepModelClassifier(BasicClassifier):
    def __init__(self, in_features, num_classes, batch_size, num_epochs=50):
        """
        Initialize the model with the given in_features and num_classes
        Parameters
        ----------
        in_features: int
            The number of input features
        num_classes: int
            The number of classes
        batch_size: int
            The batch size of dataloader
        """
        super().__init__()
        self.test_loader = None
        self.in_features = in_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = MLPModel(in_features=in_features, num_classes=num_classes)
        self.best_model = self.model.state_dict()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.device = 'cuda' if torch.cuda.is_available() else self.device
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def fit(self, x, y):
        """
        Fit the model on the given train_loader and test_loader for num_epochs epochs.
        You have to call set_test_dataloader before calling the fit function.
        Parameters
        ----------
        x: np.ndarray
            The training embeddings
        y: np.ndarray
            The training labels
        Returns
        -------
        self
        """
        dataset = ReviewDataSet(x, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for epoch in range(self.num_epochs):
            self.model.train()
            avg_loss = 0

            for i, data in enumerate(dataloader, 0):
                X, y = data

                self.optimizer.zero_grad()

                pred = self.model(X)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()

                avg_loss += loss.item()
                if i % 2000 == 1999:
                    print(f'[{epoch}, {(i + 1):5d}] loss: {avg_loss / 2000:.3f}')
                    avg_loss = 0
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {avg_loss / len(dataloader)}")
                
        return self

    def predict(self, x):
        """
        Predict the labels on the given test_loader
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        Returns
        -------
        predicted_labels: list
            The predicted labels
        """
        self.model.eval()
        dataset = ReviewDataSet(x, np.zeros(len(x)))
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        with torch.no_grad():
            preds = []
            for X, _ in dataloader:
                pred = self.model(X)
                _, pred = torch.max(pred, 1)
                preds.extend(pred.cpu().numpy())

        return np.array(preds)
    def _eval_epoch(self, dataloader: torch.utils.data.DataLoader, model):
        """
        Evaluate the model on the given dataloader. used for validation and test
        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
        Returns
        -------
        eval_loss: float
            The loss on the given dataloader
        predicted_labels: list
            The predicted labels
        true_labels: list
            The true labels
        f1_score_macro: float
            The f1 score on the given dataloader
        """
        self.model.eval()
        preds = []
        true_y = []
        avg_loss = 0

        with torch.no_grad():
            for X, y in dataloader:

                pred = self.model(X)
                _, pred = torch.max(pred, 1)
                loss = self.criterion(pred, y)
                avg_loss += loss.item()

                preds.extend(pred.cpu().numpy())
                true_y.extend(y.cpu().numpy())

        f1 = f1_score(true_y, preds)
        return avg_loss / len(dataloader), preds, true_y, f1


    def set_test_dataloader(self, X_test, y_test):
        """
        Set the test dataloader. This is used to evaluate the model on the test set while training
        Parameters
        ----------
        X_test: np.ndarray
            The test embeddings
        y_test: np.ndarray
            The test labels
        Returns
        -------
        self
            Returns self
        """
        test_dataset = ReviewDataSet(X_test, y_test)
        self.test_loader = Dataset(test_dataset)
        return self

    def prediction_report(self, x, y):
        """
        Get the classification report on the given test set
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        y: np.ndarray
            The test labels
        Returns
        -------
        str
            The classification report
        """
        pred = self.predict(x)
        return classification_report(y, pred)

# F1 Accuracy : 79%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    loader = ReviewLoader('IMDB Dataset.csv')
    loader.load_data()
    loader.get_embeddings()
    X_train, X_test, y_train, y_test = loader.split_data()
    for i in range(len(y_train)):
        y_train[i] = (y_train[i] + 1) / 2
    for i in range(len(y_test)):
        y_test[i] = (y_test[i] + 1) / 2
    
    model = DeepModelClassifier(len(X_train[0]), 2, 128, num_epochs=20)
    model.fit(X_train, y_train) 
    print(model.prediction_report(X_test, y_test)) 

# Epoch 1/20, Loss: 0.5451983192477363
# Epoch 2/20, Loss: 0.5052756700462426
# Epoch 3/20, Loss: 0.4929236109835652
# Epoch 4/20, Loss: 0.4882321340588335
# Epoch 5/20, Loss: 0.4864682413328189
# Epoch 6/20, Loss: 0.4869736005513432
# Epoch 7/20, Loss: 0.48272663897599655
# Epoch 8/20, Loss: 0.48215121030807495
# Epoch 9/20, Loss: 0.4823711341181502
# Epoch 10/20, Loss: 0.48073890167303357
# Epoch 11/20, Loss: 0.4811637126409208
# Epoch 12/20, Loss: 0.48306315956405177
# Epoch 13/20, Loss: 0.4785977539163047
# Epoch 14/20, Loss: 0.47775266526606136
# Epoch 15/20, Loss: 0.4841759774250725
# Epoch 16/20, Loss: 0.4809332119580656
# Epoch 17/20, Loss: 0.4780088289858053
# Epoch 18/20, Loss: 0.47525904648981915
# Epoch 19/20, Loss: 0.4796030829889706
# Epoch 20/20, Loss: 0.47506510611540215
#               precision    recall  f1-score   support

#          0.0       0.80      0.88      0.84      5018
#          1.0       0.87      0.78      0.82      4982

#     accuracy                           0.83     10000
#    macro avg       0.83      0.83      0.83     10000
# weighted avg       0.83      0.83      0.83     10000