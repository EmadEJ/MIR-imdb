import json
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizerFast, BertTokenizer 
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from huggingface_hub import HfFolder, Repository

class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        self.file_path = file_path
        self.top_n_genres = top_n_genres

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        with open(self.file_path, "r") as FILE:
            data = json.load(FILE)
        
        summaries = [x['first_page_summary'] for x in data]
        genres = [x['genres'][0] if len(x['genres']) > 0 else np.nan for x in data]
        
        self.dataset = pd.DataFrame({'summary': summaries, 'genre': genres})

    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres
        """
        self.dataset.dropna(inplace=True)
        
        cnt = self.dataset['genre'].value_counts()
        self.top_genres = list(cnt.index[:self.top_n_genres])

        mask = self.dataset['genre'].isin(self.top_genres)
        self.dataset = self.dataset[mask]
        
        for _, row in self.dataset.iterrows():
            row['genre'] = self.top_genres.index(row['genre'])

    def split_dataset(self, test_size=0.3, val_size=0.5):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        self.train_dataset, self.test_dataset = train_test_split(self.dataset, test_size=test_size)
        self.train_dataset, self.valid_dataset = train_test_split(self.train_dataset, test_size=val_size)

    def create_dataset(self, texts, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=200)
#         print(encodings.keys())
#         for umm in encodings['attention_mask']:
#             print(len(umm))
        return IMDbDataset(encodings, list(labels))

    def fine_tune_bert(self, epochs=5, batch_size=16, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        train_data = self.create_dataset(self.train_dataset['summary'], self.train_dataset['genre'])
        train_dataloader= DataLoader(dataset=train_data, batch_size=batch_size, sampler=RandomSampler(train_data))
        eval_data = self.create_dataset(self.valid_dataset['summary'], self.valid_dataset['genre'])
        eval_dataloader= DataLoader(dataset=eval_data, batch_size=batch_size)

        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=self.top_n_genres)

        optimizer = AdamW(self.model.parameters(), lr=1e-4, eps=1e-9, weight_decay=weight_decay)
        
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=(epochs * len(train_dataloader)))
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for batch in tqdm(train_dataloader):
                # print(batch)
                batch = {k: v.to(device) for k, v in batch.items()}
                output = self.model(**batch)
                loss = output.loss
                total_loss += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Average training loss in epoch {epoch+1}/{epochs}: {avg_loss}")
        
        self.model.save_pretrained("fine_tuned_bert")
        print("I'm Done!!!?")

    def compute_metrics(self, preds, trues):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        precision, recall, f1, support = precision_recall_fscore_support(trues, preds, average='macro')
        acc = accuracy_score(trues, preds)
        return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall, "support": support}

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        test_data = self.create_dataset(self.test_dataset['summary'], self.test_dataset['genre'])        
        test_dataloader = DataLoader(test_data, batch_size=16)
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        
        self.model.eval()
        
        preds, trues = [], []
        for batch in tqdm(test_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                output = self.model(**batch)
            
            logits = output.logits
            labels = batch["labels"]
            pred = torch.argmax(logits, dim=1).flatten().detach().cpu().numpy()
            labels = labels.flatten().cpu().numpy()
            preds.extend(pred)
            trues.extend(labels)
        
        metrics = self.compute_metrics(preds, trues)
        print("evaluation result:\n", metrics)
            
    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        token = HfFolder.get_token()
        if token is None:
            raise ValueError("You must be logged into the Hugging Face Hub. Use `huggingface-cli login`.")

        repo = Repository(local_dir=model_name, clone_from="EmadEJ/MIR-Bert", use_auth_token=True, git_email="s.emad.emamjomeh@gmail.com")

        self.model.save_pretrained(model_name)

        repo.push_to_hub(commit_message="Update model")  

class IMDbDataset(Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """
        item = {key: torch.tensor(val[idx], dtype=torch.long) for key, val in self.encodings.items()}
#         print(self.labels[idx])
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        
#         print(item)
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)