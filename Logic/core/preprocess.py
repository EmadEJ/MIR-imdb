import re
import json
from nltk.stem import WordNetLemmatizer

class Preprocessor:

    def __init__(self, documents: list):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        self.documents = documents
        self.stopwords = ['a', 'an', 'the', 'this', 'that', 'about', 'whom', 'being', 'where', 'why', 'had', 'should', 'each']
        self.punctuations = [',', "\'", '"', '.', ';', ':', '!', '?', '#', '*', '(', ')', '[', ']']
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        str
            The preprocessed documents.
        """
        processed_docs = []
        for document in self.documents:
            # summaries
            new_summaries = []
            for text in document['summaries']:
                text = self.remove_links(text)
                text = self.remove_punctuations(text)
                text = self.normalize(text)
                words = self.tokenize(text)
                new_summaries.append(' '.join(words))
            document['summaries'] = new_summaries   

            # stars
            new_stars = []
            for text in document['stars']:
                text = text.lower()
                new_stars.append(text)
            document['stars'] = new_stars

            # genres
            new_genres = []
            for text in document['genres']:
                text = text.lower()
                new_genres.append(text)
            document['genres'] = new_genres

            processed_docs.append(document)

        return processed_docs

    def normalize(self, text: str):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """
        text = text.lower()
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
        for pattern in patterns:
            text = re.sub(pattern, '', text)

        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        for punctuation in self.punctuations:
            text = text.replace(punctuation, '')
        return text

    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        words = text.split()
        words = self.remove_stopwords(words)
        return words

    def remove_stopwords(self, words: list):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        return [word for word in words if word not in self.stopwords]


with open("IMDB_crawled.json", "r") as FILE:
    documents = json.load(FILE)
prep = Preprocessor(documents)
processed_docs = prep.preprocess()
print(processed_docs[0]['summaries'])
with open("preprocessed_docs.json", 'w') as FILE:
    json.dump(processed_docs, FILE)

#### lines below are for my testing

# example_text = """
#                 I don't know what I can say or what I should do, But I think I have to Fill this Shit out. 
#                 For example here's the site to 9th WSS, www.ce-wss.com I know it's dumb but Eh. 
#                 Let's see what Happens!
#                 happy, better, best, babies, stupids
#                 happens happenning happened
#                 """
# prep = Preprocessor([example_text])
# print(prep.remove_links('asASDdf www.sagtoosh.com sdfasdf Sag@Ah.edu adsfa Heeeeey.ir awsdas'))
# print(prep.remove_stopwords(['asd', 'srhsdfg', 'asdf as dffa', 'asdf', 'weqwe', 'this', 'that', 'yo', 'ummm', 'where', 'huh']))
# print(prep.preprocess())