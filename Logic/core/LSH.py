import json
import numpy as np
import itertools
import random


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        # preprocess
        document = document.lower()
        punctuations = [',', "\'", '"', '.', ';', ':', '!', '?', '#', '*', '(', ')', '[', ']']
        for punctuation in punctuations:
            document = document.replace(punctuation, '')
        
        shingles = []
        words = document.split()
        for i in range(len(words) - k + 1):
            shingle = ()
            for j in range(k):
                shingle = shingle + (words[i + j],)
            shingles.append(shingle)
        shingles = set(shingles)

        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        all_shingles = set()
        doc_shingles = []
        for doc in self.documents:
            doc_shingle = self.shingle_document(doc)
            doc_shingles.append(doc_shingle)
            all_shingles = all_shingles.union(doc_shingle)

        matrix = np.zeros((len(doc_shingles), len(all_shingles)))
        for i, doc_shingle in enumerate(doc_shingles):
            for j, shingle in enumerate(all_shingles):
                if shingle in doc_shingle:
                    matrix[i][j] = 1

        return matrix

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        def hash(string, seed):
            hash_value = seed
            for char in string:
                hash_value = (ord(char) + (hash_value << 6) + (hash_value << 16) - hash_value) % (1<<32)
            return hash_value
        
        seeds = [random.randint(0, (1<<32)) for _ in range(self.num_hashes)]

        matrix = np.full((len(self.documents), self.num_hashes), np.inf)

        for i, doc in enumerate(self.documents):
            doc_shingles = self.shingle_document(doc)
            for j, seed in enumerate(seeds):
                for shingle in doc_shingles:
                    if hash(' '.join(shingle), seed) < matrix[i][j]:
                        matrix[i][j] = hash(' '.join(shingle), seed)

        return matrix

    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        buckets = {}
        for i in range(len(signature)):
            for b in range(bands):
                hashed_band = hash(tuple(signature[i][b*rows_per_band:(b+1)*rows_per_band]))
                if buckets.get(hashed_band) is None:
                    buckets[hashed_band] = []
                buckets[hashed_band].append(i)
        return buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        return lsh.lsh_buckets(lsh.min_hash_signature(), 25, 4)

    def jaccard_score(self, first_set: set, second_set: set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        return len(first_set.intersection(second_set)) / len(first_set.union(second_set))

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        all_combs = []

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]
                    all_combs.append((first_doc_id, second_doc_id))

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print(f"found {len(all_combs)} similar pairs")
        print("detected pairs:", set(all_combs))

        print(correct_near_duplicates, all_near_duplicates)
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)

with open("Logic/core/LSHFakeData.json", 'r') as FILE:
    fake_documents = json.load(FILE)

with open("IMDB_crawled.json", "r") as FILE:
    all_documents = json.load(FILE)

all_documents.extend(fake_documents)
doc_strings = [' '.join(doc['summaries']) for doc in all_documents if len(doc['summaries']) > 0]

lsh = MinHashLSH(doc_strings, 100)
lsh.jaccard_similarity_test(lsh.perform_lsh(), doc_strings)

# print(lsh.lsh_buckets(lsh.min_hash_signature(), 5, 2))