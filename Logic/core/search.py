import json
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join('Logic', 'core')))
from .indexer.indexes_enum import Indexes, Index_types
from .indexer.index_reader import Index_reader
from utility import Preprocessor, Scorer


class SearchEngine:
    def __init__(self):
        """
        Initializes the search engine.

        """
        path = "index/"
        self.document_indexes = {
            Indexes.STARS: Index_reader(path, Indexes.STARS).index,
            Indexes.GENRES: Index_reader(path, Indexes.GENRES).index,
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES).index
        }
        self.tiered_index = {
            Indexes.STARS: Index_reader(path, Indexes.STARS, Index_types.TIERED).index,
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.TIERED).index,
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES, Index_types.TIERED).index
        }
        self.document_lengths_index = {
            Indexes.STARS: Index_reader(path, Indexes.STARS, Index_types.DOCUMENT_LENGTH).index,
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.DOCUMENT_LENGTH).index,
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES, Index_types.DOCUMENT_LENGTH).index
        }
        self.metadata_index = Index_reader(path, Indexes.DOCUMENTS, Index_types.METADATA).index

    def search(
        self,
        query,
        method,
        weights,
        safe_ranking=True,
        max_results=10,
        smoothing_method='Naive',
        alpha=0.5,
        lamda=0.5,
    ):
        """
        searches for the query in the indexes.

        Parameters
        ----------
        query : str
            The query to search for.
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25 | Unigram
            The method to use for searching.
        weights: dict
            The weights of the fields.
        safe_ranking : bool
            If True, the search engine will search in whole index and then rank the results.
            If False, the search engine will search in tiered index.
        max_results : int
            The maximum number of results to return. If None, all results are returned.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        list
            A list of tuples containing the document IDs and their scores sorted by their scores.
        """
        preprocessor = Preprocessor([query])
        query = preprocessor.preprocess()[0]
        query = query.split()

        scores = {}
        if method == "unigram":
            self.find_scores_with_unigram_model(
                query, smoothing_method, weights, scores, alpha, lamda
            )
        elif safe_ranking:
            self.find_scores_with_safe_ranking(query, method, weights, scores)
        else:
            self.find_scores_with_unsafe_ranking(
                query, method, weights, max_results, scores
            )

        final_scores = {}
    
        self.aggregate_scores(weights, scores, final_scores)

        result = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        if max_results is not None:
            result = result[:max_results]

        return result

    def aggregate_scores(self, weights, scores, final_scores):
        """
        Aggregates the scores of the fields.

        Parameters
        ----------
        weights : dict
            The weights of the fields.
        scores : dict
            The scores of the fields.
        final_scores : dict
            The final scores of the documents.
        """

        all_docs = []
        for field in scores.keys():
            all_docs.extend(scores[field].keys())
        all_docs = list(set(all_docs))
        # print(all_docs)

        # print(scores)
        for doc in all_docs:
            score = 0
            for field in weights:
                if scores[field].get(doc) is None:
                    continue
                score += weights[field] * scores[field][doc]
            final_scores[doc] = score

    def find_scores_with_unsafe_ranking(self, query, method, weights, max_results, scores):
        """
        Finds the scores of the documents using the unsafe ranking method using the tiered index.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        max_results : int
            The maximum number of results to return.
        scores : dict
            The scores of the documents.
        """
        for field in weights:
            scores[field] = {}
        for tier in ["first_tier", "second_tier", "third_tier"]:
            for idx, field in enumerate(weights.keys()):
                scorer = Scorer(self.tiered_index[self.document_indexes[list(self.document_indexes.keys())[idx]]][tier], self.metadata_index['document_count'])
                if method == 'OkapiBM25':
                    new_score = scorer.compute_socres_with_okapi_bm25(query, 
                                                                        self.metadata_index['average_document_length'][field.value],
                                                                        self.document_lengths_index[field])
                    scores[field] = self.merge_scores(scores[field], new_score)
                else:
                    new_score = scorer.compute_scores_with_vector_space_model(query, method)
                    scores[field] = self.merge_scores(scores[field], new_score)                

    def find_scores_with_safe_ranking(self, query, method, weights, scores):
        """
        Finds the scores of the documents using the safe ranking method.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        scores : dict
            The scores of the documents.
        """
        for idx, field in enumerate(weights.keys()):
            print(self.document_indexes.keys())
            print(field)
            print(field == Indexes.STARS)
            print(Indexes.STARS in self.document_indexes.keys())
            scorer = Scorer(self.document_indexes[list(self.document_indexes.keys())[idx]], self.metadata_index['document_count'])
            if method == 'OkapiBM25':
                scores[field] = scorer.compute_socres_with_okapi_bm25(query, 
                                                                      self.metadata_index['average_document_length'][field.value],
                                                                      self.document_lengths_index[field])
            else:
                scores[field] = scorer.compute_scores_with_vector_space_model(query, method)

    def find_scores_with_unigram_model(self, query, smoothing_method, weights, scores, alpha=0.5, lamda=0.5):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        weights : dict
            A dictionary mapping each field (e.g., 'stars', 'genres', 'summaries') to its weight in the final score. Fields with a weight of 0 are ignored.
        scores : dict
            The scores of the documents.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.
        """

        for idx, field in enumerate(weights.keys()):
            document_lengths = {}
            the_index = self.document_indexes[list(self.document_indexes.keys())[idx]]
            for _, postings in the_index.items():
                for doc_id, cnt in postings.items():
                    if document_lengths.get(doc_id) is None:
                        document_lengths[doc_id] = 0
                    document_lengths[doc_id] += cnt
            scorer = Scorer(the_index, self.metadata_index['document_count'])
            scores[field] = scorer.compute_scores_with_unigram_model(query, smoothing_method, document_lengths, alpha, lamda)

    def merge_scores(self, scores1, scores2):
        """
        Merges two dictionaries of scores.

        Parameters
        ----------
        scores1 : dict
            The first dictionary of scores.
        scores2 : dict
            The second dictionary of scores.

        Returns
        -------
        dict
            The merged dictionary of scores.
        """

        final_scores = {}
        for doc in scores1:
            if final_scores.get(doc) is None:
                final_scores[doc] = 0
            final_scores[doc] += scores1[doc]
        for doc in scores2:
            if final_scores.get(doc) is None:
                final_scores[doc] = 0
            final_scores[doc] += scores2[doc]
        return final_scores


if __name__ == "__main__":
    search_engine = SearchEngine()
    query = "spider man in wonderland"
    method = "unigram"
    smoothing_method = 'mixture'
    weights = {
        Indexes.STARS: 1,
        Indexes.GENRES: 1,
        Indexes.SUMMARIES: 1
    }
    result = search_engine.search(query, method, weights, safe_ranking=False, smoothing_method=smoothing_method)

    print(result)

    with open('IMDB_Crawled.json') as FILE:
        data = json.load(FILE)
    for res in result:
        for movie in data:
            if movie['id'] == res[0]:
                print(movie['title'])
                break
