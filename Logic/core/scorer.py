import numpy as np
from indexer.index_reader import Index_reader
from indexer.indexes_enum import Indexes,Index_types

class Scorer:    
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents

    def get_list_of_documents(self,query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.
        
        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.
            
        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))
    
    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.
        
        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        idf = self.idf.get(term, None)
        if idf is None:
            if self.index.get(term) is None:
                self.idf[term] = 0    
            else:
                self.idf[term] = np.log(self.N / len(self.index[term]))
        return self.idf[term]
    
    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """
        
        cur_dict = {}
        for word in query:
            if cur_dict.get(word) is None:
                cur_dict[word] = 0
            cur_dict[word] += 1
    
        return cur_dict


    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        query = query.split()

        docs = self.get_list_of_documents(query)
        query_tfs = self.get_query_tfs(query)
        
        doc_scores = {}
        for doc_id in docs:
            doc_scores[doc_id] = self.get_vector_space_model_score(query, query_tfs, doc_id, method[0:3], method[4:7])

        return doc_scores

    def get_vector_space_model_score(self, query, query_tfs, document_id, document_method, query_method):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """

        # print(document_id, "####################################")

        doc_tfs = {}
        for term in self.index.keys():
            if self.index[term].get(document_id) is not None:
                doc_tfs[term] = self.index[term][document_id]

        doc_ws = {}
        for term in doc_tfs.keys():
            tf = doc_tfs[term]
            if document_method[0] == 'l':
                tf = 1 + np.log(tf)
            idf = 1
            if document_method[1] == 't':
                idf = self.get_idf(term)
            w = tf * idf
            doc_ws[term] = w

        doc_normalization_factor = 1
        if document_method[2] == 'c':
            doc_normalization_factor = np.sqrt(sum(w * w for w in doc_ws.values()))
            for term in doc_ws.keys():
                doc_ws[term] *= doc_normalization_factor

        # print(doc_ws)

        query_ws = {}
        for term in query:
            tf = query_tfs[term]
            if query_method[0] == 'l':
                tf = 1 + np.log(tf)
            idf = 1
            if query_method[1] == 't':
                idf = self.get_idf(term)
            w = tf * idf
            query_ws[term] = w

        query_normalization_factor = 1
        if query_method[2] == 'c':
            query_normalization_factor = np.sqrt(sum(w * w for w in query_ws.values()))
            for term in query_ws.keys():
                query_ws[term] *= query_normalization_factor

        # print(query_ws)

        score = 0
        for term in query:
            if doc_ws.get(term) is None:
                continue
            score += query_ws[term] * doc_ws[term]
        
        return score

    def compute_socres_with_okapi_bm25(self, query, average_document_field_length, document_lengths):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        
        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        query = query.split()

        docs = self.get_list_of_documents(query)
        
        doc_scores = {}
        for doc_id in docs:
            doc_scores[doc_id] = self.get_okapi_bm25_score(query, doc_id, average_document_field_length, document_lengths)

        return doc_scores        

    def get_okapi_bm25_score(self, query, document_id, average_document_field_length, document_lengths):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """
        k1 = 1.6
        b = 0.75

        rsv = 0
        for term in query:
            idf = self.get_idf(term) 
            tf = 0
            if self.index[term].get(document_id) is not None:
                tf = self.index[term][document_id]
            dl = document_lengths[document_id]
            rsv += (idf * (k1 + 1) * tf) / (k1 * ((1 - b) + b * (dl / average_document_field_length)) + tf)

        return rsv

index = Index_reader(path='index/', index_name=Indexes.SUMMARIES).get_index()
length_index = Index_reader(path='index/', index_name=Indexes.SUMMARIES, index_type=Index_types.DOCUMENT_LENGTH).get_index()
metadata_index = Index_reader(path='index/', index_name=Indexes.DOCUMENTS, index_type=Index_types.METADATA).get_index()

sc = Scorer(index, 2000)
query = "happy family farm"

scores = sc.compute_scores_with_vector_space_model(query, 'ltc.lnc')
print(dict(sorted(scores.items(), key=lambda item: -item[1])))

scores = sc.compute_socres_with_okapi_bm25(query, metadata_index['average_document_length']['summaries'], length_index)
print(dict(sorted(scores.items(), key=lambda item: -item[1])))