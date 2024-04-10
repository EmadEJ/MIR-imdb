class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        stopwords = ['a', 'an', 'in', 'the', 'this', 'that', 'about', 'whom', 'being', 'where', 'why', 'had', 'should', 'each']

        return ' '.join([word for word in query.split() if word not in stopwords])

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        final_snippet = ""
        not_exist_words = []

        query = query.lower()
        query = self.remove_stop_words_from_query(query)
        query_words = query.split()
        doc = doc.lower()
        doc_words = doc.split()

        indices = []
        exacts = []

        for query_word in query_words:
            if query_word not in doc_words:
                not_exist_words.append(query_word)
                    
            else:
                idx = doc_words.index(query_word)
                indices.append(idx)
                exacts.append(idx)
                for i in range(1, self.number_of_words_on_each_side):
                    indices.append(idx + i)
                    indices.append(idx - i)

        indices = sorted(list(set(indices)))

        if len(indices) == 0:
            return "...", query_words

        if indices[0] > 0:
            final_snippet = final_snippet + "... "

        for idx, index in enumerate(indices):
            if index < 0 or index >= len(doc_words):
                continue
            if idx > 0 and index > indices[idx - 1] + 1:
                final_snippet = final_snippet + "... "
            if index in exacts:
                final_snippet = final_snippet + "***" + doc_words[index] + "*** "
            else:
                final_snippet = final_snippet + doc_words[index] + " "

        if indices[-1] < len(doc_words) - 1:
            final_snippet = final_snippet + "..."

        return final_snippet, not_exist_words

if __name__ == '__main__':    
    snip = Snippet(3)
    query = "friend notebook home happy"
    doc = "Eight-year-old Ahmed has mistakenly taken his friend Mohammad's notebook He wants to return it, or else his friend will be expelled from school. The boy determinedly sets out to find Mohammad's home in the neighbouring village."
    print(snip.find_snippet(doc, query)[0])
