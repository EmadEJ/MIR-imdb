import json


class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set()
        
        if len(word) < k:
            shingles.add(word)

        for i in range(len(word) - k + 1):
            shingle = word[i:i+k]
            shingles.add(shingle)

        return shingles
    
    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """
        return len(first_set.intersection(second_set)) / len(first_set.union(second_set))

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = dict()

        for doc in all_documents:
            doc = doc.lower()
            words = doc.split()
            for word in words:
                if word_counter.get(word) is None:
                    word_counter[word] = 0
                word_counter[word] += 1
        
        for word in word_counter.keys():
            all_shingled_words[word] = self.shingle_word(word)
                
        return all_shingled_words, word_counter
    
    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        top5_candidates = list()
        word_shingle = self.shingle_word(word.lower())

        for candidate, candidate_shingled in self.all_shingled_words.items():
            if candidate == word:
                return [word, word, word, word, word]
            if len(top5_candidates) < 5:
                top5_candidates.append(candidate)
            else:
                for top in top5_candidates:
                    if self.jaccard_score(word_shingle, candidate_shingled) > self.jaccard_score(word_shingle, self.shingle_word(top)):
                        top5_candidates.remove(top)
                        top5_candidates.append(candidate)
                        break
        
        return top5_candidates
    
    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : stf
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        final_result = ""

        for word in query.split():
            candidates = self.find_nearest_words(word)
            # print(candidates)
            word = word.lower()
            top, top_score = '', 0
            for candidate in candidates:
                if candidate == word:
                    top = candidate
                    top_score = 1
                    break
                jaccard_score = self.jaccard_score(self.shingle_word(word), self.shingle_word(candidate))
                score = jaccard_score * self.word_counter[candidate] / max([self.word_counter[c] for c in candidates])
                # print(jaccard_score, score)
                if score > top_score:
                    top = candidate
                    top_score = score
            final_result = final_result + top + " "

        return final_result.strip()
    
if __name__ == "__main__":
    with open("preprocessed_docs.json", "r") as FILE:
        all_documents = json.load(FILE)

    doc_strings = [' '.join(doc['stars'] + doc['genres'] + doc['summaries']) for doc in all_documents if len(doc['summaries']) > 0]

    sc = SpellCorrection(doc_strings)

    print(sc.spell_check("The amazing soectacular unbeleivable astonishing alright breakaing"))