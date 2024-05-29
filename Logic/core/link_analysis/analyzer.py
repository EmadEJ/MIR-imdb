from .graph import LinkGraph
import json

class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = []
        self.authorities = []
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            self.hubs.append(movie['id'])
            self.graph.add_node(movie['id'])
            for star in movie['stars']:
                if star not in self.authorities:
                    self.graph.add_node(star)
                    self.authorities.append(star)
                self.graph.add_edge(movie['id'], star)
            
    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        for movie in corpus:
            if movie['id'] in self.hubs:
                continue
            for star in movie['stars']:
                if star in self.authorities:
                    if movie['id'] not in self.hubs:
                        self.hubs.append(movie['id'])
                        self.graph.add_node(movie['id'])
                    self.graph.add_edge(movie['id'], star)

    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        h_s = {}
        a_s = {}

        for h in self.hubs:
            h_s[h] = 1
        for a in self.authorities:
            a_s[a] = 1

        for _ in range(num_iteration):
            total_h = 0
            for h in self.hubs:
                seccessors = self.graph.get_successors(h)
                h_s[h] = 0
                for successor in seccessors:
                    h_s[h] += a_s[successor]
                total_h += h_s[h]

            total_a = 0
            for a in self.authorities:
                predecessors = self.graph.get_predecessors(a)
                a_s[a] = 0
                for predecessor in predecessors:
                    a_s[a] += h_s[predecessor]
                total_a += a_s[a]

            for h in self.hubs:
                h_s[h] /= total_h
            for a in self.authorities:
                a_s[a] /= total_a

        a_s = sorted(a_s.items(), key=lambda x: x[1], reverse=True)
        h_s = sorted(h_s.items(), key=lambda x: x[1], reverse=True)

        a_s = [x[0] for x in a_s]
        h_s = [x[0] for x in h_s]
        # print(a_s)
        # print(h_s)

        return a_s[:max_result], h_s[:max_result]

if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer
    with open('preprocessed_docs.json', 'r') as FILE:
        docs = json.load(FILE)
    corpus = []    # TODO: it shoud be your crawled data
    for doc in docs:
        movie = {}
        movie['id'] = doc['id']
        movie['title'] = doc['title']
        movie['stars'] = doc['stars']
        corpus.append(movie)
    root_set = [movie for movie in corpus if 'Avengers:' in movie['title'].split()] 
    print(root_set)

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=10, num_iteration=100)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
