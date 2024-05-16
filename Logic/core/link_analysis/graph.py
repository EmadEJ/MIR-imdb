import networkx as nx

class LinkGraph:
    """
    Use this class to implement the required graph in link analysis.
    You are free to modify this class according to your needs.
    You can add or remove methods from it.
    """
    def __init__(self):
        self.G = nx.DiGraph()

    def add_edge(self, u_of_edge, v_of_edge):
        self.G.add_edge(u_of_edge, v_of_edge)

    def add_node(self, node_to_add):
        self.G.add_node(node_to_add)

    def get_successors(self, node):
        return list(self.G.successors(node))

    def get_predecessors(self, node):
        return list(self.G.predecessors(node))
