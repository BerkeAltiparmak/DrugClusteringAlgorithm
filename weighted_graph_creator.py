"""CSC111 Winter 2022 Assignment 3: Graphs, Recommender Systems, and Clustering (Part 2)

Instructions (READ THIS FIRST!)
===============================

This Python module contains new classes to represent *weighted graphs and vertices*,
which we'll use to represent a book review network with scores of reviews as well.
This file is structured very similarly to a3_part1.py.

Copyright and Usage Information
===============================

This file is provided solely for the personal and private use of students
taking CSC111 at the University of Toronto St. George campus. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited. For more information on copyright for CSC111 materials,
please consult our Course Syllabus.

This file is Copyright (c) 2022 Mario Badr, David Liu, and Isaac Waller.
"""
from __future__ import annotations
import csv
from typing import Any, Union

from graph_creator import Graph


class _WeightedVertex:
    """A vertex in a weighted book review graph, used to represent a user or a book.

    Same documentation as _Vertex from Part 1, except now neighbours is a dictionary mapping
    a neighbour vertex to the weight of the edge to from self to that neighbour.
    Note that in Part 2, the weights will be integers between 1 and 5, but in Part 3 the
    weights will be floats.

    Instance Attributes:
        - item: The data stored in this vertex, representing a user or book.
        - kind: The type of this vertex: 'user' or 'book'.
        - neighbours: The vertices that are adjacent to this vertex, and their corresponding
            edge weights.

    Representation Invariants:
        - self not in self.neighbours
        - all(self in u.neighbours for u in self.neighbours)
        - self.kind in {'user', 'book'}
    """
    item: Any
    kind: str
    neighbours: dict[_WeightedVertex, Union[int, float]]

    def __init__(self, item: Any, kind: str) -> None:
        """Initialize a new vertex with the given item and kind.

        This vertex is initialized with no neighbours.

        Preconditions:
            - kind in {'user', 'book'}
        """
        self.item = item
        self.kind = kind
        self.neighbours = {}

    def degree(self) -> int:
        """Return the degree of this vertex."""
        return len(self.neighbours)

    ############################################################################
    # Part 2, Q2
    ############################################################################
    def similarity_score_unweighted(self, other: _WeightedVertex) -> float:
        """Return the unweighted similarity score between this vertex and other.

        The unweighted similarity score is calculated in the same way as the
        similarity score for _Vertex (from Part 1). That is, just look at edges,
        and ignore the weights.
        """
        if self.degree() == 0 or other.degree() == 0:
            return 0.0
        and_count = len([1 for u in self.neighbours if u in other.neighbours])
        or_count = len(self.neighbours) + len(other.neighbours) - and_count
        return and_count / or_count

    def similarity_score_strict(self, other: _WeightedVertex) -> float:
        """Return the strict weighted similarity score between this vertex and other.

        See Assignment handout for details.
        """
        if self.degree() == 0 or other.degree() == 0:
            return 0.0
        similar_count = len([1 for u in self.neighbours if u in other.neighbours
                             and other.neighbours[u] == self.neighbours[u]])
        and_count = len([1 for u in self.neighbours if u in other.neighbours])
        or_count = len(self.neighbours) + len(other.neighbours) - and_count
        return similar_count / or_count


class WeightedGraph(Graph):
    """A weighted graph used to represent a book review network that keeps track of review scores.

    Note that this is a subclass of the Graph class from Part 1, and so inherits any methods
    from that class that aren't overridden here.
    """
    # Private Instance Attributes:
    #     - _vertices:
    #         A collection of the vertices contained in this graph.
    #         Maps item to _WeightedVertex object.
    _vertices: dict[Any, _WeightedVertex]

    def __init__(self) -> None:
        """Initialize an empty graph (no vertices or edges)."""
        self._vertices = {}

        # This call isn't necessary, except to satisfy PythonTA.
        Graph.__init__(self)

    def add_vertex(self, item: Any, kind: str) -> None:
        """Add a vertex with the given item and kind to this graph.

        The new vertex is not adjacent to any other vertices.
        Do nothing if the given item is already in this graph.

        Preconditions:
            - kind in {'user', 'book'}
        """
        if item not in self._vertices:
            self._vertices[item] = _WeightedVertex(item, kind)

    def add_edge(self, item1: Any, item2: Any, weight: Union[int, float] = 1) -> None:
        """Add an edge between the two vertices with the given items in this graph,
        with the given weight.

        Raise a ValueError if item1 or item2 do not appear as vertices in this graph.

        Preconditions:
            - item1 != item2
        """
        if item1 in self._vertices and item2 in self._vertices:
            v1 = self._vertices[item1]
            v2 = self._vertices[item2]

            # Add the new edge
            v1.neighbours[v2] = weight
            v2.neighbours[v1] = weight
        else:
            # We didn't find an existing vertex for both items.
            raise ValueError

    def get_weight(self, item1: Any, item2: Any) -> Union[int, float]:
        """Return the weight of the edge between the given items.

        Return 0 if item1 and item2 are not adjacent.

        Preconditions:
            - item1 and item2 are vertices in this graph
        """
        v1 = self._vertices[item1]
        v2 = self._vertices[item2]
        return v1.neighbours.get(v2, 0)

    def average_weight(self, item: Any) -> float:
        """Return the average weight of the edges adjacent to the vertex corresponding to item.

        Raise ValueError if item does not corresponding to a vertex in the graph.
        """
        if item in self._vertices:
            v = self._vertices[item]
            return sum(v.neighbours.values()) / len(v.neighbours)
        else:
            raise ValueError

    ############################################################################
    # Part 2, Q2
    ############################################################################
    def get_similarity_score(self, item1: Any, item2: Any,
                             score_type: str = 'unweighted') -> float:
        """Return the similarity score between the two given items in this graph.

        score_type is one of 'unweighted' or 'strict', corresponding to the
        different ways of calculating weighted graph vertex similarity, as described
        on the assignment handout.

        Raise a ValueError if item1 or item2 do not appear as vertices in this graph.

        Preconditions:
            - score_type in {'unweighted', 'strict'}
        >>> g = WeightedGraph()
        >>> for i in range(0, 6):
        ...     g.add_vertex(str(i), kind='user')
        >>> g.add_edge('0', '2')
        >>> g.add_edge('0', '3', 2)
        >>> g.add_edge('0', '4')
        >>> g.add_edge('1', '3', 4)
        >>> g.add_edge('1', '4')
        >>> g.add_edge('1', '5')
        >>> g.get_similarity_score('0', '1')
        0.5
        >>> g.get_similarity_score('0', '1', 'strict')
        0.25
        """
        if score_type == 'unweighted':
            return self._vertices[item1].similarity_score_unweighted(self._vertices[item2])
        else:
            return self._vertices[item1].similarity_score_strict(self._vertices[item2])


################################################################################
# Part 2, Q1
################################################################################
def load_weighted_review_graph(reviews_file: str, book_names_file: str) -> WeightedGraph:
    """Return a book review WEIGHTED graph corresponding to the given datasets.

    This should be very similar to the corresponding function Part 1, except now
    the book review scores are used as edge weights.

    Preconditions:
        - reviews_file is the path to a CSV file corresponding to the book review data
          format described on the assignment handout
        - book_names_file is the path to a CSV file corresponding to the book data
          format described on the assignment handout
    """
    g = WeightedGraph()
    book_id_to_title = {}

    with open(book_names_file) as f:
        reader = csv.reader(f)
        for book_id, book_title in reader:
            book_id_to_title[book_id] = book_title

    with open(reviews_file) as f:
        reader = csv.reader(f)

        for user_id, book_id, review_score in reader:
            g.add_vertex(user_id, 'user')
            book_title = book_id_to_title[book_id]
            g.add_vertex(book_title, 'book')
            g.add_edge(book_title, user_id, int(review_score))

        return g


def get_weighted_graph_from_clusters(list_of_clusters: list[list], has_outcast_cluster: bool) -> WeightedGraph:
    g = WeightedGraph()

    for i in range(0, len(list_of_clusters)):
        if not (has_outcast_cluster and i == len(list_of_clusters) - 1):
            cluster_i = list_of_clusters[i]
            edge_tracker = []
            for drug in cluster_i:
                g.add_vertex(drug, str(i))
                for added_drug in edge_tracker:
                    g.add_edge(drug, added_drug)
                edge_tracker.append(drug)

    return g
