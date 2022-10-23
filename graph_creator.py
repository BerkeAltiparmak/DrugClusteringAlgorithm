"""CSC111 Winter 2022 Assignment 3: Graphs, Recommender Systems, and Clustering (Part 1)

Instructions (READ THIS FIRST!)
===============================

This Python module contains the modified graph and vertex classes you'll be using as the basis
for this assignment, as well as additional functions for you to complete in this part.

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
from typing import Any

# Make sure you've installed the necessary Python libraries (see assignment handout
# "Installing new libraries" section)
import networkx as nx  # Used for visualizing graphs (by convention, referred to as "nx")


class _Vertex:
    """A vertex in a book review graph, used to represent a user or a book.

    Each vertex item is either a user id or book title. Both are represented as strings,
    even though we've kept the type annotation as Any to be consistent with lecture.

    Instance Attributes:
        - item: The data stored in this vertex, representing a user or book.
        - kind: The type of this vertex: 'user' or 'book'.
        - neighbours: The vertices that are adjacent to this vertex.

    Representation Invariants:
        - self not in self.neighbours
        - all(self in u.neighbours for u in self.neighbours)
        - self.kind in {'user', 'book'}
    """
    item: Any
    kind: str
    neighbours: set[_Vertex]

    def __init__(self, item: Any, kind: str) -> None:
        """Initialize a new vertex with the given item and kind.

        This vertex is initialized with no neighbours.

        Preconditions:
            - kind in {'user', 'book'}
        """
        self.item = item
        self.kind = kind
        self.neighbours = set()

    def degree(self) -> int:
        """Return the degree of this vertex."""
        return len(self.neighbours)

    ############################################################################
    # Part 1, Q3
    ############################################################################
    def similarity_score(self, other: _Vertex) -> float:
        """Return the similarity score between this vertex and other.

        See Assignment handout for definition of similarity score.
        """
        if self.degree() == 0 or other.degree() == 0:
            return 0.0
        and_count = len(self.neighbours.intersection(other.neighbours))
        or_count = len(self.neighbours) + len(other.neighbours) - and_count
        return and_count / or_count


class Graph:
    """A graph used to represent a book review network.
    """
    # Private Instance Attributes:
    #     - _vertices:
    #         A collection of the vertices contained in this graph.
    #         Maps item to _Vertex object.
    _vertices: dict[Any, _Vertex]

    def __init__(self) -> None:
        """Initialize an empty graph (no vertices or edges)."""
        self._vertices = {}

    def add_vertex(self, item: Any, kind: str) -> None:
        """Add a vertex with the given item and kind to this graph.

        The new vertex is not adjacent to any other vertices.
        Do nothing if the given item is already in this graph.

        Preconditions:
            - kind in {'user', 'book'}
        """
        if item not in self._vertices:
            self._vertices[item] = _Vertex(item, kind)

    def add_edge(self, item1: Any, item2: Any) -> None:
        """Add an edge between the two vertices with the given items in this graph.

        Raise a ValueError if item1 or item2 do not appear as vertices in this graph.

        Preconditions:
            - item1 != item2
        """
        if item1 in self._vertices and item2 in self._vertices:
            v1 = self._vertices[item1]
            v2 = self._vertices[item2]

            v1.neighbours.add(v2)
            v2.neighbours.add(v1)
        else:
            raise ValueError

    def adjacent(self, item1: Any, item2: Any) -> bool:
        """Return whether item1 and item2 are adjacent vertices in this graph.

        Return False if item1 or item2 do not appear as vertices in this graph.
        """
        if item1 in self._vertices and item2 in self._vertices:
            v1 = self._vertices[item1]
            return any(v2.item == item2 for v2 in v1.neighbours)
        else:
            return False

    def get_neighbours(self, item: Any) -> set:
        """Return a set of the neighbours of the given item.

        Note that the *items* are returned, not the _Vertex objects themselves.

        Raise a ValueError if item does not appear as a vertex in this graph.
        """
        if item in self._vertices:
            v = self._vertices[item]
            return {neighbour.item for neighbour in v.neighbours}
        else:
            raise ValueError

    def get_all_vertices(self, kind: str = '') -> set:
        """Return a set of all vertex items in this graph.

        If kind != '', only return the items of the given vertex kind.

        Preconditions:
            - kind in {'', 'user', 'book'}
        """
        if kind != '':
            return {v.item for v in self._vertices.values() if v.kind == kind}
        else:
            return set(self._vertices.keys())

    def to_networkx(self, max_vertices: int = 5000) -> nx.Graph:
        """Convert this graph into a networkx Graph.

        max_vertices specifies the maximum number of vertices that can appear in the graph.
        (This is necessary to limit the visualization output for large graphs.)

        Note that this method is provided for you, and you shouldn't change it.
        """
        graph_nx = nx.Graph()
        for v in self._vertices.values():
            graph_nx.add_node(v.item, kind=v.kind)

            for u in v.neighbours:
                if graph_nx.number_of_nodes() < max_vertices:
                    graph_nx.add_node(u.item, kind=u.kind)

                if u.item in graph_nx.nodes:
                    graph_nx.add_edge(v.item, u.item)

            if graph_nx.number_of_nodes() >= max_vertices:
                break

        return graph_nx

    ############################################################################
    # Part 1, Q3
    ############################################################################
    def get_similarity_score(self, item1: Any, item2: Any) -> float:
        """Return the similarity score between the two given items in this graph.

        Raise a ValueError if item1 or item2 do not appear as vertices in this graph.

        >>> g = Graph()
        >>> for i in range(0, 6):
        ...     g.add_vertex(str(i), kind='user')
        >>> g.add_edge('0', '2')
        >>> g.add_edge('0', '3')
        >>> g.add_edge('0', '4')
        >>> g.add_edge('1', '3')
        >>> g.add_edge('1', '4')
        >>> g.add_edge('1', '5')
        >>> g.get_similarity_score('0', '1')
        0.5
        """
        return self._vertices[item1].similarity_score(self._vertices[item2])

    ############################################################################
    # Part 1, Q4
    ############################################################################
    def recommend_books(self, book: str, limit: int) -> list[str]:
        """Return a list of up to <limit> recommended books based on similarity to the given book.

        The return value is a list of the titles of recommended books, sorted in
        *descending order* of similarity score. Ties are broken in descending order
        of book title. That is, if v1 and v2 have the same similarity score, then
        v1 comes before v2 if and only if v1.item > v2.item.

        The returned list should NOT contain:
            - the input book itself
            - any book with a similarity score of 0 to the input book
            - any duplicates
            - any vertices that represents a user (instead of a book)

        Up to <limit> books are returned, starting with the book with the highest similarity score,
        then the second-highest similarity score, etc. Fewer than <limit> books are returned if
        and only if there aren't enough books that meet the above criteria.

        Preconditions:
            - book in self._vertices
            - self._vertices[book].kind == 'book'
            - limit >= 1
        """
        users_that_read_the_book = self._vertices[book].neighbours

        book_similarity = {}
        for common_reader in users_that_read_the_book:
            for other_book in common_reader.neighbours:
                if other_book.item != book and other_book.item not in book_similarity:
                    book_similarity[other_book.item] = \
                        self.get_similarity_score(book, other_book.item)

        value_sorted_and_swapped_brc = sorted([(value, key) for (key, value)
                                               in book_similarity.items()], reverse=True)

        size_to_return = len(value_sorted_and_swapped_brc) \
            if len(value_sorted_and_swapped_brc) < limit else limit

        recommended_books = []
        for i in range(0, size_to_return):
            recommended_books.append(value_sorted_and_swapped_brc[i][1])

        return recommended_books


################################################################################
# Part 1, Q1
################################################################################

def load_review_graph(reviews_file: str, book_names_file: str) -> Graph:
    """Return a book review graph corresponding to the given datasets.

    The book review graph stores one vertex for each user and book in the datasets.
    Each vertex stores as its item either a user ID or book TITLE (the latter is why
    you need the book_names_file). Use the "kind" _Vertex attribute to differentiate
    between the two vertex types.

    Edges represent a review between a user and a book. In this graph, each edge
    only represents the existence of a review---IGNORE THE REVIEW SCORE in the
    datasets, as we don't have a way to represent these scores (yet).

    Preconditions:
        - reviews_file is the path to a CSV file corresponding to the book review data
          format described on the assignment handout
        - book_names_file is the path to a CSV file corresponding to the book data
          format described on the assignment handout

    >>> g = load_review_graph('data/reviews_small.csv', 'data/book_names.csv')
    >>> len(g.get_all_vertices(kind='book'))
    4
    >>> len(g.get_all_vertices(kind='user'))
    5
    >>> user1_reviews = g.get_neighbours('user1')
    >>> len(user1_reviews)
    3
    >>> "Harry Potter and the Sorcerer's Stone (Book 1)" in user1_reviews
    True
    """
    g = Graph()
    book_id_to_title = {}

    with open(book_names_file) as f:
        reader = csv.reader(f)
        for book_id, book_title in reader:
            book_id_to_title[book_id] = book_title

    with open(reviews_file) as f:
        reader = csv.reader(f)

        for user_id, book_id, _ in reader:
            g.add_vertex(user_id, 'user')
            book_title = book_id_to_title[book_id]
            g.add_vertex(book_title, 'book')
            g.add_edge(user_id, book_title)

        return g
