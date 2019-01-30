"""
Code to calculate the Union Maximum Spanning Tree of the graph
"""

import numpy as np
from scipy.sparse import find, csr_matrix

from scipy.sparse.csgraph import connected_components

class unionFind:

    def __init__(self):
        self.size = {}
        self.parents = {}

    def add_element(self, elt):
        """Add a single element
        """
        self.parents[elt] = elt
        self.size[elt] = 1


    def add_list_elemnts(self, list_elements):
        """Add a list of elements to the union find
        """
        for elt in list_elements:
            self.parents[elt] = elt
            self.size[elt] = 1


    def find(self, x):
        """get the root of x
        """
        list_path = []
        par = x
        while self.parents[par] != par:
            par = self.parents[par]
            list_path.append(par)

        # PATH COMPRESSION
        for node in list_path:
            self.parents[node] = par

        return par


    def union(self, x, y):
        """Union of x and y
        """
        # Get the roots of the components
        rootx, rooty = self.find(x), self.find(y)

        if rootx != rooty:

            # WEIGHTED RANK
            if self.size[rootx] > self.size[rooty]:
                rootx, rooty = rooty, rootx

            self.parents[rootx] = rooty


"""
Calculate the union of minimum spanning tree given a graph
"""

def get_umst(img_graph, bucketing='epsilon', eps=1e-4):
    """Return the UMST of the given graph
    """

    # Initialize the Union FInd
    UF = unionFind()
    UF.add_list_elemnts(np.arange(img_graph.shape[0]))

    # List of UMST edges
    UMST_edges = []

    def _generate_edges():
        """yeilds a set of edges using bucketing
        """
        u, v, w = find(img_graph)
        wBucket = np.array(w/eps, dtype=np.int32)

        list_vals = -1*np.sort(-1*np.unique(wBucket))

        for l in list_vals:
            ind = np.where(wBucket == l)
            yield zip(u[ind],v[ind],w[ind])

    for gen_list_edges in _generate_edges():

        list_edges = list(gen_list_edges)
        for edge in list_edges:
            ex, ey, ew = edge
            if UF.find(ex) != UF.find(ey):
                UMST_edges.append((ex, ey, ew))
                UMST_edges.append((ey, ex, ew))
        for edge in list_edges:
            UF.union(edge[0], edge[1])


    # Construct the UMST graph
    u, v, w = (zip(*UMST_edges))
    UMST = csr_matrix((w,(u,v)), shape = img_graph.shape)
    return UMST

def get_mst(img_graph):
    """Return a MST of the input graph
    """

    # Initialize the Union FInd
    UF = unionFind()
    UF.add_list_elemnts(np.arange(img_graph.shape[0]))

    # List of MST edges
    MST_edges = []

    # get the edges in the graph
    u, v, w = find(img_graph)
    list_edges = list(zip(u,v,w))
    list_edges.sort(key= lambda x: -x[2])

    for edge in list_edges:
        ex, ey, ew = edge

        if UF.find(ex) != UF.find(ey):
            MST_edges.append((ex, ey, ew))
            MST_edges.append((ey, ex, ew))

            UF.union(ex, ey)

    # Construct the MST graph
    u, v, w = (zip(*MST_edges))
    MST = csr_matrix((w,(u,v)), shape = img_graph.shape)
    return MST
