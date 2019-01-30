"""
Solve Ax = 1 on an MST
"""
from __future__ import division

from scipy.sparse import find, csr_matrix
import queue as queue
import numpy as np
import copy
import progressbar

class tree_class:

    def __init__(self, root):
        self.root = root
        self.parents = dict()
        self.degree = dict()
        self.diagonal = dict()

    def add_nodes(self, list_nodes):
        """Add a list of nodes to the tree
        """
        for node in list_nodes:
            self.parents[node] = node
            self.degree[node] = 0
            self.diagonal[node] = 0.

    def add_edge(self, child, parent, weight):
        """
        """
        self.parents[child] = parent
        self.degree[parent] += 1
        self.degree[child] += 1
        self.diagonal[parent] += weight
        self.diagonal[child] += weight

    def get_size(self):
        return len(self.parents.keys())


    def get_nodes(self):
        return self.parents.keys()



def get_tree_from_csr(tree_csr, root):
    """Returns a tree from the csr_matrix
    """

    # Get indices, indptr and data from csr_matrix
    indices, indptr, data = tree_csr.indices, tree_csr.indptr, tree_csr.data

    # assert int(0.5*(data.shape[0])) == tree_csr.shape[0]-1 # Make sure that this is a tree

    # Initialize a tree and set the root
    T = tree_class(root)

    # Add nodes to the graph
    T.add_nodes(np.arange(tree_csr.shape[0]))

    # Do a Breadth-First search from the root
    Q = queue.Queue()
    Q.put(root)

    visited =  set() # Set of nodes already visited

    while not Q.empty():
        # Pick a vertex from Q
        x = Q.get()

        # Update the visited set of points
        visited.add(x)

        for i in range(indptr[x],indptr[x+1]):
            neighbor = indices[i]
            if neighbor not in visited:
                # For each neighbor add the edge in the tree
                T.add_edge(neighbor, x, -1*data[i])

                # Add each neighbor to the Queue
                Q.put(neighbor)

    return T


def ordering(treeInput):
    """Computes an ordering of the nodes given a tree
    """

    tree = copy.deepcopy(treeInput)

    # Compute the size
    size = tree.get_size()

    # Initialize the ordering
    ordering = np.zeros(size, dtype=np.int32)

    # Set degree of ground to be 0
    tree.degree[tree.root] = 0
    k = 0

    # Set the ordering of root to be size-1
    ordering[size-1] = tree.root

    for node in tree.get_nodes():
        cur_node = node
        while tree.degree[cur_node] == 1:
            # Change the ordering of the cur_node
            ordering[k] = cur_node

            # Reduce the degree
            tree.degree[cur_node] -= 1

            # Reduce the degree of the parent
            cur_node = tree.parents[cur_node]
            tree.degree[cur_node] -= 1

            # Update k
            k += 1


    return ordering


def solve_system(ordering, treeInput):
    """Solve the system of linear equations (on a tree)
    """

    # Copy the inout to avoid modifying data
    tree = copy.deepcopy(treeInput)

    size = tree.get_size()

    # Initialize the rhs
    r = np.ones(size, dtype=np.float64)

    # Intitialize the output (solution)
    output = np.zeros(size, dtype=np.float64)

    # FORWARD PASS

    k = 0
    while k < size-1:
        r[tree.parents[ordering[k]]] += r[ordering[k]]/tree.diagonal[ordering[k]]
        tree.diagonal[tree.parents[ordering[k]]] -= 1./(tree.diagonal[ordering[k]])
        k += 1

    # WRONG LINE FROM GRADY PSEUDO CODE
    # output[ordering[size-1]] = r[ordering[size-1]]/tree.diagonal[ordering[size-1]]

    k = size-2
    while k >= 0:
        output[ordering[k]] = output[tree.parents[ordering[k]]] + (r[ordering[k]]/tree.diagonal[ordering[k]])
        k -= 1

    return output

def solve(tree_csr, ground):
    """
    Returna the solution of Ax = 1, where A is the laplacian of the input tree with grounded node
    """

    # Initialize a tree structure using the matrix
    tree =  get_tree_from_csr(tree_csr, ground)

    # Compute the ordering
    order =  ordering(tree)

    # Solve the system Ax = 1
    ans = solve_system(order, tree)

    return np.array(ans, dtype=np.float64)


if __name__ == '__main__':

    A = np.array([[1, -1, 0, 0, 0],
         [-1, 2, 0, 0, -1],
         [0, 0, 1, 0, -1],
         [0, 0, 0, 1, -1],
         [0, -1, -1, -1, 3]])


    # Checking tree
    tree = get_tree_from_csr(csr_matrix(np.array(A)),4)

    # Checking ordering
    order = ordering(tree)

    # Check solve_system
    ans = np.array(solve_system(order, tree))
    print(ans)
    print("Answer should be all 1's except at one node...")
    print(A.dot(ans))

    ''' CASE 2 '''

    from getData import *
    from img_to_graph import img_to_graph
    import progressbar

    from matplotlib import pyplot as plt

    import numpy as np
    from scipy.sparse import find, csr_matrix, diags, triu
    from scipy.sparse.csgraph import laplacian, connected_components
    from scipy.sparse.linalg import spsolve
    from UMST import get_umst, get_mst
    from scipy.linalg import norm

    img, img_gt = get_random_1Obj()
    img_graph = img_to_graph(img)

    img_graph_mst = get_mst(img_graph)
    img_laplacian = csr_matrix(laplacian(img_graph_mst))

    d = img_graph.sum(axis=1)
    ground = np.argmax(d)

    # Checking tree
    tree = get_tree_from_csr(img_laplacian,int(ground))

    # Checking ordering
    order = ordering(tree)

    # Check solve_system
    ans = np.array(solve_system(order, tree))

    print("Answer should be all 1's except at one node...")
    print(img_laplacian.dot(ans))
