"""
Code to calculate the Isoperimetric Paritioning of the greph. Three different method exist -
1. Using the whole original graph
2. Constructing an UMST and then solving the problem on UMST
3. Constructing an MST and then solving the problem on UMST
"""

import numpy as np
from scipy.sparse import find, csr_matrix, diags, triu
from scipy.sparse.csgraph import laplacian, connected_components
from scipy.sparse.linalg import spsolve, cg
from UMST import get_umst, get_mst
from scipy.linalg import norm

from solveGradyMST import solve


# GLOBAL VARIABLES!!
CUTOFF = 50 # min size of the partition



def _compute_isoperimetric_partition(x, img_laplacian,CUTOFF=CUTOFF):
    """Returns the segmentation thresholded to the least isoperimetric value
    """

    size = img_laplacian.shape[0]
    eps = 1e-6

    x0 = np.array(x, dtype = np.float64)

    # Sort the input in increasing order
    x0_sort_ind = np.argsort(x0)

    # rhs
    d = np.ones(size, dtype = np.float64)

    # Calculate the denominators
    denominators = np.arange(size, dtype=np.float64) + 1.0
    denominators[denominators > ((size+0.0)/2.0)] = size + 1 - denominators[denominators > ((size+0.0)/2.0)]


    # Calculate the numerators
    # Use the same order to sort the laplacian
    L = (img_laplacian[x0_sort_ind].transpose())[x0_sort_ind]
    L = L - diags(img_laplacian.diagonal()[x0_sort_ind])
    numerators = np.cumsum(np.sum(L - 2*triu(L), axis=1))
    if np.min(numerators)<0:
        numerators =  numerators - np.min(numerators) + eps
    numerators = np.ravel(numerators)

    minCut = np.argmin(numerators[CUTOFF:-CUTOFF]/denominators[CUTOFF:-CUTOFF])
    minCut = minCut + CUTOFF + 1

    part1, part2 = np.array(x0_sort_ind[:minCut], dtype=np.int32), np.array(x0_sort_ind[minCut:], dtype=np.int32)

    assert part1.shape[0]+part2.shape[0] == img_laplacian.shape[0]

    return part1, part2, np.min(numerators[CUTOFF:-CUTOFF]/denominators[CUTOFF:-CUTOFF])


def _isoParition(img_graph, ground=0, algCode='full', verbose=False):
    """Returns the isoperimetric parition.
    """

    n_comp, dummy = connected_components(img_graph)

    d = img_graph.sum(axis=1)
    ground = np.argmax(d)

    # Get the laplacian on which to calculate the solution based on algCode
    if algCode == 'full':
        img_laplacian = csr_matrix(laplacian(img_graph))
    elif algCode == 'umst':
        img_graph_umst = get_umst(img_graph)
        img_laplacian = csr_matrix(laplacian(img_graph_umst))
    elif algCode == 'mst' or 'mstGrady':
        img_graph_mst = get_mst(img_graph)
        img_laplacian = csr_matrix(laplacian(img_graph_mst))
    else:
        raise Exception("algCode should be one of {'full', 'umst', 'mst'. 'mstGrady'}")

    # get the seeded laplacian
    ind = np.arange(img_graph.shape[0], dtype = np.int32)
    ind = np.hstack([ind[:ground], ind[(ground+1):]])

    # Remove the row and column indicated by ground
    img_laplacian_seeded = (img_laplacian[ind]).transpose()[ind]

    # Solve the isoperimetric equation
    d = np.ones(img_laplacian_seeded.shape[0], dtype=np.float64)
    if algCode == 'mstGrady':
        x0 = solve(img_laplacian,ground)
        x0 = x0[ind]
    else:
        x0 = spsolve(img_laplacian_seeded, d)

    minVal = np.min(x0)
    if minVal < 0:
        x0[x0<0] = np.max(x0) + 1

    if verbose:
        print("Error is {:4f}".format(norm(img_laplacian_seeded.dot(x0) - d)/norm(d)))


    x0 = x0 - np.min(x0) + 1e-6
    x0 = x0/np.max(x0) # Normalize to get values between [0,1]

    # Get the total answer
    ans = np.zeros(img_graph.shape[0], dtype=np.float64)
    ans[ind]= x0

    # Compute the threshold
    img_laplacian = csr_matrix(laplacian(img_graph))
    part1, part2, val = _compute_isoperimetric_partition(ans, img_laplacian)

    return part1, part2, val, ans


def isoperimetric_Full(img_graph, ground=0):

    part1, part2, val, iso_solution = _isoParition(img_graph, ground, algCode='full')
    ans_segmented = np.zeros(img_graph.shape[0], dtype=np.float64)
    ans_segmented[part1] = 1.

    return ans_segmented, iso_solution

def isoperimetric_UMST(img_graph, ground=0):

    part1, part2, val, iso_solution = _isoParition(img_graph, ground, algCode='umst')

    ans_segmented = np.zeros(img_graph.shape[0], dtype=np.float64)
    ans_segmented[part1] = 1.

    return ans_segmented, iso_solution

def isoperimetric_MST(img_graph, ground=0):

    part1, part2, val, iso_solution = _isoParition(img_graph, ground, algCode='mst')
    ans_segmented = np.zeros(img_graph.shape[0], dtype=np.float64)
    ans_segmented[part1] = 1.

    return ans_segmented, iso_solution

def isoperimetric_MST_Grady(img_graph, ground=0):

    part1, part2, val, iso_solution = _isoParition(img_graph, ground, algCode='mstGrady')
    ans_segmented = np.zeros(img_graph.shape[0], dtype=np.float64)
    ans_segmented[part1] = 1.

    return ans_segmented, iso_solution

def recursive_iso_parition(img_graph, algCode='full'):
    """Performs the recursive partition
    """

    if algCode in ['full', 'umst', 'mst']:
        pass
    else:
        raise Exception("algCode should be one of {'full', 'umst' 'mst'}")


    stopAlg = 1e-1
    ans = np.zeros(img_graph.shape[0], dtype=np.float64)
    ind = np.arange(img_graph.shape[0], dtype=np.int32)
    ans = _perform_recursion(img_graph, stopAlg, algCode, ans, ind, 0 )

    return ans


def _perform_recursion(img_graph, stopAlg, algCode, ans, ind, recursion_depth, verbose=False):
    """Recrusively calculate the paritions
    """

    n_components, dummy = connected_components(img_graph)
    if n_components > 1:
        if verbose:
            print("Stopped recursion. Number of connected components is {} which is greater than 1.".format(n_components))
        return ans

    if recursion_depth > 2:
        if verbose:
            print("Stopped recursion. Recursion depth exceeded with depth {}.".format(recursion_depth-1))
        return ans

    if img_graph.shape[0] > 2*CUTOFF:
        part1, part2, val, x0 = _isoParition(img_graph, ground=0, algCode=algCode)
    else:
        val = 2

    if val > stopAlg:
        if verbose:
            print("Stopped recursion. value obtained is {:.4f} while stopping criteria is {:.4f} (units of 1e-4).".format(val*1e4, stopAlg*1e4))
        return ans


    tmp_ind = np.where(ans >= ans[ind[0]])
    ans[tmp_ind] += 1
    ans[ind[part2]] += 1

    if part1.shape[0] > 2*CUTOFF:
        W = (img_graph[part1]).transpose()[part1]
        ans = _perform_recursion(W, stopAlg, algCode, ans, ind[part1],recursion_depth+1)

    if part2.shape[0] > 2*CUTOFF:
        W = (img_graph[part2]).transpose()[part2]
        ans = _perform_recursion(W, stopAlg, algCode, ans, ind[part2],recursion_depth+1)

    return ans
