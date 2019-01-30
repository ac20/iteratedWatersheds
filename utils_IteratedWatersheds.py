"""
This file contains the code required for IteratedWatersheds
"""

#----------------------------------------------------------------------------------------------#
#--------------------------------------- PRIORITY QUEUE ---------------------------------------#
#----------------------------------------------------------------------------------------------#

import itertools
import heapq

class priorityQueue:
    
    def __init__(self):
        self.pq = []
        self.entry_finder = {}
        self.REMOVED = "REMOVED"
        self.counter = itertools.count()
        
    def add_element(self, elt, priority=0):
        """ Add an element to the queue
        """
        if elt in self.entry_finder.keys():
            self.remove_element(elt)
        count = next(self.counter)
        entry = [priority, count, elt]
        self.entry_finder[elt] = entry
        heapq.heappush(self.pq, entry)
        
    def remove_element(self, elt):
        """
        """
        entry = self.entry_finder[elt]
        entry[-1] = self.REMOVED
        
    def pop_element(self):
        while self.pq:
            priority, count, elt = heapq.heappop(self.pq)
            if elt != self.REMOVED:
                del self.entry_finder[elt]
                return elt
        raise KeyError('Cannot pop an element from empty queue')



#-----------------------------------------------------------------------------------------------#
#---------------------------------- IMAGE FORESTING TRANSFORM ----------------------------------#
#-----------------------------------------------------------------------------------------------#

import numpy as np

def _get_cost(a,b,flag='SP_SUM'):
    if flag == 'SP_SUM':
        return a+b
    elif flag == 'SP_MAX':
        return max(a,b)
    else:
        raise Exception('flag should be SP_SUM or SP_MAX but got {}'.format(flag))

def _ift(graph,init_labels,alg_flag='SP_SUM'):
    """Return the image foresting transform for the labels
    
    graph : sparse matrix
        The edge weighted graph on which the shortest path must be calculated
    init_labels : ndarray
        Initial Labelling. 0 indicates unlabelled pixels.
    """
    
    size = graph.shape[0]
    indices, indptr, data = graph.indices, graph.indptr, graph.data
    
    # Initialization - Labels and Cost
    labelling = np.array(init_labels)
    cost = np.inf*np.ones(size, dtype=np.int32)
    cost[init_labels > 0] = 0
    
    pq = priorityQueue()
    for i in np.where(init_labels > 0)[0]:
        pq.add_element(i,0)
        
    while pq.pq:
        try:
            x = pq.pop_element()
        except:
            break
        
        for i in range(indptr[x],indptr[x+1]):
            y = indices[i]
            c_prime = _get_cost(cost[x],data[i],alg_flag) # New cost
            if c_prime < cost[y]:
                cost[y] = c_prime
                pq.add_element(y,priority=c_prime)
                labelling[y] = labelling[x]
        
                
        
    assert np.all(labelling > 0), "Some labellings are still 0. Check if the graph is connected!!"
    
    return labelling, np.sum(cost)
                               



#-----------------------------------------------------------------------------------------------#
#-------------------------------------- CALCULATE CENTERS --------------------------------------#
#-----------------------------------------------------------------------------------------------#

from scipy.sparse.csgraph import floyd_warshall

def _calc_centers(graph, X, labelling, method='nearest'):
    """Return the new centers
    
    graph : sparse matrix
        Indicates the graph constructed from X
    X : ndarray
        Original Data
    labelling: 1d array
        The labelling of the vertices
    method : one of 'nearest', 'floyd_warshall', 'erosion'
        Method to calculate the new centers
    """
    size = graph.shape[0]
    centers = np.zeros(size)
    
    max_label = int(np.max(labelling))
    for label in range(1, max_label+1):
        index_vert = np.where(labelling == label)[0]
        if method == 'floyd_warshall':
            subgraph = ((graph[index_vert]).transpose())[index_vert]
            FW = floyd_warshall(subgraph, directed=False)
            ind_center = np.argmin(np.max(FW, axis=-1))
            centers[index_vert[ind_center]] = label
        elif method == 'nearest':
            mean_subgraph = np.mean(X[index_vert,:], axis=0, keepdims=True)
            dist_from_mean = np.sum((X[index_vert,:] - mean_subgraph)**2, axis = -1)
            ind_center = np.argmin(dist_from_mean.flatten())
            centers[index_vert[ind_center]] = label
        else:
            raise Exception("Only use floyd_warshall or nearest methods (for now)")
        
    return centers
        
#------------------------------------------------------------------------------------------------#
#-------------------------------------- ITERATED WATERSHED --------------------------------------#
#------------------------------------------------------------------------------------------------#

import numpy as np

def iterated_watershed(graph, X, number_clusters=6, max_iterations=100):
    """
    """
    
    size = graph.shape[0]
    
    #Initialize Random Centers
    centers = np.zeros(size, dtype=np.int32)
    index_centers = np.random.choice(size,number_clusters,replace=False)
    centers[index_centers] = np.arange(number_clusters) + 1
    
    #Cost
    cost_history = []
    opt_cost = np.inf
    opt_labels = None
    opt_centers = None
    for i in range(max_iterations):
        # Label all the vertices
        labels, cost_arr = _ift(graph,centers)
        
        # Update the optimal cost
        if cost_arr < opt_cost:
            opt_labels = labels
            opt_cost = cost_arr
            opt_centers = centers
        
        # Compute the cost and append it to the history
        cost_history.append(cost_arr)
        
        # Compute the new centers
        centersNew = _calc_centers(graph, X, labels)
        
        # Break if the centers did not change!
        if np.all(centers==centersNew):
            break
        else:
            centers=centersNew
        
    return opt_labels, cost_history, opt_centers

#-------------------------------------------------------------------------------------#
#------------------------------- MAKE GRAPH UNDIRECTED -------------------------------#
#-------------------------------------------------------------------------------------#

import scipy as sp


def make_undirected(G):
    """This function takes the graph and returns the undirected version.
    """
    u,v,w = sp.sparse.find(G)
    
    edges = dict()
    for i in range(u.shape[0]):
        edges[(u[i],v[i])] = w[i]
        edges[(v[i],u[i])] = w[i]
        
   
    sizeNew = len(edges)
    uNew = np.zeros(sizeNew, dtype=np.int32)
    vNew = np.zeros(sizeNew, dtype=np.int32)
    wNew = np.zeros(sizeNew, dtype=np.float64)
    
    i = 0
    for ((u,v),w) in edges.items():
        uNew[i], vNew[i], wNew[i] = u, v, w
        i += 1
        
    assert i == sizeNew, "Something went wrong"
    
    return sp.sparse.csr_matrix((wNew,(uNew,vNew)), shape=G.shape)

#-----------------------------------------------------------------------------------------------#
#------------------------------------ CONSTRUCT 4-ADJ GRAPH ------------------------------------#
#-----------------------------------------------------------------------------------------------#

from scipy.sparse import csr_matrix



def img_to_graph(img, beta=1., eps=1e-6, which='similarity'):
    """
    """
    
    s0, s1, s2 = img.shape
    
    xGrid, yGrid = np.meshgrid(np.arange(s0), np.arange(s1))
    indGrid = (xGrid*s1 + yGrid).transpose()
    
    data_vert = np.sum((img[:-1,:,:] - img[1:,:,:])**2, axis = -1).flatten()
    row_vert = indGrid[:-1,:].flatten()
    col_vert = indGrid[1:,:].flatten()
    
    data_horiz = np.sum((img[:,:-1,:] - img[:,1:,:])**2, axis = -1).flatten()
    row_horiz = indGrid[:,:-1].flatten()
    col_horiz = indGrid[:,1:].flatten()
    
    data = np.concatenate((data_vert, data_horiz))
    row = np.concatenate((row_vert, row_horiz))
    col = np.concatenate((col_vert, col_horiz))
    
    if which == 'similarity':
        # Make the data into similarities
        data = np.exp(-beta*data/data.std()) + eps
    elif which == 'dissimilarity':
        data += eps
    else:
        raise Exception("Should be one of similarity or dissimilarity.")
    
    graph = csr_matrix((data,(row, col)), shape = (s0*s1, s0*s1))
    graph = make_undirected(graph)
    
    return graph




#-------------------------------------------------------------------------------------------------#
#----------------------------------------- GENERATE DATA -----------------------------------------#
#-------------------------------------------------------------------------------------------------#

from PIL import Image
import numpy as np
import os

def generate_data_1Object(number_images=10**6):
    """Generate data from weizman 1-Object dataset
    """
    
    list_names = list(filter(lambda x:(x[0] != '.') and (x[-3:] != "mat"), os.listdir("./Workstation_files/1obj")))
    np.random.shuffle(list_names)
    
    total_count = len(list_names)
    
    for i in range(min(total_count, number_images)):
        fname = list_names[i]
        img = np.array(Image.open("./Workstation_files/1obj/"+fname+"/src_color/"+fname+".png"), dtype=np.float64)
        img = img/255.

        list_gt_fname = list(filter(lambda x: x[0] != '.', os.listdir("./Workstation_files/1obj/"+fname+"/human_seg/")))
        gt = []
        for gt_name in list_gt_fname:
            tmp = np.array(Image.open("./Workstation_files/1obj/"+fname+"/human_seg/"+gt_name), dtype=np.int32)
            z = np.zeros(tmp.shape[:2], dtype=np.int32)
            z[np.where(tmp[:,:,0]/255. == 1)] = 1
            gt.append(z)
            
            
        yield img, gt, fname
        

def generate_data_2Object(number_images=10**6):
    """Generate data from weizman 2-Object dataset
    """
    
    list_names = list(filter(lambda x: (x[0] != '.') and (x[-3:] != "mat"), os.listdir("./Workstation_files/2obj")))
    np.random.shuffle(list_names)
    
    total_count = len(list_names)
    
    for i in range(min(total_count, number_images)):
        fname = list_names[i]
        img = np.array(Image.open("./Workstation_files/2obj/"+fname+"/src_color/"+fname+".png"), dtype=np.float64)
        img = img/255.

        list_gt_fname = list(filter(lambda x: x[0] != '.', os.listdir("./Workstation_files/2obj/"+fname+"/human_seg/")))
        gt = []
        for gt_name in list_gt_fname:
            tmp = np.array(Image.open("./Workstation_files/2obj/"+fname+"/human_seg/"+gt_name), dtype=np.int32)
            z = np.zeros(tmp.shape[:2], dtype=np.int32)
            z[np.where(tmp[:,:,0]/255. == 1)] = 1
            z[np.where(tmp[:,:,2]/255. == 1)] = 2
            gt.append(z)
            
            
        yield img, gt, fname    
    

#-------------------------------------------------------------------------------------------------#
#---------------------------------------- EVAULATE OUTPUT ----------------------------------------#
#-------------------------------------------------------------------------------------------------#


from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster.supervised import _comb2

def evaluate_output(ypred, list_gt):
    """
    """
    
    list_AMI, list_ARI, list_fScore, list_acc = [], [], [], []
    for gt in list_gt:
        ytrue = gt.flatten()
        ypred = ypred.flatten()
        
        AMI = adjusted_mutual_info_score(ytrue, ypred)
        list_AMI.append(AMI)
        ARI = adjusted_rand_score(ytrue, ypred)
        list_ARI.append(ARI)
        
        # Get the contigency matrix        
        contingency = contingency_matrix(ytrue, ypred)
        
        # F-Score : 
        TP = sum(_comb2(n_ij) for n_ij in contingency.flatten())
        total_positive_pred = sum(_comb2(n_c) for n_c in np.ravel(contingency.sum(axis=1)))
        total_positive_true = sum(_comb2(n_c) for n_c in np.ravel(contingency.sum(axis=0)))
        
        precision, recall = TP/total_positive_pred, TP/total_positive_true
        f_score = 2*precision*recall/(precision + recall)
        list_fScore.append(f_score)
        
        # Assume that the class of a predicted label is the class with highest intersection
        accuracy = np.sum(np.max(contingency, axis=0))/np.sum(contingency)
        list_acc.append(accuracy)
        
        
    return np.max(list_AMI), np.max(list_ARI), np.max(list_fScore), np.max(list_acc)
        
#-------------------------------------------------------------------------------------------------#
#-------------------------------------- SPECTRAL CLUSTERING --------------------------------------#
#-------------------------------------------------------------------------------------------------#        
        

from scipy.sparse import csr_matrix
from sklearn.cluster import k_means
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import eigsh
import scipy as sp
from scipy import sparse

from sklearn.cluster import spectral_clustering as _spectral_clustering

def spectral_clustering(graph, n_clusters, beta_weight=1., eps_weight=1e-6):
    """
    """
    
    graphTmp = csr_matrix(graph, copy=True)
    graphTmp.data = np.exp(-beta_weight*graphTmp.data/graphTmp.data.std()) + eps_weight
    
    L = laplacian(graphTmp, normed=True)
    eigval, embed = eigsh(L, 6, sigma = 1e-10)
    d0, labels, d2 = k_means(embed,6, n_init=10)
    
    return labels
    
    
#--------------------------------------------------------------------------------------------------#
#----------------------------------- ISOPERIMETRIC PARTITIONING -----------------------------------#
#--------------------------------------------------------------------------------------------------#



from IsoperimetricPartitioning import recursive_iso_parition, isoperimetric_Full
"""
isoperimetric_Full(img_graph, ground=0)
recursive_iso_parition(img_graph, algCode='full')
"""
def isoperimetric_partitioning(graph, beta_weight=1., eps_weight=1e-6, which='full'):
    """
    """
    graphTmp = csr_matrix(graph, copy=True)
    graphTmp.data = np.exp(-beta_weight*graphTmp.data/graphTmp.data.std()) + eps_weight
    
    seed = 0
    if which == 'full':
        labels, isoSolution = isoperimetric_Full(graphTmp, ground=seed)
    elif which == 'recursive':
        labels = recursive_iso_parition(graphTmp, algCode='full')
        
    return labels

#--------------------------------------------------------------------------------------------------#
#-------------------------------------- K-MEANS PARTITIONING --------------------------------------#
#--------------------------------------------------------------------------------------------------#

from sklearn.cluster import KMeans

def kmeans_adapted(img, n_clusters):
    """
    """
    
    s0, s1, s2 = img.shape
    X = img.reshape((s0*s1, s2))
    
    xgrid, ygrid = np.meshgrid(np.arange(s0), np.arange(s1))
    xgrid, ygrid = xgrid.transpose(), ygrid.transpose()
    xgrid, ygrid = (xgrid.flatten()).reshape((-1,1)), (ygrid.flatten()).reshape((-1,1))
    grid = np.hstack((xgrid, ygrid))
    grid = grid/np.max(grid)
    
    X = np.hstack((X, grid))
    
    clf = KMeans(n_clusters=n_clusters)
    labels = clf.fit_predict(X)
    
    return labels



#---------------------------------------------------------------------------------------------------#
#-------------------------------------- GET ROAD NETWORK DATA --------------------------------------#
#---------------------------------------------------------------------------------------------------#


import pandas as pd
import numpy as np
import networkx as nx
import scipy as sp

def get_road_network_data(city='Mumbai'):
    """
    """
    data = pd.read_csv("./RoadNetwork/"+city+"/"+city+"_Edgelist.csv")
    
    size = data.shape[0]
    
    X = np.array(data[['XCoord','YCoord']])
    u, v = np.array(data['START_NODE'], dtype=np.int32), np.array(data['END_NODE'], dtype=np.int32)
    w = np.array(data['LENGTH'], dtype=np.float64)
    w = w/np.max(w) + 1e-6
    
    G = sp.sparse.csr_matrix((w, (u,v)), shape = (size, size))
    
    n, labels = sp.sparse.csgraph.connected_components(G)
    
    if n == 1:
        return G
    
    # If there are more than one connected component, return the largest connected component
    count_size_comp = np.bincount(labels)
    z = np.argmax(count_size_comp)
    indSelect = np.where(labels==z)
    
    Gtmp = G[indSelect].transpose()[indSelect]
    Gtmp = make_undirected(Gtmp)
    
    return X[indSelect], Gtmp
    
    

#---------------------------------------------------------------------------------------------------#
#------------------------------------- K-MEANS ON ROAD NETWORK -------------------------------------#
#---------------------------------------------------------------------------------------------------#    
    
from sklearn.cluster import KMeans        
from scipy.sparse.csgraph import dijkstra

def kmeans_on_roadNetwork(G, X, nClusters):
    """
    """
    
    clf = KMeans(n_clusters=nClusters, n_init=20)
    labels = clf.fit_predict(X)
    
    seeds = np.zeros(X.shape[0], dtype=np.int32)
    for l in np.unique(labels):
        center = np.mean(X[labels==l], axis=0)
        closest_point = np.argmin(np.sum((X - center.reshape((1,-1)))**2, axis=1))
        seeds[closest_point] = l+1
        
    
    labels, cost = _ift(G,seeds,alg_flag='SP_SUM')
        
    return cost, labels
    
    
def calculate_cost_on_labels(G, X, labels):
    """
    """
    seeds = np.zeros(X.shape[0], dtype=np.int32)
    for l in np.unique(labels):
        center = np.mean(X[labels==l], axis=0)
        closest_point = np.argmin(np.sum((X - center.reshape((1,-1)))**2, axis=1))
        seeds[closest_point] = l+1

    _, cost = _ift(G,seeds,alg_flag='SP_SUM')
        
    return cost





#---------------------------------------------------------------------------------------------------#
#----------------------------------------- GREEDY K-CENTER -----------------------------------------#
#---------------------------------------------------------------------------------------------------#   

from scipy.sparse.csgraph import dijkstra

def greedy_kCenter(G, X, n_clusters=6):
    """
    """
    
    size_data = X.shape[0]
    list_points = np.arange(size_data, dtype=np.int32)
    
    # Initialize a random point as center
    center = np.random.choice(list_points, size=1, replace=False)[0]
    list_points = np.delete(list_points, center)
    
    list_centers = [center]
    
    distances_from_centers = []
    dist = dijkstra(G, directed=False, indices = center)
    distances_from_centers.append(dist.flatten())
    
    for i in range(n_clusters-1):
        tmp = np.array(distances_from_centers)
        new_center = np.argmax(np.min(tmp, axis=0))
        
        list_centers.append(new_center)
        
        dist = dijkstra(G, directed=False, indices = new_center)
        distances_from_centers.append(dist.flatten())
        
        

    seeds = np.zeros(size_data, dtype=np.int32)
    i = 1
    for c in list_centers:
        seeds[c] = i
        i += 1
        
    labels, cost = _ift(G,seeds,alg_flag='SP_SUM')
    
    return cost, labels
    
        
        
        
    
    











