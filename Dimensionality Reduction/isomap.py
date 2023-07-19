#!/usr/bin/env python
# coding: utf-8

# In[57]:


#import seaborn as sea
import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import math
from scipy import sparse
import pandas as pd
import networkx as nx
from sklearn.utils.graph_shortest_path import graph_shortest_path



class isomap():
    def __init__(self):
        file_path = input("Paste your isomap.mat filepath:")
        data = spio.loadmat(file_path.replace('"',""))["images"]
        distance = input("Input distance:")
        self.distance = int(distance)
        self.df = data.copy()
        self.data = data.copy()
        self.main()
        
    def adjacency_matrix(self):
        '''
            Find neighbors N(i) of each data point within distance 'e' and let A be an 
            Adjacency matrix: records neighbor Euclidean distances.
            
            
        '''
        self.A = pairwise_distances(self.data.T,self.data.T,metric='l2')
        self.m,self.n = self.A.shape
        self.A[self.A>self.distance] = 0
        sA = sparse.csr_matrix(self.A)
        G = nx.from_scipy_sparse_array(sA)
        self.adjacency_matrix = nx.to_numpy_array(G)
        
    def shortest_path_matrix(self):
        '''
            Find shortest path distance matrix D between each pair of points, x_i and x_j in A.
            Uses Floyd-Warshall algorithm to compute shortest pathh problem
            
        '''
        self.D = graph_shortest_path(self.A, method = 'FW')
        
    def centering_matrix(self):
        '''
            Find centering matrix H = I - (1/m)*(11.T). 
            Use to solve for entrywise square of distance matrix:
                C = (-1/2)*H*[D^2]*H
            
        '''
        H = np.eye(self.m) - (1/self.m)*np.ones((self.m, self.m))*11
        self.K = (-1/(2*self.m)) * H.dot(self.D).dot(H)
    
    
    def leading_eigen(self):
        '''
            Computes leading eigenvectors and eigenvalues of C
            
        '''
        self.vals, self.vecs = np.linalg.eigh(self.K)
        self.dim1 = self.vecs[:,-1] *math.sqrt(self.vals[-1])
        self.dim2 = self.vecs[:,-2]*math.sqrt(self.vals[-2])

        
    def plot(self):
        fig = plt.figure()
        fig.set_size_inches(10, 10)
        ax = fig.add_subplot(111)

        ax.set_title('Isomap')
        Z = pd.DataFrame(self.dim1,self.dim2).reset_index()
        Z = Z.rename(columns= {'index':'x',0:"y"})

        # Adds images onto plot
        for i in range(40):
            img = np.reshape(np.asarray(self.data[:,i].T), (64,64)).T
            ax.imshow(img, aspect='auto',interpolation='nearest', zorder=100000, extent=(Z.loc[i, 'x']-0.004, Z.loc[i, 'x']+0.004, Z.loc[i, 'y']-0.004, Z.loc[i, 'y']+0.004))

        plt.scatter(self.dim1,self. dim2)
        ax.set_ylabel('Pose 1')
        ax.set_xlabel('Pose 2')

        plt.show()
    
    def main(self):
        self.adjacency_matrix()
        self.shortest_path_matrix()
        self.centering_matrix()
        self.leading_eigen()
        self.plot()

