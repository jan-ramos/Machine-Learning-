#!/usr/bin/env python
# coding: utf-8

# In[143]:


import numpy as np
import numpy.matlib
import pandas as pd
from scipy.stats import multivariate_normal as mvn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing

class GMM():
    def __init__(self):
        file_path = input("Paste your data's filepath:")
        self.data = pd.read_csv(file_path.replace("/","\\").replace('"',"")).to_numpy()
       
        self.y = self.data[:,0]
        self.data = self.data[:,1:]
        self.rows = self.data.shape[0]
        self.cols = self.data.shape[1]
                                                       
        K = input("Input number of clusters: ")
        self.K = int(K)
        
        iterations = input("Input max number of iterations: ")
        self.iterations = int(iterations)
                
        pca_ind = input("Perform PCA (Y/N)?: ")
        if (pca_ind == 'Y') | (pca_ind == 'y'):
            n_components = input("Input number of components: ")
            self.n_components = int(n_components)
            self.pca()
            
        self.parameter_set()
    
    def pca(self):
       
        self.data = self.data[:,1:]
        self.ndata = preprocessing.scale(self.data)
        self.rows = self.ndata.shape[0]
        self.cols = self.ndata.shape[1]

        C = np.matmul(self.ndata.T,self.ndata)/self.rows 
        V,G,_ = np.linalg.svd(C)
        V = V[:,:self.n_components]
        self.data = np.dot(self.ndata ,V)

    def parameter_set(self):   
        """Initialize means, weights and variance randomly"""
        self.pi = np.random.random(self.K)   #Initializes Prior
        self.pi = self.pi/np.sum(self.pi)
        
        self.mu = np.random.randn(self.K,self.n_components)   #Initializes Mean
        self.mu_old = self.mu.copy()
        
        self.sigma = []                              #Initializes Covariance
        for i in range(self.K):
            dummy = np.random.randn(self.n_components, self.n_components)
            self.sigma.append(dummy@dummy.T)
        
        self.tau = np.full((self.rows,self.K),fill_value = 0.) #Initialize Posterior

    def e_step(self):
        for j in range(self.K):
            self.tau[:, j] = self.pi[j] * mvn.pdf(self.data, self.mu[j], self.sigma[j])
            
        sum_tau = np.sum(self.tau, axis=1)
        sum_tau.shape = (self.rows,1)    
        self.tau = np.divide(self.tau, np.tile(sum_tau, (1, self.K)))
        
    def m_step(self):
        for kk in range(self.K):
            self.pi[kk] = np.sum(self.tau[:, kk])/self.rows

            # update component mean
            self.mu[kk] = self.data.T @ self.tau[:,kk] / np.sum(self.tau[:,kk], axis = 0)

            # update cov matrix
            dummy = self.data - np.tile(self.mu[kk], (self.rows,1)) # X-mu
            self.sigma[kk] = dummy.T @ np.diag(self.tau[:,kk]) @ dummy / np.sum(self.tau[:,kk], axis = 0)

    def train(self):
        for ii in range(self.iterations):
            self.e_step()
            self.m_step()
            
            print('-----iteration---',ii)    
            plt.scatter(self.data[:,0], self.data[:,1], c= self.tau)
            plt.axis('scaled')
            plt.draw()
            plt.pause(0.1)
            if np.linalg.norm(self.mu-self.mu_old) < 1e-3:
                print('training coverged')
                break
            self.mu_old = self.mu.copy()
            if ii==self.iterations:
                print('max iteration reached')
                break
        return self.data,self.tau

                

