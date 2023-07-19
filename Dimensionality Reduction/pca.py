#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import math
import pandas as pd
import scipy.sparse.linalg as ll
import matplotlib.pyplot as plt

class PCA():
    def __init__(self):
        file_path = input("Paste your csv filepath:")
        data = pd.read_csv(file_path.replace('"',""))
        self.df = data
        self.data = data.dropna().to_numpy()
        self.main()
        
    def attributes(self):
        #Extract Attributes from Data
        self.m,self.n = self.data.shape
        self.attributes = self.data[:,1:]
        
        #Create Indicator Matrix
        self.ind_matrix = self.data[:,0]
        
    def normalization(self):
        #Normalize
        stdA = np.std(self.attributes.astype(float),axis = 0)
        self.attributes = self.attributes @ np.diag(np.ones(stdA.shape[0])/stdA[0])
        self.attributes = self.attributes.T
        
    def pca(self):
        #PCA
        mu = np.mean(self.attributes,axis = 1)
        xc = self.attributes - mu[:,None]
        C = np.dot(xc,xc.T)/self.m
        K = 4
        self.S,self.W = ll.eigs(C.astype(float),k = K)

        self.dim1 = np.dot(self.W[:,0].T,xc)/math.sqrt(abs(self.S[0]))
        self.dim2 = np.dot(self.W[:,1].T,xc)/math.sqrt(abs(self.S[1]))
        
        
    def plot(self):
        plt.figure(figsize=(10,10))
        plt.plot(self.W[:,0],self.W[:,1],'o')
        
        foods = self.df.reset_index().columns[2:]
        for i in range(0,foods.shape[0]):
            plt.text(self.W[i,0]-.02,self.W[i,1]+.005, foods[i])
    
        plt.figure(figsize=(10,10))
        plt.plot(abs(self.dim1),abs(self.dim2),'ro')

        countries = self.df.reset_index()['Country']
        for i in range(0,countries.shape[0]):
            plt.text(abs(self.dim1[i])-0.06,abs(self.dim2[i])+0.01, countries[i])
    
    def main(self):
        self.attributes()
        self.normalization()
        self.pca()
        self.plot()


# In[24]:


import numpy as np
import math
import pandas as pd
import scipy.sparse.linalg as ll
import matplotlib.pyplot as plt
import scipy.io as spio

class PCA_isomap_dat():
    def __init__(self):
        file_path = input("Paste your isomap.mat filepath:")
        data = spio.loadmat(file_path.replace('"',""))["images"]
        self.df = data.copy()
        self.data = data.copy()
        self.main()
        
    def attributes(self):
        #Extract Attributes from Data
        self.m,self.n = self.data.shape
        self.attributes = self.data.copy()
        
        #Create Indicator Matrix
        self.ind_matrix = self.data[:,0]
        
    def normalization(self):
        #Normalize
        stdA = np.std(self.attributes.astype(float),axis = 0)
        self.attributes = self.attributes @ np.diag(np.ones(stdA.shape[0])/stdA[0])
        self.attributes = self.attributes.T
        
    def pca(self):
        #PCA
        mu = np.mean(self.attributes,axis = 1)
        xc = self.attributes - mu[:,None]
        C = np.dot(xc,xc.T)/self.m
        K = 4
        self.S,self.W = ll.eigs(C.astype(float),k = K)

        self.dim1 = np.dot(self.W[:,0].T,xc)/math.sqrt(abs(self.S[0]))
        self.dim2 = np.dot(self.W[:,1].T,xc)/math.sqrt(abs(self.S[1]))
        
        
    def plot(self):
        Z = pd.DataFrame(self.W[:,1],self.W[:,0]).reset_index()
        Z = Z.rename(columns= {'index':'x',0:"y"})

        fig = plt.figure()
        fig.set_size_inches(10, 10)
        ax = fig.add_subplot(111)

        ax.set_title('PCA')

        for i in range(80):
            img = np.reshape(np.asarray(self.data[:,i].T), (64,64)).T
            ax.imshow(img, aspect='auto',interpolation='nearest', zorder=100000, extent=(Z.loc[i, 'x']-0.002, Z.loc[i, 'x']+0.002, Z.loc[i, 'y']-0.004, Z.loc[i, 'y']+0.004))

        ax.scatter(Z['x'],Z['y'])
        ax.set_ylabel('Pose 1')
        ax.set_xlabel('Pose 2')

        plt.show()
    
    def main(self):
        self.attributes()
        self.normalization()
        self.pca()
        self.plot()

