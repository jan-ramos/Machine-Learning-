#!/usr/bin/env python
# coding: utf-8

# In[46]:


from skimage import io

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from random import sample
import random
import time as time


class KMeans():

    def __init__(self):
        file_path = input("Paste your .jpg image filepath:")
        self.image = io.imread(file_path.replace("/","\\").replace('"',""))
        self.rows = self.image.shape[0]
        self.cols = self.image.shape[1]
        self.data = self.image.reshape(self.rows*self.cols, 3)
        
        K = input("Input number of clusters: ")
        self.K = int(K)
        self.iterations = 1
	self.main()
        
        
    def random_assignment(self):
        centroids = random.sample(list(self.data),self.K)

        return centroids
    
    def new_centroids(self,clusters,centroids):
        for i in range(self.K):
            centroids[i] = np.median(self.data[np.where(clusters == i)],axis= 0)
            
        return centroids    

    def clusters_assignment(self,centroids):
        clusters =  np.zeros((np.size(self.data,0),1))
        total_distance = np.empty((np.size(self.data,0),1))
        
        for i in range(self.K):
            cent_vectors = np.ones((np.size(self.data,0),1))*centroids[i]
            distance = np.sum(np.power(np.subtract(self.data,cent_vectors),2),axis=1)
            distance.resize((np.size(self.data,0),1))
            total_distance = np.append(total_distance,distance,axis=1)
    
        clusters = np.argmin(np.delete(total_distance,0,axis=1),axis=1)
        
        return clusters
    
    def classify(self, centroids, clusters):
        new_classifier = self.data.copy()
        
        for i in enumerate(clusters):
                new_classifier[i[0]] = centroids[i[1]]
        
        return new_classifier
       
    def main(self):
        centroids = self.random_assignment()
        classification = [0]*len(self.data)
        start_time = time.time()
        
        while self.iterations < 100:
            new_class = self.clusters_assignment(centroids)
            if (classification == new_class).all()==True:
                break
            
            classification = new_class
            centroids = self.new_centroids(classification,centroids)
            re_classify = self.classify(centroids,classification)
            self.iterations += 1 
        
        end_time = time.time()
        timed = end_time - start_time
        
    
        image1 = np.reshape(re_classify, (self.rows, self.cols,self.image.shape[2]))
        print("Number of iterations until Convergence: " + str(self.iterations))
        print("Time Elapsed: " + str(timed) + " seconds")
        plt.imshow(image1)
        plt.axis('off')
        plt.show()

