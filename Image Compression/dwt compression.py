#!/usr/bin/env python
# coding: utf-8

# In[41]:


import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.linalg import dft
import pywt
from sklearn.linear_model import OrthogonalMatchingPursuit

class dwt_compression():
    def __init__(self,image_path,compression_level):
        self.image = plt.imread(image_path)
        self.compression_level = compression_level
        
        
    def grayscale(self):
        r,g,b = self.image[:,:,0],self.image[:,:,1],self.image[:,:,2]
        self.image = 0.2989*r + 0.587*g + 0.114*b                              #LUMA coding to RGB
        self.m, self.n = self.image.shape
        
    def compress(self):
        K = np.round(self.compression_level * self.m).astype(int)
        I = np.eye(self.n)
        self.R = np.random.normal(size =(K,self.m))
        self.y = self.R@self.image
        
    def dwt(self,N):
        g, h = pywt.Wavelet('sym8').filter_bank[:2]
        L = len(h)  # Length of bandwidth
        rank_max = int(np.log2(N))  # Maximum Layer
        rank_min = int(np.log2(L))+1  # Minimum Layes
        ww = np.eye(2**rank_max)  # Proprocessing Matrix

        for jj in range(rank_min, rank_max+1):
            nn = 2**jj
            # Construct vector
            p1_0 = np.concatenate([g, np.zeros(nn-L)])
            p2_0 = np.concatenate([h, np.zeros(nn-L)])
            p1 = []
            p2 = []
            # Circular move
            for ii in range(2**(jj-1)):
                shift = 2*ii
                p1.append(np.roll(p1_0, shift))
                p2.append(np.roll(p2_0, shift))
            p1 = np.stack(p1)
            p2 = np.stack(p2)
            # Orthogonal Matrix
            w1 = np.concatenate([p1, p2])
            wL = len(w1)
            w = np.eye(2**rank_max)
            w[:wL, :wL] = w1
            ww = ww@w
        return ww 
    
    def dwt_transform(self):
        self.ww = self.dwt(self.n)
        
        #  Measure value
        self.y = self.y@self.ww.T

        # Measure Matricompressed_image
        self.R = self.R@self.ww.T

        self.reg = OrthogonalMatchingPursuit(n_nonzero_coefs=256,tol=1e-5, fit_intercept=False, normalize=False)
        self.compressed_image2 = np.zeros((self.m,self.n))
        for i in range(self.n):
            self.reg.fit(self.R, self.y[:, i])
            self.compressed_image2[:, i] = self.reg.coef_
            
    def plot(self):
        # original Image
        plt.figure()
        plt.imshow(self.image, cmap='gray')
        plt.title('original Image')
        plt.show()
        
        # Transfered Image
        plt.figure()
        plt.imshow(np.clip(self.compressed_image2, 0, 255).astype('uint8'), cmap='gray')
        plt.title('Transferred Image')
        plt.show()
        
        # Recovered image
        plt.figure()
        compressed_image3 = self.ww.T@self.compressed_image2@self.ww  # inverse DWT
        plt.imshow(np.clip(compressed_image3, 0, 255).astype('uint8'), cmap='gray')
        plt.title('Recovered Image')
        plt.show()
        
    def main(self):
        self.grayscale()
        self.compress()
        self.dwt_transform()
        self.plot()

