#!/usr/bin/env python
# coding: utf-8

# In[47]:


import random
import cv2
import numpy as np
import matplotlib.pyplot as plt


class spatial_filters():
    def __init__(self,image_path):
        self.image = cv2.imread(image_path)
    
    def grayscale(self):
        r,g,b = self.image[:,:,0],self.image[:,:,1],self.image[:,:,2]
        self.image = 0.2989*r + 0.587*g + 0.114*b                              #LUMA coding to RGB
        #self.image = self.image@np.array([0.2989, 0.5870, 0.1140])
        self.m, self.n = self.image.shape

    def gaussian_noise(self):
        gaussian_noise  = np.random.normal(0,0.1,size=self.image.shape)
        maxx = np.max(self.image)

        gaussian = self.image + (maxx*gaussian_noise)

        for i in range(len(gaussian)):
            for j in range(len(gaussian[i])):
                if gaussian[i][j] > 255:
                    gaussian[i][j] = 255
                elif gaussian[i][j] < 0:
                    gaussian[i][j] = 0

        self.image = gaussian.copy()


    def salt_pepper_noise(self):
        pixels = Ig.reshape(1,-1)
        sp_num = 0.05*324000
        white  = [0]*(int(sp_num/2))
        black = [255]*(int(sp_num/2))

        new_pixels = pixels[0].copy()
        for i in range(int(sp_num/2)):
            new_pixels[random.randint(0,324000)] = 0
            new_pixels[random.randint(0,324000)] = 255
    
    
        self.image = np.array(new_pixels).reshape(450,720)


    def median_filter(self, kernel = 3):
        """ 
        Applies Median filter to image. 
        
        1. Defined as the median of all pixels within a local region of an image.
        2. The median filter is normally used to reduce salt and pepper noise in an image.
        3. Does a better job than the mean filter of preserving useful detail in the image.
        
        """
        rows,cols = self.image.shape[:2]
        median_ = np.zeros((rows,cols))
    
        for i in range(1,rows-1):
            for j in range(1,cols-1):
                neighborhood_med = float(np.median(self.image[i-1:i+(kernel - 1),j-1:j+(kernel - 1)]))
                median_[i,j] = neighborhood_med
        
        return median_
    

    def harmonic(self, kernel = 3):
        """ 
        Applies Harmonic mean filter to image. 
        
        1. In the harmonic mean method, the color value of each pixel 
           is replaced with the harmonic mean of color values of 
           the pixels in a surrounding region.
        2. The harmonic mean filter is better at removing Gaussian type noise 
           and preserving edge features than the arithmetic mean filter. 
        3. The harmonic mean filter is very good at removing positive outliers.
        
        """
        rows,cols = self.image.shape[:2]
        harmonic = np.zeros((rows,cols))
        
        for i in range(1,rows-1):
            for j in range(1,cols-1):
                
                neighborhood = self.image[i-1:i+(kernel - 1),j-1:j+(kernel - 1)]
                num = rows*cols
                denum = np.sum(1/neighborhood)
                ans = num/denum
                harmonic[i,j] = ans
        
        return harmonic

    def contra_harmonic(self,kernel = 3,Q=0):
        """ 
        Applies Contraharmonic mean filter to image. 
        
        1. With a contraharmonic mean filter, the color value of each pixel 
           is replaced with the contraharmonic mean of color values of the 
           pixels in a surrounding region.
        2. Reduces or virtually eliminates the effects of salt-and-pepper noise.
           For positive values of Q, the filter eliminates pepper noise. 
           For negative values of Q it eliminates salt noise. 
           It cannot do both simultaneously.
        3. Arithmetic mean filter if Q = 0.
        
        """
        rows,cols = self.image.shape[:2]
        contra_harmonic = np.zeros((rows,cols))
        
        for i in range(1,rows-1):
            for j in range(1,cols-1):
                
                neighborhood = self.image[i-1:i+(kernel - 1),j-1:j+(kernel - 1)]
                num = neighborhood**(Q+1)
                denum = np.int64(neighborhood)**Q
                ans = np.sum(num)/np.sum(denum)
                contra_harmonic[i,j] = ans
        
        return contra_harmonic
    
    
    def arithmetic_mean(self,kernel = 3):
        """ 
        Applies Arithmetic mean filter to image. 
        
        1. An arithmetic mean filter operation on an image removes short tailed 
           noise such as uniform and Gaussian type noise from the image 
           at the cost of blurring the image.
        2. The arithmetic mean filter is defined as the average 
           of all pixels within a local region of an image.
        3. The larger the filtering mask becomes the more predominant the 
           blurring becomes and less high spatial frequency detail that 
           remains in the image.
        
        """
        rows,cols = self.image.shape[:2]
        arithmetic_mean = np.zeros((rows,cols))
        
        for i in range(1,rows-1):
            for j in range(1,cols-1):
                
                neighborhood = self.image[i-1:i+(kernel - 1),j-1:j+(kernel - 1)]
                num = neighborhood**1
                denum = np.int64(neighborhood)**0
                ans = np.sum(num)/np.sum(denum)
                arithmetic_mean[i,j] = ans
        
        return arithmetic_mean
    
    
    def geometric_mean(self,kernel = 3):
        """ 
        Applies Geometric mean filter to image. 
        
        1. The color value of each pixel is replaced with the 
           geometric mean of color values of the pixels in a 
           surrounding region.
        2. The geometric mean filter is better at removing Gaussian 
           type noise and preserving edge features than the arithmetic 
           mean filter.
        3. The geometric mean filter is very susceptible to negative outliers.
        
        """
        rows,cols = self.image.shape[:2]
        geometric_mean = np.zeros((rows,cols))
        
        for i in range(1,rows-1):
            for j in range(1,cols-1):
                
                neighborhood = self.image[i-1:i+(kernel - 1),j-1:j+(kernel - 1)]
                num = neighborhood**1
                root = np.int64(neighborhood)**0
                ans = np.prod(num)**(1/np.sum(root))
                geometric_mean[i,j] = ans
        
        return geometric_mean
    

    def min_filter(self,kernel = 3):
        """ 
        Applies Minimum filter to image. 
        
        1. The minimum filter is defined as the minimum of all pixels 
           within a local region of an image.
        2. Typically applied to an image to remove positive outlier noise.
        
        """
        rows,cols = self.image.shape[:2]
        min_filter = np.zeros((rows,cols))
        
        for i in range(1,rows-1):
            for j in range(1,cols-1):
                neighborhood_min = self.image[i-1:i+(kernel - 1),j-1:j+(kernel - 1)].min()
                min_filter[i,j] = neighborhood_min
        
        return min_filter
    
    def max_filter(self, kernel = 3):
        """ 
        Applies Maximum filter to image. 
        
        1. The maximum filter is defined as the maximum of all pixels 
           within a local region of an image.
        2. Typically applied to an image to remove negative outlier noise.
        
        """
        rows,cols = self.image.shape[:2]
        max_filter = np.zeros((rows,cols))
        
        for i in range(1,rows-1):
            for j in range(1,cols-1):
                neighborhood_max = self.image[i-1:i+(kernel - 1),j-1:j+(kernel - 1)].max()
                max_filter[i,j] = neighborhood_max
        
        return max_filter

    def midpoint_filter(self, kernel = 3):
        """ 
        Applies Midpoint filter to image. 
        
        1. In the midpoint method, the color value of each pixel is 
           replaced with the average of maximum and minimum 
           (i.e. the midpoint) of color values of the pixels in 
           a surrounding region. 
        2. Typically applied to to filter images containing short tailed 
           noise such as Gaussian and uniform type noise.
        
        """
        rows,cols = self.image.shape[:2]
        mid_filter = np.zeros((rows,cols))
        
        for i in range(1,rows-1):
            for j in range(1,cols-1):
                neighborhood_mid = 0.5*(self.image[i-1:i+(kernel - 1),j-1:j+(kernel - 1)].max()+self.image[i-1:i+(kernel - 1),j-1:j+(kernel - 1)].min())
                mid_filter[i,j] = neighborhood_mid
        
        return mid_filter
    
    
    def return_image(self):
        return self.image

