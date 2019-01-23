#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 14:21:38 2019

@author: veeraballi
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
  
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def createClusters(list_of_centers,data):
    no_of_clusters = len(list_of_centers)
    clusters = []
    list_of_centers = np.asarray(list_of_centers)
    
    #Run K-Means
    kmeans = KMeans(n_clusters=no_of_clusters, random_state = 3, init = list_of_centers , n_init = 1)
    cluster_labels = kmeans.fit_predict(data)
    
    # create a new df by joining column wise old df and cluster_labels
    new_df = pd.DataFrame(data)
    new_df['cluster_labels'] = pd.Series(cluster_labels)
    
    # filter for each cluster_label and add to a list and return list
    for each_label in range(no_of_clusters):
        cluster = new_df[new_df['cluster_labels'] == each_label] 
        cluster.drop(['cluster_labels'],axis = 1, inplace = True)
        clusters.append(cluster.values)
    return clusters,kmeans.cluster_centers_.tolist(),cluster_labels

def isGaussian(cluster,centroid):
    if cluster.shape[0] < 10:
        return None
    critical_val = 1.8692
    pca = PCA(n_components=3)
    pca.fit(cluster)
    main_pc = pca.components_[:,0]  # this is s
    # extracting lambda(eigen val) from pca.explained_variance_ i.e 2.93433815
    lambda_val =  pca.explained_variance_[0] 
    m = np.sqrt((2 * lambda_val) / np.pi) * main_pc
    
    child_center1 = centroid + m
    child_center2 = centroid - m
    list_of_centers = np.asarray([child_center1,child_center2])
    child_clusters, child_centers, labels = createClusters(list_of_centers,cluster)
    
    # Project all points in the  cluster onto child 1 - child 2 line i.e onto vector
    vector = np.array(child_centers[0]) - np.array(child_centers[1])
    cluster_prime = np.inner(vector, cluster) / np.linalg.norm(vector, ord = 2)
    
    # Apply anderson darling statistic for gaussiun check
    
    A2, critical, sig = sp.stats.anderson(cluster_prime)
    print("--------------------------------------------------------")
    print("======",A2)
    print("======",critical)
    print("======",sig)
    print("--------------------------------------------------------")
    
    if A2 < critical_val:
        return None
    else:
        return child_centers
    
def plot_clusters(data,cluster_centers,labels):
    filename = "{}.png".format(len(cluster_centers))
    cluster_centers = np.asarray(cluster_centers)
    plt.scatter(data[:,0], data[:,1], c=labels, cmap='rainbow', s= 7)
    plt.scatter(cluster_centers[:,0], cluster_centers[:,1], c='black')
    plt.savefig(filename)
    
if __name__ == '__main__':
    data_frame = pd.read_csv("/home/veeraballi/Desktop/Study/G-means/data.csv")
    scalar = StandardScaler()
    scaled_values = scalar.fit_transform(data_frame.values)   
    # Run k-means once to pick a center for the first cluster
    kmeans = KMeans(n_clusters=1, init= "k-means++", random_state = 3)
    kmeans.fit(scaled_values)
    list_of_centers = kmeans.cluster_centers_.tolist()
    cluster_centers = list_of_centers
    # intialize list_of_center to average of all 720 points in a data frame
    while(True):
        change = True
        clusters,cluster_centers,labels = createClusters(cluster_centers,scaled_values)
        plot_clusters(scaled_values,cluster_centers,labels)
        for cluster,cluster_center in zip(clusters,cluster_centers):
            new_centers = isGaussian(cluster,cluster_center)
            if(new_centers):
                print cluster_centers
                print cluster_center
                cluster_centers.remove(cluster_center)
                cluster_centers.extend(new_centers)
                change = False
                break
            else:
                continue  
        if change:
            
            break
    print len(cluster_centers)
         