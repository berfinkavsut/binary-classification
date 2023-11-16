"""
Authors: Berfin Kavşut
         Mert Ertuğrul
"""

import utilities

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler


class PCA_maker:
    
    def __init__(self, X, n_components):
        
        self.k = n_components
    
        #normalizing the features wrt training set
        self.standardScaler = StandardScaler().fit(X) 
        X = self.standardScaler.transform(X)
        
        #p x p, p is feature number
        sigma = np.cov(X.T)  
        
        #each column of eigen_vectors is u, eigen_values are lambdas
        eigen_values, eigen_vectors = np.linalg.eig(sigma) 
       
        #now, each row is eigenvector (for sorting)
        eigen_vectors = eigen_vectors.T 

        #sort wrt eigenvalues
        zipped_lists = zip(eigen_values, eigen_vectors)
        sorted_pairs = sorted(zipped_lists, reverse=True)
        tuples = zip(*sorted_pairs)
        eigen_values, eigen_vectors = [ list(tup) for tup in  tuples]
        
        #each row is one eigenvector
        eigen_vectors = np.array(eigen_vectors) 
   
        #saving eigenvectors/principal component loading vectors 
        self.U = eigen_vectors.T 
        
        #Z = np.dot( X, self.U[:,0:self.k] )
        #D = np.diag(eigen_values)
        #error = np.sum(np.square(np.dot(U,D)-np.dot(sigma,U)))
        #print('Diagonalization MSE:', error)
        

    def apply_pca(self, X, display=False):
   
        X = self.standardScaler.transform(X)
        n = X.shape[0] #sample number 
        
        #principal component scores, n x k
        Z = np.dot( X, self.U[:,0:self.k] )
        
        if display == True:
            PVE_k_list = []
            PVE_m_list = []
            PVE_k = 0
            for m in range(self.k):
                total_variance = (1/n)*np.sum(np.square(X)) 
                variance_explained = (1/n)*np.sum(np.square(Z[:,m])) 
    
                PVE_m = variance_explained/total_variance
                PVE_m_list.append(PVE_m)
                PVE_k = PVE_k + PVE_m
                PVE_k_list.append(PVE_k)
            
            x_axis = np.arange(self.k)
    
            fig = plt.figure(figsize=(10,10))
    
            ax = fig.add_subplot(2, 1, 1)
            ax.plot(x_axis, PVE_m_list)
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Explained Variance')
    
            ax2 = fig.add_subplot(2, 1, 2)
            ax2.plot(x_axis, PVE_k_list)
            ax2.set_xlabel('Number of Principal Components')
            ax2.set_ylabel('Cumulative Explained Variance')
    
            ax.set_title('Explained Variance for Each Principal Component')
            ax2.set_title('Cumulative Explained Variance for PCA')
            
            fig.tight_layout(pad=3.0)
    
            plt.savefig('PCA.png')
            plt.show()
    
        return Z
    

def main_pca():
    
    data_heart = pd.read_csv('./heart.csv')
    data = data_heart.values
    
    #split features and label
    X = data[:, :-1]
    X = StandardScaler().fit_transform(X) #normalizing the features
    
    Y = data[:, -1]
    Y = np.reshape(Y,(Y.shape[0],1)) #to concatenate later 
    
    #my model 
    k = 13 #principal component analysis, dimension number 
    
    pca_obj = PCA_maker(X=X, n_components=k)
    principal_component_scores = pca_obj.apply_pca(X=X, display=True)

    #plotting the first two principal components against each other
    plt.figure()
    plt.figure(figsize=(10,10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel('Principal Component - 1',fontsize=20)
    plt.ylabel('Principal Component - 2',fontsize=20)
    plt.title("Principal Component Analysis of Heart Disease Dataset",fontsize=20)
    targets = [0, 1] #benign, malignant 
    colors = ['r', 'g']
    
    for target, color in zip(targets,colors):
        indicesToKeep = data_heart['target'] == target
        plt.scatter(principal_component_scores[indicesToKeep, 0],
                    principal_component_scores[indicesToKeep, 1], c = color, s = 50)
    
    plt.legend(targets,prop={'size': 15})
    
    
if __name__ == '__main__':
  main_pca()