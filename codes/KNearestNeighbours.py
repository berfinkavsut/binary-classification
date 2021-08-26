"""
Authors: Berfin Kavşut -  21602459
         Mert Ertuğrul - 21703957
"""

import numpy as np
from scipy.stats import mode
import utilities, PCA
import pandas as pd
import matplotlib.pyplot as plt

#Distance Metrics

def euclidian_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

def cosine_distance(x1,x2):
	return 1 - np.dot(x1,x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

def manhattan_distance(x1, x2):
    return np.abs(x1 - x2).sum()


def KNN_classify(x_train, y_train , x_test, k, distance = "euclidian"):
    
    """
    K Nearest Neighbours algorithm computes the data point's distance to other 
    data points accoridng to a given metric in the feauture space and selects the
    majority vote label of k nearest neighbours.
    
    x_train: training set features
    y_train: training set labels
    x_test: test set features
    k: number of nearest neighbors to vote for label
    distance: the distance metric to be used
    """
    
    result_labels = []
     
    #Loop through the datapoints to be classified
    for item in x_test: 
         
        #distances to training data points
        distances = []
         
        for i in range(len(x_train)): 
            #choose distance metric
            if distance == "euclidian":
              new_dist = euclidian_distance(np.array(x_train[i,:]) , item) 
            elif distance == "cosine":
              new_dist = cosine_distance(np.array(x_train[i,:]) , item) 
            else:
              new_dist = manhattan_distance(np.array(x_train[i,:]) , item) 
            distances.append(new_dist) 

        #sorts distances and returns their original indices, takes the last k distances
        indices = np.argsort( np.array(distances) )[:k] 
        labels = y_train[indices]
         
        #most common label selected
        label = mode(labels) 
        label = label.mode[0]
        result_labels.append(label)
 
    return result_labels
        
        
#standalone main function to demonstrate the model's k-fold cross validation results on graphs     
def main_knn_grid_search(add_PCA = True):
    np.random.seed(1)
    
    
    
    #reading the data into a pandas dataframe and converting to numpy array
    data_heart = pd.read_csv('./heart.csv')
    
    D = data_heart.values
    
    #train test split
    D_train, D_test = utilities.split_train_test(D)  

    #standardizing and shuffling the data 
    standardizer = utilities.Standardizer()
    #train data
    
    D_train = standardizer.standardize_data( D_train )
    #test data - statistical proeprties of test data do not leak into standardization parameters
    D_test = standardizer.standardize_data( D_test, testing=True )
    
    #----------- adding PCA
    if add_PCA:
        pca_obj = PCA.PCA_maker(X= D_train[:, :-1], n_components=8)
        PCA_features_train = pca_obj.apply_pca(X=D_train[:, :-1], display=False)
    
        D_train = np.hstack( ( D_train[:, :-1], PCA_features_train, D_train[:, -1].reshape(D_train.shape[0],1) ) )
        #print(D_train.shape)
        
        PCA_features_test = pca_obj.apply_pca(X=D_test[:, :-1], display=False)
        D_test = np.hstack( ( D_test[:, :-1], PCA_features_test, D_test[:, -1].reshape(D_test.shape[0],1) ) )
    
    #number of folds for k fold cross validation
    k_fold=5
    

    #creating the individual folds and seperating the label vector
    X_folds,Y_folds = utilities.k_fold_split( D_train, k_fold )
    
    #range of k values to be tested for k nearest neighbors  
    K_values = [3,5,7,9,11,13,15,19,21,23,25,50,70,100,125,150,175,200,225,250]
    #different distance types to be tested
    distances = ["euclidian", "cosine", "manhattan"]
    

    #keeps accuracy values for each distance metric
    accuracies_per_dist = []
    
    #preparing plot layout
    fig = plt.figure(figsize=(10,10))
    
    ax = fig.add_subplot(2, 1, 1)
    ax.set_xlabel('K value')
    ax.set_ylabel('Cross Validation Accuracy')
    ax.set_ylim([0, 1])
    
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_xlabel('K value')
    ax2.set_ylabel('Cross Validation Accuracy')
    ax2.set_title('Impact of k and Distance Type on Average Accuracy ')
    ax2.set_ylim([0, 1])
    
    max_params= ()
    max_accuracy = 0
    
    #for each distance metric
    for dist_index, dist in enumerate(distances):
        
        #print("For distance type: "+dist)
        
        #keeps accuracy values for each run
        accuracy_data_k = np.zeros( (k_fold,len(K_values) ))
        
        #for each of the folds being used for validation
        for i in range(k_fold):
          #taking all folds other than the ith one for training
          X_train = np.concatenate( X_folds[:i] + X_folds[i+1:] , axis=0 )
          Y_train = np.concatenate( Y_folds[:i] + Y_folds[i+1:] , axis=0 )
          #ith fold is used for validation
          X_val = X_folds[i]
          Y_val = Y_folds[i]
          
          for k_index, k in enumerate(K_values):
        
              labels = KNN_classify(X_train, Y_train , X_val, k, dist)
              #average accuracy accross the validation fold
              accuracy = np.sum(labels == Y_val) / Y_val.shape[0]
              
              accuracy_data_k[i][ k_index] = accuracy
            
              
        #averaging accross folds
        avg_accuracy =  np.mean(accuracy_data_k, axis=0)
        
        #looking for the maximium accuracy
        j = np.argmax(avg_accuracy)
        if avg_accuracy[j]> max_accuracy:
             max_accuracy = avg_accuracy[j]
             max_params = (dist_index,j)
        
        accuracies_per_dist.append(accuracy_data_k)
        ax2.plot(K_values, avg_accuracy, label= dist)
        
    print("Maximum cv accuracy of " + str(max_accuracy) + " found for k="+str(K_values[max_params[1]])+" distance:"+distances[max_params[0]])

    for i in range(k_fold):
        ax.plot(K_values, accuracies_per_dist[ max_params[0] ][i] )          
    ax.plot(K_values, avg_accuracy, label= "average", linewidth=5.0)
    
    ax.set_title('k (knn parameter) vs CV Accuracy Graph for K Folds, Distance:'+distances[max_params[0]])
    ax.legend()
    ax2.legend()
    plt.show()
    
    #fig.savefig('KNN_Plots_2.png')
    
    #----- TESTING ----------------------------
    
    print("Now we test the model using the best parameter set.")
          
    X_train = D_train[:, :-1]
    Y_train = D_train[:, -1]
    
    X_test = D_test[:, :-1]
    Y_test = D_test[:, -1]
    

    #testing
    Y_predicted = KNN_classify(X_train, Y_train , X_test, K_values[max_params[1]], distances[max_params[0]])
    

    print("---- KNN Test Results ----")
    utilities.get_metrics(Y_test, Y_predicted, print_metrics=True)
    

if __name__ == '__main__':
  main_knn_grid_search()

  print("---Without adding PCA---")
  main_knn_grid_search(add_PCA = False)
  print("---With PCA---")
  main_knn_grid_search(add_PCA = True)
