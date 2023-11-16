"""
Authors: Berfin Kavşut
         Mert Ertuğrul
"""

import NeuralNetwork, utilities, PCA

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def grid_search_neural_network(add_PCA = True, display_search_logs=False):
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
        
    X_test = D_test[:, :-1]
    Y_test = D_test[:, -1]
    
    #adding PCA
    if add_PCA:
        pca_obj = PCA.PCA_maker(X= D_train[:, :-1], n_components=8)
        PCA_features_train = pca_obj.apply_pca(X=D_train[:, :-1], display=False)
    
        D_train = np.hstack( ( D_train[:, :-1], PCA_features_train, D_train[:, -1].reshape(D_train.shape[0],1) ) )
        #print(D_train.shape)
        
        PCA_features_test = pca_obj.apply_pca(X=X_test, display=False)
        X_test = np.hstack( ( X_test, PCA_features_test) )
    
    #number of folds for k fold cross validation
    k_fold=5
    
    #parameter grid for grid search
    num_epochs_list = [2,5,10,25]
    learning_rate_list = [0.01,0.1, 0.3,0.5,0.7]
    
    #batch_size = 0 -> batch grad decent
    #batch_size = 1 -> stochastic grad decent
    #batch_size = other -> mini batch grad decent
    mini_batch_size_list = [0,1,5,10,20,50]
    num_hidden_layers_list = [0,1,2,3]

    #Preparing the training data for cross validation
    X_folds,Y_folds = utilities.k_fold_split( D_train, k_fold )
    
    best_params = ()
    best_accuracy = 0
    
    for num_epochs in num_epochs_list: 
        for learning_rate in learning_rate_list:  
            for hidden_layers in num_hidden_layers_list:
                for mini_batch_size in mini_batch_size_list:
        
                    cv_accuracy = []
    
                    for i in range(k_fold):
    
                        #creating training data by omitting the ith fold
                        X_train = np.concatenate( X_folds[:i] + X_folds[i+1:] , axis=0 )
                        Y_train = np.concatenate( Y_folds[:i] + Y_folds[i+1:] , axis=0 )
    
                        #ith fold is the test data
                        X_val = X_folds[i]
                        Y_val = Y_folds[i]
    
                        newNetwork = NeuralNetwork.NeuralNetwork(learning_rate)
    
                        newNetwork.form_architecture(input_dim = X_train.shape[1], hidden_layers = hidden_layers)
    
                        #training the network
                        newNetwork.train(X_train,Y_train,num_epochs, False, mini_batch_size)
    
                        #validating with ith fold
                        val_accuracy,_ = newNetwork.test(X_val,Y_val)
                        cv_accuracy.append( val_accuracy )
    
                    #overall average test accuracy
                    avg_cv_accuracy = np.mean(cv_accuracy)
    
                    if avg_cv_accuracy > best_accuracy:
                        best_accuracy = avg_cv_accuracy
                        best_params = (num_epochs,learning_rate,hidden_layers,mini_batch_size)
                        
                        if display_search_logs:
                            print("New Best Accuracy of:",best_accuracy)
                            print("New Best Params:    Epochs:",num_epochs,"  LR:",learning_rate,"  HL:",hidden_layers,"  MBS:",mini_batch_size)
                            print("-----------------------------------------------------------------------")
       
    print("\n*******************************************************\n")  
    print("Final CV Accuracy: ",best_accuracy)
    print("Final Params:    Epochs:", best_params[0], "  LR:",best_params[1], "  HL:", best_params[2], "  MBS:", best_params[3])
    
    print("Now we test the model using the best parameter set.")
          
    X_train = D_train[:, :-1]
    Y_train = D_train[:, -1]
    
    newNetwork = NeuralNetwork.NeuralNetwork(learning_rate=best_params[1])
    
    newNetwork.form_architecture(input_dim = X_train.shape[1], hidden_layers = best_params[2])
    
    #training the network and displaying graphs
    newNetwork.train(X_train,Y_train,best_params[0], True, best_params[3], X_val = X_test,  Y_val = Y_test)
    
    #testing
    test_accuracy, Y_predicted = newNetwork.test(X_test,Y_test)

    print("---- Neural Network Test Results ----")
    utilities.get_metrics(Y_test, np.squeeze(Y_predicted), print_metrics=True)

    
if __name__ == '__main__':
  
  print("---Without adding PCA---")
  grid_search_neural_network(add_PCA = False)
  print("---With PCA---")
  grid_search_neural_network(add_PCA = True)