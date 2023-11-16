"""
Authors: Berfin Kavşut
         Mert Ertuğrul
"""

import SVM, utilities, PCA

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math


def grid_search_svm(add_PCA = True, display_search_logs=False):
    #reading the data
    data_heart = pd.read_csv('./heart.csv')
    D = data_heart.values
    
    #train test split
    D_train, D_test = utilities.split_train_test(D)  

    #standardizing and shuffling the data 
    standardizer = utilities.Standardizer()
    
    #train data
    D_train = standardizer.standardize_data(D_train)
    
    #test data - statistical proeprties of test data do not leak into standardization parameters
    D_test = standardizer.standardize_data(D_test, testing=True)
    X_test = D_test[:, :-1]
    Y_test = D_test[:, -1]
    
    #adding PCA
    if add_PCA:
        pca_obj = PCA.PCA_maker(X= D_train[:, :-1], n_components=8)
        PCA_features_train = pca_obj.apply_pca(X=D_train[:, :-1], display=False)
    
        D_train = np.hstack( ( D_train[:, :-1], PCA_features_train, D_train[:, -1].reshape(D_train.shape[0],1) ) )
        
        PCA_features_test = pca_obj.apply_pca(X=X_test, display=False)
        X_test = np.hstack( ( X_test, PCA_features_test) )
    
    #number of folds for k fold cross validation
    k_fold=5
    
    #parameter grid for grid search
    num_epochs_list = [10,25,50,75]
    learning_rate_list = [0.0001,0.001,0.01, 0.1]
    reg_param_list = [0.1, 0.5, 1, 2, 5]

    #preparing the training data for cross validation
    X_folds,Y_folds = utilities.k_fold_split( D_train, k_fold )
    
    best_params = ()
    best_accuracy = 0
    
    for num_epochs in num_epochs_list: 
        for learning_rate in learning_rate_list:  
            for  reg_param in reg_param_list:
                    
                cv_accuracy = []
                for i in range(k_fold):

                    #creating training data by omitting the ith fold
                    X_train = np.concatenate( X_folds[:i] + X_folds[i+1:] , axis=0 )
                    Y_train = np.concatenate( Y_folds[:i] + Y_folds[i+1:] , axis=0 )

                    #ith fold is the test data
                    X_val = X_folds[i]
                    Y_val = Y_folds[i]

                    svm_model = SVM.svm(reg_param=reg_param)

                    #training the network
                    svm_model.fit(X_train,Y_train,num_epochs=num_epochs,
                                    learning_rate=learning_rate, show_graph=False)

                    #validating with ith fold
                    val_accuracy,_ = svm_model.score( X_val, Y_val)
                    cv_accuracy.append( val_accuracy )

                #overall average test accuracy
                avg_cv_accuracy = np.mean(cv_accuracy)

                if avg_cv_accuracy > best_accuracy:
                    best_accuracy = avg_cv_accuracy
                    best_params = (num_epochs,learning_rate,reg_param)
                    
                    if display_search_logs:
                        print("New Best Accuracy of:",best_accuracy)
                        print("New Best Params:    Epochs:",num_epochs,"  LR:",learning_rate,"  REG_PARAM:",reg_param)
                        print("-----------------------------------------------------------------------")
    
    print("\n*******************************************************\n")  
    print("Final CV Accuracy: ",best_accuracy)
    print("Final Params:    Epochs:",best_params[0],"  LR:",best_params[1],"  REG_PARAM:",best_params[2])
    
    #TESTING
    print("Now we test the model using the best parameter set.")
          
    X_train = D_train[:, :-1]
    Y_train = D_train[:, -1]
    
    svm_model = SVM.svm(reg_param=best_params[2])

    #training the SVM
    svm_model.fit(X_train,Y_train,num_epochs=best_params[0],
                  learning_rate=best_params[1], show_graph=True, X_val = X_test, Y_val = Y_test)
    
    #testing
    _, Y_predicted = svm_model.score( X_test, Y_test)

    print("---- SVM Test Results ----")
    utilities.get_metrics(Y_test, Y_predicted, print_metrics=True)

    
if __name__ == '__main__':
  print("---Without adding PCA---")
  grid_search_svm(add_PCA = False)
  print("---With PCA---")
  grid_search_svm(add_PCA = True)