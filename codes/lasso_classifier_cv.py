"""
Authors: Berfin Kavşut -  21602459
         Mert Ertuğrul - 21703957
"""


import Lasso, utilities, PCA

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

"""
Dataset is seperated as 90% training and cross validation set
                        10% test set
                        
This class carries out grid search for the parameters of Lasso using cross-validation.
Afterwards, the best performing parameter set is used to train on the model on the entire training set
and test accuracy is obtained using the test set.

"""
feature_names = [ "age",
   "sex",
   "cp",
   "trestbps",
   "chol",
   "fbs",
   "restecg",
   "thalach",
   "exang",
   "oldpeak",
   "slope",
   "ca",
   "thal",
   "PC1","PC2","PC3","PC4","PC5","PC6","PC7","PC8"]

def main_lasso_classifier_cv(add_PCA = True, display_search_logs=False):
    
    np.random.seed(1)

    #reading the data into a pandas dataframe and converting to numpy array
    data_heart = pd.read_csv('./heart.csv')
    D = data_heart.values

    #train test split
    D_train, D_test = utilities.split_train_test(D)

    #standardizing and shuffling the data 
    standardizer = utilities.Standardizer()
    
    #cross validation and training data
    D_train = standardizer.standardize_data( D_train )
    #test data - statistical proeprties of test data do not leak into standardization parameters
    D_test = standardizer.standardize_data( D_test, testing=True )
    
    X_test = D_test[:, :-1]
    Y_test = D_test[:, -1]
    Y_test = np.where(Y_test == 0, -1, Y_test)
    
    #----------- adding PCA
    if add_PCA:
        pca_obj = PCA.PCA_maker(X= D_train[:, :-1], n_components=8)
        PCA_features_train = pca_obj.apply_pca(X=D_train[:, :-1], display=False)
    
        D_train = np.hstack( ( D_train[:, :-1], PCA_features_train, D_train[:, -1].reshape(D_train.shape[0],1) ) )
        #print(D_train.shape)
        
        PCA_features_test = pca_obj.apply_pca(X=X_test, display=False)
        X_test = np.hstack( ( X_test, PCA_features_test) )
    
        
    #------------------- CROSS VALIDATION -----------------------------------------
    k_fold = 5
    
    #Preparing the training data for cross validation
    X_folds,Y_folds = utilities.k_fold_split( D_train, k_fold )

    #parameter grid
    num_epochs_list = [1,2,3,4,5,10]
    learning_rate_list = [0.0001, 0.001, 0.005,0.01, 0.05, 0.1,]
    lambda_list = [0,0.001,0.01,0.1]
    
    best_params = ()
    best_accuracy = 0
    
    for num_epochs in num_epochs_list: 
        for learning_rate in learning_rate_list:  
            for l1_lambda in lambda_list:

                cv_accuracy = []
                cv_weights = []

                for i in range(k_fold):

                    #creating training data by omitting the ith fold
                    X_train = np.concatenate( X_folds[:i] + X_folds[i+1:] , axis=0 )
                    Y_train = np.concatenate( Y_folds[:i] + Y_folds[i+1:] , axis=0 )
                    # change labels of 0 to -1, negative class
                    Y_train = np.where(Y_train == 0, -1, Y_train)

                    #ith fold is the test data
                    X_val = X_folds[i]
                    Y_val = Y_folds[i]

                    # change labels of 0 to -1, negative class
                    Y_val = np.where(Y_val == 0, -1, Y_val)

                    # Lasso model
                    lasso_model = Lasso.Lasso(l1_lambda=l1_lambda)

                    weights,_,_,_ = lasso_model.fit( X_train, Y_train, num_epochs=num_epochs, 
                                               learning_rate=learning_rate, show_graph=False )
                    #store final weights of features
                    cv_weights.append(weights)

                    #validation
                    Y_pred = lasso_model.predict(X_val)
                    accuracy = lasso_model.get_accuracy(Y_val, Y_pred)
                    cv_accuracy.append(accuracy)


                    #overall average test accuracy
                    avg_cv_accuracy = np.mean(cv_accuracy)

                    if avg_cv_accuracy > best_accuracy:
                        
                        best_accuracy = avg_cv_accuracy
                        best_params = (num_epochs,learning_rate,l1_lambda)
                        
                        if display_search_logs:
                        
                            print("New Best Accuracy of:",best_accuracy)
                            print("New Best Params:    Epochs:",num_epochs,"  LR:",learning_rate,"  LAMB:",l1_lambda)
                            print("-----------------------------------------------------------------------")
                        
                        
    print("Final CV Accuracy: ",best_accuracy)
    print("Final Params:    Epochs:",best_params[0],"  LR:",best_params[1],"  LAMB:",best_params[2])
                        
    print("Now we test the best parameters on test set:")
    X_train = D_train[:, :-1]
    Y_train = D_train[:, -1]
    Y_train = np.where(Y_train == 0, -1, Y_train)
    
    lasso_model = Lasso.Lasso( l1_lambda = best_params[2] )


    # Training the model
    weights,weight_arr,_,_  = lasso_model.fit( X_train, Y_train, num_epochs=best_params[0], 
                                               learning_rate=best_params[1], show_graph=True, X_val = X_test,  Y_val = Y_test )
    
    # Displaying how the weights of the features changed over the epochs
    fig = plt.figure(figsize=(10,10))
    
    x_axis = np.arange(best_params[0])
    
    for i in range(weight_arr.shape[0]):
        plt.plot(x_axis, np.abs(weight_arr[i]), label= feature_names[i] )
     
    plt.title("Weight Change vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Absoulte Value of Weight")
    plt.legend(loc=(1.04,0.5))
    plt.show()

    #testing
    Y_pred = lasso_model.predict(X_test)
    
    print("---- Lasso Test Results ----")
    utilities.get_metrics(Y_test, np.squeeze(Y_pred), print_metrics=True)

    print("Final Weights:\n",weights)
    
    
    
if __name__ == '__main__':
    
  print("---Without adding PCA---")
  main_lasso_classifier_cv(add_PCA = False)
  print("---With PCA---")
  main_lasso_classifier_cv(add_PCA = True)
    
