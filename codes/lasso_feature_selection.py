"""
Authors: Berfin Kavşut
         Mert Ertuğrul
"""


import Lasso, utilities, PCA

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

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

def main_lasso_feature_selection(add_PCA = True, display_search_logs=False):
    np.random.seed(1)
    num_epochs=10
    learning_rate=0.01
    l1_lambda = 0.05

    #reading the data into a pandas dataframe and converting to numpy array
    data_heart = pd.read_csv('./heart.csv')
    D = data_heart.values

    #train-test split
    D_train, D_test = utilities.split_train_test(D)

    #standardizing and shuffling the data 
    standardizer = utilities.Standardizer()
    
    #cross validation and training data
    D_train = standardizer.standardize_data( D_train )
    
    X_train = D_train[:, :-1]
    Y_train = D_train[:, -1]
    Y_train = np.where(Y_train == 0, -1, Y_train)
    
    #adding PCA
    if add_PCA:
        pca_obj = PCA.PCA_maker(X=X_train, n_components=8)
        PCA_features_train = pca_obj.apply_pca(X=X_train, display=False)
    
        X_train = np.hstack( ( X_train, PCA_features_train ) )

    #Lasso model
    lasso_model = Lasso.Lasso( l1_lambda = l1_lambda)
    final_weights, weight_arr,_,_  = lasso_model.fit( X_train, Y_train, num_epochs=num_epochs, 
                                               learning_rate=learning_rate, show_graph=False)

    x_axis = np.arange(num_epochs)
    
    for i in range(weight_arr.shape[0]):
        plt.plot(x_axis, np.abs(weight_arr[i]), label= feature_names[i])
        
    plt.title("Weight Change vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Absoulte Value of Weight")
    plt.legend(loc=(1.04,0.5))
    plt.show()

    #-----------------------------------------------------------------------------------------
    lambda_list = [0,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5]
   
    weights_vs_lambda = np.zeros((X_train.shape[1], len(lambda_list)))
    for index, lamb in enumerate(lambda_list):
        # Lasso model
        lasso_model = Lasso.Lasso( l1_lambda =lamb)
        final_weights, weight_arr,_,_  = lasso_model.fit( X_train, Y_train, num_epochs=num_epochs, 
                                                   learning_rate=learning_rate, show_graph=False)

        weights_vs_lambda[:,index] = final_weights

    #showing the plot for different lambdas
    fig = plt.figure()
    for i in range(weight_arr.shape[0]):
        plt.plot(lambda_list, np.abs(weights_vs_lambda[i]), label= feature_names[i])
        
    plt.title("Weight Change vs Lambda")
    plt.xlabel("Lambda")
    plt.ylabel("Absoulte Value of Weight")
    plt.legend(loc=(1.04,0.5))
    plt.show()    
        
if __name__ == '__main__':

  print("---Without adding PCA---")
  main_lasso_feature_selection(add_PCA = False)
  print("---With PCA---")
  main_lasso_feature_selection(add_PCA = True)  
