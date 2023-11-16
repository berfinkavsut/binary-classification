"""
Authors: Berfin Kavşut
         Mert Ertuğrul
"""

import numpy as np
import matplotlib.pyplot as plt
import utilities
import pandas as pd
import math

from sklearn.utils import shuffle

class Lasso():
    def __init__(self, l1_lambda=0.1):
        self.l1_lambda = l1_lambda
        self.w = None
        #no bias term, since data is expected to be centered 

    def fit(self, X, Y, random_seed=1, num_epochs=3, learning_rate=0.01, show_graph=True, X_val = None, Y_val = None):
        
        sample_no = X.shape[0]
        input_dim = X.shape[1]

        # initializing coefficients of hyperplane
        np.random.seed(random_seed)
        self.w = np.random.randn(input_dim, 1) * 0.1
        
        average_costs = []
        accuracy_list = []
        weight_list = np.zeros((input_dim,num_epochs))
        if Y_val is not None:
            val_accuracy_list = []

        for epoch in range(num_epochs):

            X,Y = shuffle(X,Y)  # stochastic gradient descent
            cost = 0
            for i in range(sample_no):
                x = X[i, :]
                x = np.reshape(x, (1, input_dim))
                y = Y[i]
                
                # update coefficients
                self.w = self.gradient_descent(self.w, x, y, learning_rate)
                cost += self.calc_cost(self.w, x, y)

            cost = cost / sample_no

            cost = np.asarray(cost)
            cost = np.squeeze(cost)

            #accuracy, y_preds = self.score(X, Y)
            
            y_preds = self.predict(X)
            
            accuracy =  self.get_accuracy(Y, y_preds)
            
            if Y_val is not None: 
                y_val_preds = self.predict(X_val)
            
                val_accuracy =  self.get_accuracy(Y_val, y_val_preds)
                val_accuracy_list.append(val_accuracy)

            average_costs.append(cost)
            accuracy_list.append(accuracy)
            weight_list[:,epoch] = self.w 

        if show_graph:
            x_axis = np.arange(num_epochs)

            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(2, 1, 1)
            ax.plot(x_axis, average_costs)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')

            ax2 = fig.add_subplot(2, 1, 2)
            ax2.plot(x_axis, accuracy_list, label = "Training Accuracy")
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_ylim((0.5,1))
            
            if Y_val is not None:
                ax2.set_ylabel('Accuracy')
                ax2.plot(x_axis, val_accuracy_list, label = "Test Accuracy")
                ax2.legend()
                ax2.set_title('Training and Test Accuracy for Lasso')
            else:
                ax2.set_title('Training Accuracy for Lasso')

            ax.set_title('Training Loss for Lasso')
           
            #plt.savefig('lasso_training_graphs.png')
            plt.show()

        return self.w, weight_list, average_costs, accuracy_list

    def calc_cost(self, w, x, y):
        # calculate rss 
        rss = (1/2) * np.sum( np.square( y - (np.dot(x, w) ) )) 

        # calculate regularization term
        reg_term = self.l1_lambda * np.sum(np.abs(w))

        # calculate cost
        cost = rss + reg_term
        return cost
    
    # returns average accuracy given real and predicted labels
    def get_accuracy(self, Y_real, Y_predicted):
        Y_labelled = (Y_predicted >= 0)
        Y_labelled = np.where(Y_labelled == 0, -1, Y_labelled)
        
        return ( Y_labelled == Y_real).mean()

    def gradient_descent(self, w, x, y, learning_rate):
    
        x = np.squeeze(x)
        w = np.squeeze(w)
        
        n = w.shape[0]
        dw = np.zeros([1,n])
        dw = np.squeeze(dw)

        # gradient of mse loss with l1 regularization
        for i in range(n):
            rss_der = -1 * x[i] * (y-np.dot( x,w.T ))
            if w[i] > 0:
                dw[i] = rss_der + self.l1_lambda
            else:
                dw[i] = rss_der - self.l1_lambda

        # update weights
        w = w - learning_rate * dw
        return w
    
    def predict(self,X):
        
        Y_pred = np.dot(X,self.w)
        return Y_pred