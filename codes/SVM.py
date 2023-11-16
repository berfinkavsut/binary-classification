"""
Authors: Berfin Kavşut
         Mert Ertuğrul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utilities

from sklearn.utils import shuffle


class svm:
    def __init__(self, reg_param=1.0):
        # regularization parameter, C
        self.reg_param = reg_param
        self.w = None

    def fit(self, X, Y, random_seed=1, num_epochs=50, learning_rate=0.001, show_graph=True, X_val = None, Y_val = None):
        
        # change labels of 0 to -1, negative class
        Y = np.where(Y == 0, -1, Y)

        sample_no = X.shape[0]
        input_dim = X.shape[1]

        # initializing coefficients of hyperplane
        np.random.seed(random_seed)
        self.w = np.random.randn(input_dim, 1) * 0.1

        average_costs = []
        accuracy_list = []
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

            accuracy, y_preds = self.score(X, Y)
            
            if Y_val is not None: 
               val_accuracy,_ = self.score(X_val,Y_val)
               val_accuracy_list.append(val_accuracy)

            average_costs.append(cost)
            accuracy_list.append(accuracy)

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
            
            if Y_val is not None:
                ax2.set_ylabel('Accuracy')
                ax2.plot(x_axis, val_accuracy_list, label = "Test Accuracy")
                ax2.legend()
                ax2.set_title('Training and Test Accuracy for SVM')
            else:
                ax2.set_title('Training Accuracy for SVM')

            ax.set_title('Training Loss for SVM')

            # plt.savefig('svm_accuracy_loss.png')
            plt.show()

    def score(self, X, Y):
        # change labels of 0 to -1, negative class
        Y = np.where(Y == 0, -1, Y)

        # find score for given input X and Y
        sample_no = X.shape[0]
        y_preds = []
        count = 0

        # get rid of for loop later
        for i in range(sample_no):
            x = X[i, :]
            y_real = Y[i]

            output = np.dot(x, self.w)
            if output >= 0:
                y_pred = 1
            else:
                y_pred = -1
            y_preds.append(y_pred)

            if y_pred == y_real:
                count += 1

        accuracy = count / sample_no
        y_preds = np.array(y_preds)

        return accuracy, y_preds

    def calc_cost(self, w, x, y):
        # calculate hinge loss
        d = 1 - y * (np.dot(x, w))
        hinge_loss = max(d, 0)

        # calculate regularization term
        reg_term = 1 / 2 * np.dot(np.transpose(w), w)

        # calculate cost
        cost = reg_term + self.reg_param * hinge_loss
        return cost

    def gradient_descent(self, w, x, y, learning_rate):

        dw = 0
        # gradient of hinge loss
        if 1 - y * np.dot(x, w) <= 0:
            dw = w
        else:
            dw = w - self.reg_param * y * np.transpose(x)

        # update weights
        w = w - learning_rate * dw
        return w

def main():
    np.random.seed(1)

    data_heart = pd.read_csv('./heart.csv')

    # standardize data
    input = utilities.standardize_data(data_heart.values)

    # split features and label
    X = input[:, :-1]
    Y = input[:, -1]

    # split training set and test set
    X_train = X[:int(X.shape[0] * 0.9)]
    X_test = X[int(X.shape[0] * 0.9):]
    Y_train = Y[:int(Y.shape[0] * 0.9)]
    Y_test = Y[int(Y.shape[0] * 0.9):]

    # change labels of 0 to -1, negative class
    Y_train = np.where(Y_train == 0, -1, Y_train)
    Y_test = np.where(Y_test == 0, -1, Y_test)

    # SVM model
    svm_model = svm()
    costs = svm_model.fit(X_train, Y_train)

    # accuracy on training set
    accuracy, y_preds = svm_model.score(X_train, Y_train)
    print('Training Accuracy:', accuracy)

    # accuracy on test set
    accuracy, y_preds = svm_model.score(X_test, Y_test)
    print('Test Accuracy:', accuracy)


if __name__ == '__main__':
    main()
