"""
Authors: Berfin Kavşut -  21602459
         Mert Ertuğrul - 21703957
"""

import numpy as np
import matplotlib.pyplot as plt
import utilities
import pandas as pd
import math


class networkLayer:
    
    """
    This class represents a signel layer of a multi layer perceptron.
    The layer stores:
        -W: weight matrix parameter 
        -b: bias parameter
        -Z: output of the last iteration - used for back propagation
        -d_W: gradient of W
        -d_b: gradient of b

    """

    def __init__(self, in_dim, out_dim, activation_func):

      self.activation_func = activation_func
      
      #initializing parameters of layer
      
      #we keep the random seed constant to make our results repeatable by 3rd parties
      self.W = np.random.randn(out_dim, in_dim) * 0.1
      self.b = np.random.randn(out_dim, 1) * 0.1
      
      #storing output for back prop
      self.Z = 0
      self.input_X = 0
      
      #each layer stores its gradient
      self.d_W = 0
      self.d_b = 0
      
    #applies forward propogation to the layer and keeps result
    def forward_prop(self, input_X, training=True):
        
      if training:
          self.input_X = input_X
          self.Z = np.dot(self.W, input_X) + self.b
          
          #activation function
          if self.activation_func == "sigmoid":
            return self.sigmoid(self.Z)
          elif self.activation_func == "relu":
            return self.relu(self.Z)
      else:

          Z = np.dot(self.W, input_X) + self.b
          
          #activation function
          if self.activation_func == "sigmoid":
            return self.sigmoid(Z)
          elif self.activation_func == "relu":
            return self.relu(Z)
      
    #applies back propogation, stores the gradients of the parameters,
    #returns gradient of input to be used for the preceding layer
    def backward_prop(self, d_out):
      
      #gradient of Z
      if self.activation_func == "sigmoid":
        d_Z = d_out * self.sigmoid(self.Z) * (1 - self.sigmoid(self.Z))
        
      elif self.activation_func == "relu":
        d_Z = d_out.copy()
        d_Z[self.Z <= 0] = 0 

      #gradient of params
      self.d_W = (1/self.input_X.shape[1]) * np.dot(d_Z, self.input_X.T) 
      self.d_b = (1/self.input_X.shape[1]) * np.sum(d_Z, axis=1, keepdims=True) 
      #gradient of input
      d_in = np.dot(self.W.T, d_Z)
      
      #clearing stored input for next epoch
      self.input_X = 0
      
      return d_in
      
    #updates paraemeters based on gradients and learning rate
    def update(self, learning_rate):
      self.W -= self.d_W * learning_rate
      self.b -= self.d_b * learning_rate
      
    #relu activation function used for hidden layers
    def relu(self, X):
        return np.maximum(0,X)
    
    #sigmoid activation function used for last layer
    def sigmoid(self, X):
      return 1/(1+np.exp(-X))
     


class NeuralNetwork:
    
    """
    This class represents a Multi Layer Perceptron. New layers are added, 
    forward and backward propogation, training and testing functionalities 
    are provided.
    
    """
    epsilon = 1e-6
    
    def __init__(self, learning_rate = 0.01, threshold = 0.5):
      self.learning_rate = learning_rate
      self.threshold = threshold
      self.layers = []

    def addLayer(self, input_dim, output_dim, activation):
      self.layers.append(networkLayer(input_dim, output_dim, activation))
    
    def forward_prop(self, X, training=True):
    
      Z_temp = X
      for layer_num in range(len(self.layers)):
        Z_temp = self.layers[layer_num].forward_prop(Z_temp, training=training)
      return Z_temp
  
  
    def form_architecture(self, input_dim = 13, hidden_layers = 2):
        #forming architecture
        if hidden_layers==0:
            self.addLayer(input_dim, 1, "sigmoid")
        else:
            if hidden_layers==1:
                self.addLayer(input_dim, 8, "relu")
            else:
                self.addLayer(input_dim, 16, "relu")
                for i in range(hidden_layers-2):
                    self.addLayer(16, 16, "relu")    
                self.addLayer(16, 8, "relu")
    
            self.addLayer(8, 1, "sigmoid")
    
    def backward_prop(self, Y_real, Y_predicted):
      
      #derivative of logit loss to be sent back
      d_in = - (np.divide(Y_real, Y_predicted + self.epsilon) - np.divide(1 - Y_real, 1 - Y_predicted + self.epsilon ))
      
      for layer_num in reversed( range(len(self.layers) ) ):
        #store output gradient (input gradient of the following layer)
        d_out = d_in
        #get input gradient
        d_in = self.layers[layer_num].backward_prop( d_out )
    
    def update_params(self):
      for layer_num in range(len(self.layers)):
        self.layers[layer_num].update(self.learning_rate)
    
    
    # returns the logit loss given real and predicted labels
    def logit_loss(self, Y_real, Y_predicted):
      loss = (-1 / Y_predicted.shape[1] )*(np.dot(Y_real, np.log(Y_predicted + self.epsilon).T) + np.dot(1 - Y_real, np.log(1 - Y_predicted  + self.epsilon).T))  
      return np.squeeze(loss)
      
    # returns average accuracy given real and predicted labels
    def get_accuracy(self, Y_real, Y_predicted):
      return ( (Y_predicted >= self.threshold)== Y_real).mean()
    
    
    def train(self, X, Y_real,num_epochs, show_graph = True, mini_batch_size=0, X_val = None, Y_val = None):
    
      loss_list = []
      accuracy_list = []
      
      if Y_val is not None:
          val_accuracy_list = []
    
      if mini_batch_size != 0:
        batch_num = math.floor( X.shape[0]/mini_batch_size )
        
        X_batches = np.array_split(X, batch_num)
        Y_batches = np.array_split(Y_real, batch_num)
      else:
        X_batches = [X]
        Y_batches = [Y_real]
    
      for i in range(num_epochs):
          
        for b in range(len(X_batches)):
            
            Y_predicted = self.forward_prop(X_batches[b].T)
            
            self.backward_prop(Y_batches[b], Y_predicted)
            self.update_params()
    
    
        accuracy,Y_predicted = self.test(X , Y_real)
        loss = self.logit_loss(Y_real, Y_predicted)
        #print( "Loss: " + str(loss) )
        loss_list.append( loss )
        
        #print( "Accuracy: " + str(accuracy) )
        accuracy_list.append( accuracy )
        
        if Y_val is not None:     
            val_acc,_ = self.test(X_val , Y_val)
            val_accuracy_list.append(val_acc)

            
        
        
      #option to display the training loss and accuracy graph
    
      if show_graph:
    
        x_axis = np.arange(num_epochs)
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(x_axis, loss_list)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss for Neural Network')
    
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(x_axis, accuracy_list, label = "Training Accuracy")
        #We are using the test acuracy only for display, 
        #it does not influence training so we do not call it Validation Accuracy here.
        ax2.set_xlabel('Epoch')
        ax2.set_ylim((0.5,1))
        
        if Y_val is not None:
            ax2.set_ylabel('Accuracy')
            ax2.plot(x_axis, val_accuracy_list, label = "Test Accuracy")
            ax2.set_title('Training and Test Accuracy for Neural Network')
            ax2.legend()
        else:
            ax2.set_ylabel('Training Accuracy')
            ax2.set_title('Training Accuracy for Neural Network')
            
        plt.show()
        
      return loss_list, accuracy_list
    
    #given a test set, returns the accuracy and predicted labels
    def test(self, X, Y_real):
      Y_predicted = self.forward_prop(X.T, training=False)
      return self.get_accuracy(Y_real, Y_predicted), Y_predicted
  
#standalone main function to demonstrate the model's k-fold cross validation results on graphs
def main_cv_grapher():
    
    np.random.seed(1)
    
    #reading the data into a pandas dataframe and converting to numpy array
    data_heart = pd.read_csv('./heart.csv')

    #standardizing and shuffling the data 
    standardizer = utilities.Standardizer()

    D = standardizer.standardize_data( data_heart.values )
    
    #number of folds for k fold cross validation
    k=10
    num_epochs = 200
    learning_rate = 0.1
    
    X_folds,Y_folds = utilities.k_fold_split( D, k )
    
    loss_data = np.zeros( (10,num_epochs) )
    accuracy_data = np.zeros( (10,num_epochs) )
    
    val_accuracy = []
    
    x_axis = np.arange(num_epochs)
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(2, 1, 1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_ylim([0, 1])
    
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    
    for i in range(10):
        
      #creating training data by omitting the ith fold
      X_train = np.concatenate( X_folds[:i] + X_folds[i+1:] , axis=0 )
      Y_train = np.concatenate( Y_folds[:i] + Y_folds[i+1:] , axis=0 )
    
      #ith fold is the validation data
      X_val = X_folds[i]
      Y_val = Y_folds[i]
    
      newNetwork = NeuralNetwork(learning_rate)
      #forming architecture
      newNetwork.addLayer(13, 16, "relu")
      newNetwork.addLayer(16, 8, "relu")
      newNetwork.addLayer(8, 1, "sigmoid")
     #training the network
      loss_series, train_accuracy_series = newNetwork.train(X_train,Y_train,num_epochs, False)
      
      #keeping track of the logit loss and train set accuracy series to 
      #plot against epochs
      loss_data[i] = np.array(loss_series)
      accuracy_data[i] = np.array(train_accuracy_series)
    
      #testing with ith fold
      avg_val_result,val_result = newNetwork.test(X_val,Y_val)
      print("Iteration " + str(i) + " validation accuracy: " + str(avg_val_result) )
      val_accuracy.append( avg_val_result )
    
      ax.plot(x_axis, loss_series )
      ax2.plot(x_axis, train_accuracy_series )
      
    avg_accuracy =  np.mean(accuracy_data, axis=0)
    avg_loss =  np.mean(loss_data, axis=0)
    
    #average values accross the 10 folds
    ax.plot(x_axis, avg_loss, label= "average loss", linewidth=3.0)
    ax2.plot(x_axis, avg_accuracy, label= "average accuracy", linewidth=3.0)
    
    
    # Set a title of the current axes.
    ax.set_title('Neural Network Training Loss Graph for 10-fold Cross Validation')
    ax2.set_title('Neural Network Training Accuracy Graph for 10-fold Cross Validation')
    ax.legend()
    ax2.legend()
    plt.show()
    
    #overall average test accuracy
    avg_val_accuracy = np.mean(val_accuracy)
    
    print("Average validation accuracy: "+str(avg_val_accuracy))



if __name__ == '__main__':
   main_cv_grapher()
    


