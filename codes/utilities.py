"""
Authors: Berfin Kavşut -  21602459
         Mert Ertuğrul - 21703957
"""
import math
import numpy as np
from sklearn.metrics import precision_score, \
    recall_score, confusion_matrix, classification_report, \
    accuracy_score, f1_score
import matplotlib.pyplot as plt



class Standardizer:

  def __init__(self):
    self.mean_vector = None
    self.std_vector = None

  #function for normalizing with training mean and std
  def standardize_data(self,X, testing=False):

    """
    Standardizes the data by subtracting its mean and dividing by its stdandard deviation

    X: data matrix with each row = one data point
    testing: whether the data is the testing set, in which case the old mean and std values from
    the training set are used

    Features:  to normalize?

    0.   age______yes
    1.   sex______no
    2.   cp_______yes
    3.   trestbps_yes
    4.   chol_____yes
    5.   fbs______no
    6.   restecg__no
    7.   thalach__yes
    8.   exang____no
    9.   oldpeak__yes
    10.  slope____yes
    11.  ca_______yes
    12.  thal_____yes

    Label:

    13.   target___no 
    """

    if not testing:
      self.mean_vector = np.mean(X, axis=0)
      self.std_vector = np.std(X, axis=0)

    A = (X - self.mean_vector) / self.std_vector

        #some rows should not be standardized 
    A[:,1] = X[:,1]
    A[:,5] = X[:,5]
    A[:,6] = X[:,6]
    A[:,8] = X[:,8]
    if X.shape[1]>13:
        A[:,13] = X[:,13]

    np.take(A,np.random.permutation(A.shape[0]),axis=0,out=A)

    return A



def k_fold_split( D, k ):

  """
  Prepares the data for k-fold cross validation, outputs the prepared data
  D: data matrix including features (X) and labels (Y)
  k: number of folds to divide data into
  """

  np.take(D,np.random.permutation(D.shape[0]),axis=0,out=D)
  fold_size = math.ceil( D.shape[0]/k )
  X_folds = []
  Y_folds = []

  X_folds = [ D[ i*fold_size: min( D.shape[0], (i+1)*fold_size ) , :-1] for i in range(k) ]
  Y_folds = [ D[ i*fold_size: min( D.shape[0], (i+1)*fold_size ) , -1] for i in range(k) ]

  #for i in range(k):
     #print(Y_folds[i].shape) 

  return X_folds, Y_folds 



def split_train_test(D, test_fraction = 0.1):
    
  np.take(D,np.random.permutation(D.shape[0]),axis=0,out=D)
    
  test_idx = int( D.shape[0]*0.1 )
        
  D_test = D[:test_idx]
  D_train = D[test_idx:]
    
  return D_train, D_test

def get_metrics(Y_real, Y_predicted, print_metrics=True):
    
    Y_real = np.where(Y_real == -1, 0, Y_real)
    Y_predicted = np.where(Y_predicted == -1, 0, Y_predicted)
    Y_predicted = (Y_predicted >= 0.5)

    
    accuracy = accuracy_score(Y_real, Y_predicted)
    precision = precision_score(Y_real, Y_predicted)
    recall = recall_score(Y_real, Y_predicted)
    f1 = f1_score(Y_real, Y_predicted)
    
    if print_metrics:
        print("Accuracy: ",accuracy)
        print( classification_report(Y_real, Y_predicted) )
        
        cm = confusion_matrix(Y_real, Y_predicted)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(cm, cmap='OrRd')
        plt.title('Confusion Matrix')
        fig.colorbar(cax)
        plt.show()
        
        
    return accuracy,precision,recall,f1

    