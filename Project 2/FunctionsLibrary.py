import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.model_selection import train_test_split


def importtest():
    print("The module has been succesfully imported.")

def FrankeFunction(x,y,noisefactor):
    """
    Input:
        x  : x part of meshgrid of x,y
        y  : y part of meshgrid of x,y
        noisefactor : magnitude of the noise.

    Computes the frankefunction on a meshgrid with added noise, if noisefactor >0, and returns it
    as a meshgrid.
    Output:
        FrankeFunction : (N,N) dataset of the franke function.
    """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    term5 = noisefactor*np.random.normal(0,1,(len(x),len(y)))
    return term1 + term2 + term3 + term4 + term5


def logisticX(X):
    """
    Input:
        The features classes as a vector with shape (?,?)

    Ouput:
        The design matrix used in logistic regression

    Add 1s to the matrix
    """
    return X

    
def regressionX(x, y, n ):
    """
    Inputs:
        x  : x part of meshgrid of x,y
        y  : y part of meshgrid of x,y
        n  : Polymomial degree of the fit, (x+y)^n
    Creates the design matrix for polynomial regression,
    where the columns are ordered as:
    [1, x,y,x^2,xy,y^2,x^3,x^2y, ..., xy^(n-1), y^(n)]
    
    Outputs:
        X  : Design matrix (N,l) where l=(n+1)(n+2)/2
    """
    
    
    if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)          # Number of elements in beta                                                               
    X = np.ones((N,l))

    for i in range(1,n+1):
            q = int((i)*(i+1)/2)
            for k in range(i+1):
                    X[:,q+k] = (x**(i-k))*(y**k)

    return X






"""
Should create a class for both regression and logistic-regression. 
The regression class should specifiy which regression method to be used
 - Ordinary Least Squares
 - Ridge
 - Lasso if needed in the problems (since this is a class by itself imported by sklearn it will probably not be needed)

It should also take in the parameters needed which are

"""

def SVDinv(A):
    """
    Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    # SVD is numerically more stable than the inversion algorithms provided by
    # numpy and scipy.linalg at the cost of being slower. (Hjorth-Jensen, 2020) 
    """
    U, s, VT = np.linalg.svd(A, full_matrices = False)
    invD = np.diag(1/s)
    UT = np.transpose(U); V = np.transpose(VT);
    return np.matmul(V,np.matmul(invD,UT))
    

    
def scores(z,z_predict):
    """
    Computes the MSE and R2 scores for the model with given inputs
    """
    MSE = np.mean((z-z_predict)**2)
    R2 = 1-np.sum((z-z_predict)**2)/np.sum((z-np.mean(z))**2)
    return MSE, R2



class Regression(object):
    
    def __init__(self, X, z, n, method, resampling,
                 learning_rate = 0.01, batch_size = 30, max_epoch = 100, verbose = 0):
        self.X = X   
        self.z = z
        self.n = n
        self.method = method
        self.resampling = resampling
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.verbose = verbose
    
   

    """
    Need 
     - if tests for OLS/ridge
     - training test split into the class.
    """

    def fit(self,X, z):
        
        #self.BETA = np.array()
        
        if self.resampling == None:
            if self.method[0] == "OLS":
                XTXinv = SVDinv(X.T.dot(X))
                self.BETA = XTXinv.dot(X.T).dot(z)

            elif self.method[0] == "RIDGE":
                XTXinv = SVDinv(X.T.dot(X)+lamb*np.eye(len(X[0])))
                self.BETA = XTXinv.dot(X.T).dot(z)
        
        elif self.resampling[0] == "bootstrap":
        
            N = X.shape[0]
            for n_boot in range(self.resampling[1]):
                idx = np.random.randint(0,N,size=(N)) # Resampling the indexes with replacement
                X = X[idx,:]
                z = z[idx]
                
                if self.method[0] == "OLS":
                    XTXinv = SVDinv(X.T.dot(X))
                    self.BETA = XTXinv.dot(X.T).dot(z)

                elif self.method[0] == "RIDGE":
                    XTXinv = SVDinv(X.T.dot(X)+lamb*np.eye(len(X[0])))
                    self.BETA = XTXinv.dot(X.T).dot(z)
                    
        elif self.resampling[0] == "CV":
            N = X.shape[0]
            
            ind = np.random.permutation(len(X))
            X = X[idx,:]
            z = z[idx,:]
            #Split data into k folds
            folds_X = np.array_split(X,folds)
            folds_z = np.array_split(X,folds)
      

            # Setting up the array for the k fold predictions
            foldsize = np.int(len(X)/self.resampling[1])  
            
            
            
            
            for j in range(resampling[1]):
                idx = np.random.randint(0,N,size=(N)) # Resampling the indexes with replacement
                # Copy data frames
                tmp_X = folds_X.copy()
                tmp_z = folds_z.copy()
                
                # Save target fold from data frame
                cur_leaveout_X = tmp_X[j]
                cur_leaveout_z = tmp_z[j]
                
                # Remove leaveouts 
                tmp_X.pop(j)
                tmp_z.pop(j)
                
                cur_X_trainfold = np.concatenate(tmp_X)
                cur_z_trainfold = np.concatenate(tmp_z)
                
                
                
                if self.method[0] == "OLS":
                    XTXinv = SVDinv(cur_X_trainfold.T.dot(cur_X_trainfold))
                    self.BETA = XTXinv.dot(cur_X_trainfold.T).dot(cur_z_trainfold)

                elif self.method[0] == "RIDGE":
                    XTXinv = SVDinv(cur_X_trainfold.T.dot(cur_X_trainfold)+lamb*np.eye(len(cur_X_trainfold[0])))
                    self.BETA = XTXinv.dot(cur_X_trainfold.T).dot(cur_z_trainfold)
        return self

    
    def predict(self, X):
        z_predict = X.dot(self.BETA)
        return z_predict





"""
Neural Network class beneath
with inspiration from
https://medium.com/@udaybhaskarpaila/multilayered-neural-network-from-scratch-using-python-c0719a646855
and Nielsen 2015 text
"""
class NN(object):

    def __init__(self, layer_dims, hidden_layers, learning_rate=0.01,
                 batch_size = 20, max_epoch=100, verbose=0):

        self.layer_dims = layer_dims
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.verbose = verbose

    @staticmethod
    def weights_init(layer_dims):
        parameters = {}
        # L = number of layers,
        # The values in the layer_dims is the amount of weights in each of the nodes
        L = len(layer_dims)
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.normal(-1,1,(layer_dims[l], layer_dims[l-1]) )
            parameters['b' + str(l)] = np.random.normal(-1,1,(layer_dims[l], 1) )
        return parameters

    @staticmethod
    def sigmoid(X, derivative=False):
        if derivative == False:
            out = 1 / (1 + np.exp(-np.array(X)))
        elif derivative == True:
            s = 1 / (1 + np.exp(-np.array(X)))
            out = s*(1-s)
        return out

    @staticmethod
    def relu(X, derivative=False):
        leak = 0.1
        if derivative == False:
            out = np.maximum(0,X)
            return out
        elif derivative == True:
            out = np.ones_like(X)
            out[X<0] = 0
            return out
    """
    @staticmethod
    def relu(X,derivative=False):
        leak = 0.01
        X = np.array(X,dtype=np.float64)

        if derivative == False:
            out = np.where(X<0,leak*X,X)
        elif derivative == True:
            out = np.ones_like(X,dtype=np.float64)
            out[X < 0] = leak

        return out
    """

    @staticmethod
    def linear(X,derivative=False):

        X = np.array(X)
        if derivative ==  False:
            return X
        elif derivative == True:
            return np.ones_like(X)

    @staticmethod
    def forward_propagation(X, hidden_layers, parameters):

        caches = []
        A = X
        L = len(hidden_layers)

        for l, active_function in enumerate(hidden_layers,start=1):
            A_l = A

            # Taking the inputs from the A_l-th layer
            # Z is the previous layer times the weights and biases
            Z = np.dot(parameters["W" + str(l)], A_l)+parameters["b" + str(l)]


            # Running through the activaiton function of the given, hidden layer
            if active_function == "sigmoid":
                A = NN.sigmoid(Z)
            elif active_function == "relu":
                A = NN.relu(Z)
            elif active_function == "linear":
                A = NN.linear(Z)

            # Storing the inputs, activation and the parameters into a cache
            # to be used in the back propagation

            cache = ((A_l, parameters['W' + str(l)], parameters['b' + str(l)]), Z)
            caches.append(cache)

        return A, caches

    @staticmethod
    def compute_cost(Ytilde, Y, parameters):

        # MSE cost function for regression.
        # A here is the input
        cost = np.squeeze(0.5*np.sum((Ytilde-Y)**2))

        # For a logistic, the cost/loss function would be cross entropy / log likelihood

        return cost

    @staticmethod
    def back_propagation(AL, Y, caches, hidden_layers):

        # dZ is used for the error term, as the error term is given as dC/dZ


        # Initalizing the gradients
        grads = {}
        L = len(caches)

        m = AL.shape[1]

        # Reshaping the output Y
        Y = Y.reshape(AL.shape)

        # The first error term
        # The derivative of the cost function with respect to z^L
        dZL = AL - Y

        # Getting the second last cache, which contains
        # ( (A_(L-1), W_(L-1), b_(L-1) ), Z_(L-1) )
        cache = caches[L-1]
        linear_cache, activation_cache = cache
        AL, W, b = linear_cache

        # Calculating the gradients for the last layer parameters
        grads["dW" + str(L)] = np.dot(dZL,AL.T)                    # Error term times output
        grads["db" + str(L)] = np.sum(dZL,axis=1,keepdims=True)/m  # Mean of the error term

        # Gradient for the next last layer needed for deeper computation
        grads["dA" + str(L-1)] = np.dot(W.T,dZL)


        # Loop from l=L-1 to l=0
        for l in reversed(range(L-1)):
            cache = caches[l]
            active_function = hidden_layers[l]

            linear_cache, Z = cache
            A_prev, W, b = linear_cache


            dA_prev = grads["dA" + str(l + 1)]

            if active_function == "sigmoid":
                dZ = np.multiply(dA_prev, NN.sigmoid(Z,derivative=True))
            elif active_function == "relu":
                dZ = np.multiply(dA_prev, NN.relu(Z,derivative=True))
            elif active_function == "linear":
                dZ = np.multiply(dA_prev, NN.linear(Z,derivative=True))


            grads["dA" + str(l)] = np.dot(W.T,dZ)
            grads["dW" + str(l + 1)] = np.dot(dZ,A_prev.T)
            grads["db" + str(l + 1)] = np.sum(dZ,axis=1,keepdims=True)
        return grads



    @staticmethod
    def update_parameters(parameters, grads,learning_rate,iter_no):

        L = len(parameters) // 2 # number of layers in the neural network

        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l + 1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l + 1)]
        return parameters

    def fit(self,X,y):
        # Initlizing gradient-dictionary and cost log.
        self.grads = {}
        self.costs = []


        # Initiliazing the weights
        if self.verbose == 1:
            print('Initiliazing Weights...')

        self.parameters = self.weights_init(self.layer_dims)

        # Creatomg counter for verbosity
        self.iter_no = 0

        # Creating randomized index for the batches
        M = X.shape[1]
        idx = np.arange(0,M)

        if self.verbose == 1:
            print('Starting Training...')

        for epoch_no in range(1,self.max_epoch+1):

            # Randomizing the data for each epoch
            np.random.shuffle(idx)
            X = X[:,idx]
            y = y[:,idx]

            for i in range(0,M, self.batch_size):
                self.iter_no = self.iter_no + 1

                # Looping through the batches
                X_batch = X[:,i:i + self.batch_size]
                y_batch = y[:,i:i + self.batch_size]

                # Forward propagation:
                AL, cache = self.forward_propagation(X_batch, self.hidden_layers, self.parameters)

                # Cost function
                cost = self.compute_cost(AL, y_batch, self.parameters)
                self.costs.append(cost)

                # Getting the gradients from the back propagation
                grads = self.back_propagation(AL, y_batch, cache,self.hidden_layers)

                # Updating the weights and biases
                self.parameters = self.update_parameters(self.parameters,grads,self.learning_rate,
                                                                        self.iter_no-1)

                if self.verbose == 1:
                    if self.iter_no % (self.max_epoch//10) == 0:
                        print("Cost after iteration {}: {}".format(self.iter_no, cost))

        return self

    def predict(self,X):

        out, _ = self.forward_propagation(X,self.hidden_layers,self.parameters)
        return out
