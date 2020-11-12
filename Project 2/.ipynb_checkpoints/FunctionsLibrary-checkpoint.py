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

def frankeplot(z_pred_surf, title):
    datapoints,_ = z_pred_surf.shape
    x = np.linspace(0,1,datapoints)
    y = np.linspace(0,1,datapoints)
    x,y = np.meshgrid(x,y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(x, y, z_pred_surf, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)

    # Labeling the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.set_title(title, fontsize = 16)
    ax.set_zlim(-0.10, 1.40)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def logisticX(X):
    """
    Input:
        The features classes as a matrix of the  with shape (N,p-1)
    Ouput:
        The design matrix used in logistic regression with shape 
    Add 1s to the matrix of the features as the first column.
    """
    ONE = np.ones((X.shape[0])).reshape(X.shape[0],1)
    X = np.concatenate((ONE, X), axis=1)
    
    return X

class logisticmulticlass(object):
    
    def __init__(self, X, y, optimizer, learning_rate= 0.01, batch_size = 32, max_epoch=100):
        
        self.X = X
        self.y = y
        self.C = y.shape[1]
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epoch = max_epoch
    @staticmethod  
    def compute_cost(X, y, beta, verbose=0):
        # Computing the cost function 
        C = beta.shape[1]
        N = X.shape[0]
        cost = np.sum( np.multiply(y,X.dot(beta))-np.log(np.ones((N,C))+np.exp(X.dot(beta)) ))/N
        return cost

    def fit(self,X,y):
        # Initializing beta as a matrix, where each row is corresponding to a model defining that class.        
        self.beta = np.random.normal(0, 1, size=(X.shape[1], y.shape[1]))/y.shape[1]
        
        # Intializing the cost  
        self.costs = []
        
        # Create randomized index for the batches
        N = X.shape[0] 
        idx = np.arange(0,N)
        
        for epoch in range(self.max_epoch):
            # Randomizing the data for each epoch
            np.random.shuffle(idx)
            X = X[idx,:]
            y = y[idx,:]

            for i in range(0, N, self.batch_size):
                
                X_batch = X[i:i+self.batch_size,:]
                y_batch = y[i:i+self.batch_size,:]
                for c in range(0, self.C):
                    
                    # Batch prediction 
                    y_batch_pred = sigmoid(X_batch.dot(self.beta[:,c]))

                    # Calculating the gradient 
                    gradient = -X_batch.T.dot(y_batch[:,c]-y_batch_pred)

                    # Updating Beta
                    self.beta[:,c] -= self.learning_rate*gradient 

            # Computing the cost after each epoch storing it 
            cost = self.compute_cost(X, y, self.beta)
            self.costs.append(cost)
        return self
    
    def predict(self,X):
        self.y_pred = sigmoid(X.dot(self.beta))
        return self.y_pred
    
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

# One-hot in numpy (from lecture notes week 41)
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector

def accuracy(y,y_pred):
    y = np.argmax(y, axis=1) #Returns one-hot encoded vectors back to class integers
    acc = np.sum(y==y_pred)/len(y)
    return acc

def softmax(X):
    return np.exp(X-np.max(X))/np.sum(np.exp(X-np.max(X)))


class logisticmulticlass(object):
    
    def __init__(self, X, y, optimizer, learning_rate= 0.01, batch_size = 32, max_epoch=100):
        
        self.X = X
        self.y = y
        self.C = y.shape[1]
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epoch = max_epoch
    @staticmethod  
    def compute_cost(X, y, beta, verbose=0):
        # Computing the cost function 
        C = beta.shape[1]
        N = X.shape[0]
        cost = np.sum( np.multiply(y,X.dot(beta))-np.log(np.ones((N,C))+np.exp(X.dot(beta)) ))/N
        return cost

    def fit(self,X,y):
        # Initializing beta as a matrix, where each row is corresponding to a model defining that class.        
        self.beta = np.random.normal(0, 1, size=(X.shape[1], y.shape[1]))/y.shape[1]
        
        # Intializing the cost  
        self.costs = []
        
        # Create randomized index for the batches
        N = X.shape[0] 
        idx = np.arange(0,N)
        
        for epoch in range(self.max_epoch):
            # Randomizing the data for each epoch
            np.random.shuffle(idx)
            X = X[idx,:]
            y = y[idx,:]

            for i in range(0, N, self.batch_size):
                
                X_batch = X[i:i+self.batch_size,:]
                y_batch = y[i:i+self.batch_size,:]
                for c in range(0, self.C):
                    
                    # Batch prediction 
                    y_batch_pred = sigmoid(X_batch.dot(self.beta[:,c]))

                    # Calculating the gradient 
                    gradient = -X_batch.T.dot(y_batch[:,c]-y_batch_pred)

                    # Updating Beta
                    self.beta[:,c] -= self.learning_rate*gradient 

            # Computing the cost after each epoch storing it 
            cost = self.compute_cost(X, y, self.beta)
            self.costs.append(cost)
        return self
    
    def predict(self,X):
        self.y_pred = sigmoid(X.dot(self.beta))
        return self.y_pred


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
# Activation Functions
def sigmoid(X,derivative=False):
    if derivative == False:
        # Stabilization trick to prevent underflow/overflow by using different
        # algebraic definitions of sigmoid when the input is negative/positive
        return np.where(X >= 0, 1 / (1 + np.exp(-X)), np.exp(X) / (1 + np.exp(X)))
    elif derivative == True:
        sig = 1 / (1 + np.exp(-np.array(X)))
        out = sig*(1-sig)
    return out

def ReLU(X,alpha=0,derivative=False):
    # ReLU activation function
    # alpha is the leaky parameter, and gets fed into the network as a tuple of shape
    #("relu", alpha=alpha)
    X = np.array(X,dtype=np.float64)
    if derivative == False:
        return np.where(X<0,alpha*X,X)
    elif derivative == True:
        X_relu = np.ones_like(X,dtype=np.float64)
        X_relu[X < 0] = alpha
        return X_relu

def Tanh(X,derivative=False):
    X = np.array(X)
    if derivative == False:
        return np.tanh(X)
    if derivative == True:
        return 1 - (np.tanh(X))**2

def softmax(X):
    C=np.max(X)
    return np.exp(X-C) / np.sum(np.exp(X-C),axis=0)

def linear(X,derivative=False):
    # Linear activation function used for regression
    X = np.array(X)
    if derivative ==  False:
        return X
    if derivative == True:
        return np.ones_like(X)




class NN(object):

    def __init__(self,layer_dims,hidden_layers, cost_function, learning_rate=0.1,
                 optimization_method = "SGD",batch_size=64,max_epoch=100,penalty=None,beta1=0.9,
                 beta2=0.999,lamb=0,verbose=0):
        self.layer_dims = layer_dims
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.verbose = verbose
        
        self.optimization_method = optimization_method
        self.cost_function = cost_function
        self.penalty = penalty
        self.lamb = lamb
        self.beta1 = beta1
        self.beta2 = beta2

    @staticmethod
    def weights_init(layer_dims):
        parameters = {}
        opt_parameters = {}
        L = len(layer_dims)

        for l in range(1, L):
            parameters["W" + str(l)] = np.random.normal(0,np.sqrt(2.0/layer_dims[l-1]),(layer_dims[l], layer_dims[l-1]))
            parameters["b" + str(l)] = np.random.normal(0,np.sqrt(2.0/layer_dims[l-1]),(layer_dims[l], 1))
        return parameters

    @staticmethod
    def forward_propagation(X, hidden_layers,parameters):
        caches = []
        A = X
        L = len(hidden_layers)
        for l,active_function in enumerate(hidden_layers,start=1):
            A_prev = A 
        
            Z = np.dot(parameters["W" + str(l)],A_prev)+parameters["b" + str(l)]
            
            if type(active_function) is tuple:
                
                if  active_function[0] == "relu":
                    A = ReLU(Z,active_function[1])
                elif active_function[0] == "elu":
                    A = elu(Z,active_function[1])
            else:
                if active_function == "sigmoid":
                    A = sigmoid(Z)
                elif active_function == "linear":
                    A = linear(Z)
                elif active_function == "tanh":
                    A = Tanh(Z)
                elif active_function == "softmax":
                    A = softmax(Z)
                
            # Caching the activations, A_prev
            # Weights and biases
            # And the Z weighted sums
            cache = ((A_prev, parameters["W" + str(l)],parameters["b" + str(l)]), Z)
            caches.append(cache)      
        return A, caches
    
    
    @staticmethod
    def compute_cost(A, Y, parameters, cost_function, lamb=0,penalty=None):
        m = Y.shape[1]
        if cost_function == "CrossEntropy":
            cost = np.squeeze(-np.sum(np.multiply(np.log(A),Y))/m)
        elif cost_function == "MSE":
            cost = np.squeeze(0.5*np.sum((Y-A)**2))
        else:
            cost = np.squeeze(-np.sum(np.multiply(np.log(A),Y))/m)
        
        L = len(parameters)//2
    
        if penalty == "l2" and lamb != 0:
            sum_weights = 0
            for l in range(1, L):
                sum_weights = sum_weights + np.sum(np.square(parameters["W" + str(l)]))
            cost = cost + sum_weights * (lamb/(2*m))
        elif penalty == "l1" and lamb != 0:
            sum_weights = 0
            for l in range(1, L):
                sum_weights = sum_weights + np.sum(np.abs(parameters["W" + str(l)]))
            cost = cost + sum_weights * (lamb/(2*m))
        return cost

    @staticmethod
    def back_propagation(AL, Y, caches, hidden_layers, penalty=None,lamb=0):
        grads = {} # Initalizing the gradients
        L = len(caches) # the number of layers
    
        m = AL.shape[1]

        Y = Y.reshape(AL.shape)

        # Initializing the backpropagation
        dZL = AL - Y

        cache = caches[L-1]
        linear_cache, activation_cache = cache
        AL, W, b = linear_cache
        grads["dW" + str(L)] = np.dot(dZL,AL.T)/m
        grads["db" + str(L)] = np.sum(dZL,axis=1,keepdims=True)/m
        grads["dA" + str(L-1)] = np.dot(W.T,dZL)
    
        # Loop from l=L-2 to l=0
        for l in reversed(range(L-1)):
            cache = caches[l]
            active_function = hidden_layers[l]
        
            linear_cache, Z = cache
            A_prev, W, b = linear_cache
            
            # Temporary gradient, the w(l+1).T error(l+1) term.
            dA_prev = grads["dA" + str(l + 1)]
        
            # Finding next error term
            if type(active_function) is tuple:
                # Leaky ReLU:
                if  active_function[0] == "relu":
                    dZ = np.multiply(dA_prev,ReLU(Z,active_function[1],derivative=True))
            else:
                if active_function == "sigmoid":
                    dZ = np.multiply(dA_prev,sigmoid(Z,derivative=True))
                elif active_function == "relu":
                    dZ = np.multiply(dA_prev,ReLU(Z,derivative=True))
                elif active_function == "tanh":
                    dZ = np.multiply(dA_prev,Tanh(Z,derivative=True))
                elif active_function == "linear":
                    dZ = np.multiply(dA_prev,linear(Z,derivative=True))

            
            grads["dA" + str(l)] = np.dot(W.T,dZ)

            
            # Adding regularization
            m = A_prev.shape[1]
            if penalty == "l2":
                grads["dW" + str(l + 1)] = (np.dot(dZ,A_prev.T)/m)  + ((lamb * W)/m)
            elif penalty == "l1":
                grads["dW" + str(l + 1)] = (np.dot(dZ,A_prev.T)/m)  + ((lamb * np.sign(W+1e-8))/m)
            else:
                grads["dW" + str(l + 1)] = (np.dot(dZ,A_prev.T)/m)
            
            grads["db" + str(l + 1)] = np.sum(dZ,axis=1,keepdims=True)/m   
        return grads


    @staticmethod
    def update_parameters(parameters, grads,learning_rate, iter_no, optimization_method = "SGD", opt_parameters=None, beta1=0.9, beta2=0.999):
        L = len(parameters) // 2 # number of layers in the neural network
        
        # Stochastic Gradient Descent Optimization Method
        if optimization_method == "SGD":
            for l in range(L):
                parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l + 1)]
                parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l + 1)]
            opt_parameters = None
        
        # Stochastic Gradient Descent with Momentum
        elif optimization_method == "SGDM":
            for l in range(L):
                opt_parameters["vdb"+str(l+1)] = beta1*opt_parameters["vdb"+str(l+1)] + (1-beta1)*grads["db" + str(l + 1)]
                opt_parameters["vdW"+str(l+1)] = beta1*opt_parameters["vdW"+str(l+1)] + (1-beta1)*grads["dW" + str(l + 1)]
                
                parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*opt_parameters["vdW"+str(l+1)]
                parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*opt_parameters["vdb"+str(l+1)]

        # Adaptive Moment Estimator
        elif optimization_method == "Adam":
            # From wikiepdia on Adaptive Moment Estimation
            for l in range(L):
                # Decaying averages:
                opt_parameters["mdW"+str(l+1)] = beta1*opt_parameters["mdW"+str(l+1)]+(1-beta1)*grads["dW"+str(l+1)]    
                opt_parameters["mdb"+str(l+1)] = beta1*opt_parameters["mdb"+str(l+1)]+(1-beta1)*grads["db"+str(l+1)]
                                                                                       
                opt_parameters["vdW"+str(l+1)] = beta2*opt_parameters["vdW"+str(l+1)]+(1-beta2)*grads["dW"+str(l+1)]**2    
                opt_parameters["vdb"+str(l+1)] = beta2*opt_parameters["vdb"+str(l+1)]+(1-beta2)*grads["db"+str(l+1)]**2
                
                # Bias correction
                opt_parameters["mvdW"+str(l+1)] = ((1-beta2**iter_no)/(1-beta1**iter_no))*opt_parameters["mdW"+str(l+1)]/(np.sqrt(opt_parameters["vdW"+str(l+1)])+1e-8)
                opt_parameters["mvdb"+str(l+1)] = ((1-beta2**iter_no)/(1-beta1**iter_no))*opt_parameters["mdb"+str(l+1)]/(np.sqrt(opt_parameters["vdb"+str(l+1)])+1e-8)
                
                # Updating
                parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*opt_parameters["mvdW"+str(l+1)]
                parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*opt_parameters["mvdb"+str(l+1)]                                                                       
        return parameters,opt_parameters

    def fit(self,X,y):
        self.grads = {}
        self.costs = []
        self.acc = []
        M = X.shape[1]
        opt_parameters = {}
        
        if self.verbose == 1:
            print('Initilizing Weights...')
        self.parameters = self.weights_init(self.layer_dims)
        idx = np.arange(0,M)
        
        if self.optimization_method != "SGD":
            for l in range(1, len(self.layer_dims)):
                opt_parameters["vdW" + str(l)] = np.zeros((self.layer_dims[l], self.layer_dims[l-1]))
                opt_parameters["vdb" + str(l)] = np.zeros((self.layer_dims[l], 1))
                opt_parameters["mdW" + str(l)] = np.zeros((self.layer_dims[l], self.layer_dims[l-1]))
                opt_parameters["mdb" + str(l)] = np.zeros((self.layer_dims[l], 1))
                opt_parameters["mvdW"+ str(l)] = np.zeros((self.layer_dims[l], self.layer_dims[l-1]))
                opt_parameters["mvdb"+ str(l)] = np.zeros((self.layer_dims[l], 1))
    
        if self.verbose == 1:
            print('Starting Training...')
        self.iter_no = 0
        for epoch_no in range(1,self.max_epoch+1):
            # Shuffling the dataset for each epoch
            np.random.shuffle(idx)
            X = X[:,idx]
            y = y[:,idx]
            for i in range(0,M, self.batch_size):
                X_batch = X[:,i:i + self.batch_size]
                y_batch = y[:,i:i + self.batch_size]
                self.iter_no +=1
                
                # Forward propagation:
                AL, cache = self.forward_propagation(X_batch,self.hidden_layers,self.parameters)

                # Backpropagation
                grads = self.back_propagation(AL, y_batch, cache,self.hidden_layers,self.penalty,self.lamb)
                #print(grads)

                # Update parameters
                self.parameters, opt_parameters = self.update_parameters(self.parameters,grads,self.learning_rate,self.iter_no,
                                                                         self.optimization_method, opt_parameters, self.beta1,self.beta2)
                
            # Cost function after each epoch
            AL, _ = self.forward_propagation(X,self.hidden_layers,self.parameters)
            cost = self.compute_cost(AL, y, self.parameters, self.cost_function, self.lamb, self.penalty)
            self.costs.append(cost)
            
            # Calculating the accuracy after each epoch
            accur = accuracy(y.T, np.argmax(AL, axis=0))
            self.acc.append(accur)
            
            if self.verbose == 1:
                if epoch_no % np.int(self.max_epoch/10) == 0:
                    print("Cost function after epoch {}: {}".format(epoch_no, cost))
        return self

    def predict(self,X,proba=False):
        out, cache = self.forward_propagation(X,self.hidden_layers,self.parameters)
        return out