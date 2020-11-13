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

    
    
# One-hot in numpy (from lecture notes week 41)
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector    

# Activation Functions
def sigmoid(X,derivative=False):
    if derivative == False:
        out = 1 / (1 + np.exp(-np.array(X)))
    elif derivative == True:
        s = 1 / (1 + np.exp(-np.array(X)))
        out = s*(1-s)
    return out

def ReLU(X,alpha=0,derivative=False):
    '''Compute ReLU function and derivative'''
    X = np.array(X,dtype=np.float64)
    if derivative == False:
        return np.where(X<0,alpha*X,X)
    elif derivative == True:
        X_relu = np.ones_like(X,dtype=np.float64)
        X_relu[X < 0] = alpha
        return X_relu

def Tanh(X,derivative=False):
    '''Compute tanh values and derivative of tanh'''
    X = np.array(X)
    if derivative == False:
        return np.tanh(X)
    if derivative == True:
        return 1 - (np.tanh(X))**2

def softmax(X):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(X-np.max(X)) / np.sum(np.exp(X-np.max(X)),axis=0)

def linear(X,derivative=False):
    X = np.array(X)
    if derivative ==  False:
        return X
    if derivative == True:
        return np.ones_like(X)
def softmax(X):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(X) / np.sum(np.exp(X),axis=0)

# Neural Network Function
# See this for inspiration:
# https://compphysics.github.io/MachineLearning/doc/pub/week41/html/week41.html
# under "Full object-oriented implementation"

class NN(object):

    def __init__(self,layer_dims,hidden_layers, cost_function, learning_rate=0.1,
                 optimization_method = "SGD",batch_size=64,max_epoch=100,penality=None,beta1=0.9,
                 beta2=0.999,lamda=0,verbose=0):
        self.layer_dims = layer_dims
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.verbose = verbose
        
        self.optimization_method = optimization_method
        self.cost_function = cost_function
        self.penality = penality
        self.lamda = lamda
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

            cache = ((A_prev, parameters["W" + str(l)],parameters["b" + str(l)]), Z)
            caches.append(cache)      
        return A, caches
    
    
    @staticmethod
    def compute_cost(A, Y, parameters, cost_function, lamda=0,penality=None):
        m = Y.shape[1]
        if cost_function == "CrossEntropy":
            cost = np.squeeze(-np.sum(np.multiply(np.log(A),Y))/m)
        elif cost_function == "MSE":
            cost = np.squeeze(0.5*np.sum((Y-A)**2))
        else:
            cost = np.squeeze(-np.sum(np.multiply(np.log(A),Y))/m)
        
        L = len(parameters)//2
    
        if penality == "l2" and lamda != 0:
            sum_weights = 0
            for l in range(1, L):
                sum_weights = sum_weights + np.sum(np.square(parameters["W" + str(l)]))
            cost = cost + sum_weights * (lamda/(2*m))
        elif penality == "l1" and lamda != 0:
            sum_weights = 0
            for l in range(1, L):
                sum_weights = sum_weights + np.sum(np.abs(parameters["W" + str(l)]))
            cost = cost + sum_weights * (lamda/(2*m))
        return cost

    @staticmethod
    def back_propagation(AL, Y, caches, hidden_layers, penality=None,lamda=0):
        grads = {}
        L = len(caches) # the number of layers
    
        m = AL.shape[1]

        Y = Y.reshape(AL.shape)

        # Initializing the backpropagation
        dZL = AL - Y
        """
        # NOTE! If we wanted to generalize the error term we could write something like below for all different last layer activation functions.
        However as pointed out in the methods, using only Cross entropy and MSE as cost functions, the error is the same. Therefore it is omitted to
        alleviate writing more
        another block checking for all the different activationfunctions
        
        cache = caches[L]; linear_cache, Z = cache
        active_function = hidden_layers[L]
        if active_function == "linear":
            error_L = (AL - Y).dot(linear(Z), derivative=False)
        elif active_function == "relu"
        ...
        """
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
                if  active_function[0] == "relu":
                    dZ = np.multiply(dA_prev,ReLU(Z,active_function[1],derivative=True))
                elif active_function[0] == "elu":
                    dZ = np.multiply(dA_prev,elu(Z,active_function[1],derivative=True))
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
            if penality == "l2":
                grads["dW" + str(l + 1)] = (np.dot(dZ,A_prev.T)/m)  + ((lamda * W)/m)
            elif penality == "l1":
                grads["dW" + str(l + 1)] = (np.dot(dZ,A_prev.T)/m)  + ((lamda * np.sign(W+1e-8))/m)
            else:
                grads["dW" + str(l + 1)] = (np.dot(dZ,A_prev.T)/m)
            
            grads["db" + str(l + 1)] = np.sum(dZ,axis=1,keepdims=True)/m   
        return grads


    @staticmethod
    def update_parameters(parameters, grads,learning_rate, iter_no, optimization_method = "SGD", opt_parameters=None, beta1=0.9, beta2=0.999):
        L = len(parameters) // 2 # number of layers in the neural network
        if optimization_method == "SGD":
            for l in range(L):
                parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l + 1)]
                parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l + 1)]
            opt_parameters = None
        elif optimization_method == "SGDM":
            for l in range(L):
                opt_parameters["vdb"+str(l+1)] = beta1*opt_parameters["vdb"+str(l+1)] + (1-beta1)*grads["db" + str(l + 1)]
                opt_parameters["vdW"+str(l+1)] = beta1*opt_parameters["vdW"+str(l+1)] + (1-beta1)*grads["dW" + str(l + 1)]
                
                parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*opt_parameters["vdW"+str(l+1)]
                parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*opt_parameters["vdb"+str(l+1)]
        elif optimization_method == "Adam":
            # From wikiepida on Adaptive Moment Estimation
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
                grads = self.back_propagation(AL, y_batch, cache,self.hidden_layers,self.penality,self.lamda)
                #print(grads)

                # Update parameters
                self.parameters, opt_parameters = self.update_parameters(self.parameters,grads,self.learning_rate,self.iter_no,
                                                                         self.optimization_method, opt_parameters, self.beta1,self.beta2)
                
            # Cost function after each epoch
            AL, _ = self.forward_propagation(X,self.hidden_layers,self.parameters)
            cost = self.compute_cost(AL, y, self.parameters, self.cost_function, self.lamda, self.penality)
            self.costs.append(cost)
            
            if self.verbose == 1:
                if epoch_no % np.int(self.max_epoch/10) == 0:
                    print("Cost function after epoch {}: {}".format(epoch_no, cost))
        return self

    def predict(self,X,proba=False):
        out, _ = self.forward_propagation(X,self.hidden_layers,self.parameters)
        if proba == True:
            return out.T
        else:
            return np.argmax(out, axis=0)
