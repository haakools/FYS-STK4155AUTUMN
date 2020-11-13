'''
Replace the analytical solution by Stochastic gradient descsend for OLS and Ridge linear regression Methods
'''
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pandas as pd
import math
from IPython.display import display



##########################################################################################
################################### Code for SGD routine #################################
##########################################################################################


def SGD(X, y, X_test, y_test, learn_rate=0.001, batch_size=30, stop_at=0.001,\
        max_epochs=500, method="OLS", lmbda=0.001, verbose=True):
    
    # Suppress warnings if verbose==False
    if verbose == False:
        import warnings
        warnings.filterwarnings("ignore")
    
    ### Keep track of iterations
    counter = 1

    # Set seed?
    np.random.seed(10)
    
    ### Initialize random beta start values
    beta_init = np.random.normal(loc=1.0,scale=0.2,size=X.shape[1])
    # Initialize optimized parameters
    beta_opt = np.ones((beta_init.shape))
    
    ### MINI BATCHES ACCORDING TO LECTURE
    m = int(len(X)/batch_size) # number of minibatches
        
    ### Randomly shuffle data
    rng = np.random.default_rng()       # Randomly permute obs. indices
    ind = rng.permutation(len(X))

    X_shuff = X[ind]
    y_shuff = y[ind]
    
    # Create batches
    X_batches = np.array_split(X_shuff,m)
    y_batches = np.array_split(y_shuff,m)

        
    ### Initialize vectors to save results
    cost_list_train = []
    
    mse_list_train = []
    mse_list_test = []
    
    
    ### Perform gradient descend until threshold of step size approached or max iterations reached ###
    while True:
        
        ### Random batch index for current iteration
        b_ind = np.random.randint(0,m)
        
        cur_X = X_batches[b_ind]
        cur_y = y_batches[b_ind]
        
        ### Calculate partial derivatives of cost function, i.e. gradient
        
            
        # Calculate gradient for current example, for both x and y partial deriv.
        if method=="OLS":
            beta_new, step_new, step_old, error_new, error_old =\
            compute_gradient(cur_X, cur_y, beta_opt, learn_rate)
        
        if method=="ridge":
            beta_new, step_new, step_old, error_new, error_old =\
            compute_gradient(cur_X, cur_y, beta_opt, learn_rate, lmbda, method="ridge")
            
        # Check if loop needs to continue
        if counter>=max_epochs:
            if verbose == True:
                print("Max epochs reached!")
            break
        elif step_new <= stop_at:
            if verbose == True:
                print("Step size below threshold!")
            break
        else:
            counter += 1
        
        
        if counter == 1:
            # Add initial cost to vector
            cost_list_train.append(step_old)
            mse_list_train.append(error_old)
            
            # Calculate test error
            pred_test = X_test @ beta_opt
            mse_list_test.append(MSE(y_test, pred_test))
            
        
        # Update beta
        beta_opt = beta_new
            
        cost_list_train.append(step_new)
        mse_list_train.append(error_new)
        
        # Calculate test error
        pred_test = X_test @ beta_opt
        mse_list_test.append(MSE(y_test, pred_test))
        
            
        #step_size = slope*learning_rate
        
        # Print feedback every 10th iteration
        #if (counter % 10) == 0:
            #print("Current RMSE: " + str(cost_new) + ", decreased by " + str(cost_old - cost_new))
        
        
    
    
    return beta_opt, cost_list_train, mse_list_train, mse_list_test


'''
Calculates the partial derivatives of a 2 variable input polynomial of the form:
f(x,y) = x^1,x^2,...,xy,...,x^n-1,y^1,...,y^1,y^2,...,y^n ; with n = polyn. degree.

The loss function to be minimized is MSE.
'''
##### Computes gradient for MSE as loss function and 2-variable input polynomial #####

def compute_gradient(X, y, beta, learning_rate=0.001,lmbda=0.001,method="OLS"):
    
    #term1 = 2*(y-(X @ beta)) # Chain rule: derivative of outer function
    
    ### Partial derivative function

    m = len(y)
    
    ### Partial derivatives?
    predict_old = X @ beta
    error_old = predict_old - y
    
    ### Different for ridge regression
    if method=="ridge":
        #XT_X = X.T @ X
        #Ridge parameter lambda
        #Id = lmbda* np.eye(XT_X.shape[0])
        
        #gradients = 2.0/m*X.T @ (X @ (beta)-y)+2*lmbda*beta
        
        step_old = 1/(2*m) * np.dot(error_old.T, error_old) + 2*lmbda*beta
        beta_update = beta-(learning_rate*step_old)
        #beta_update = beta-(learning_rate*gradients)
        
    else:
        #error_old = predict_old - y
        step_old = 1/(2*m) * np.dot(error_old.T, error_old)   #  (1/2m)*sum[(error)^2]
        beta_update = beta - (learning_rate * (1/m) * np.dot(X.T, error_old))

    
    mse_old = MSE(y, predict_old)

    
    ### Compute new error!
    predict_new = X @ beta_update
    error_new = predict_new - y
    step_new = (0.5*m)*np.dot(error_new.T, error_new)
    
    mse_new = MSE(y, predict_new)
    
    return beta_update, step_new, step_old, mse_new, mse_old



'''
OLD FUNCTIONS NEEDED AGAIN HERE
'''


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
        

def create_X(x, y, n ):
    """
    Inputs:
        x  : x part of meshgrid of x,y
        y  : y part of meshgrid of x,y
        n  : Polymomial degree of the fit, (x+y)^n
    Creates the design matrix, where the columns are ordered as
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

def init_data(N,noisefactor):
    """
    Input: 
        N = datapoints, 
        noisefactor = scalar for gaussian distributed noise
    
    Output: 
        x_ = meshgrid of datapoints to compute the Franke Function in X directions
        y_ = meshgrid of datapoints to compute the Franke Function in Y directions
        z  = 2D array of the frankefunction

    """
    x, y = np.linspace(0,1,N), np.linspace(0,1,N)
    x_,y_ = np.meshgrid(x,y)
    z = FrankeFunction(x_, y_, noisefactor)
    return x_, y_, z    

def Scaling(X_train, X_test):
    
    """ 
    Input:
        X_train   : Training part of the design matrix
        X_test    : testing part of the design matrix
    
    Scales the Design matrix with standard scaler. Removes the intercept as it is just 1s and adds it after the scaling.
    
    Output:
        X_train   : Training design matrix scaled
        X_test   : Testing design matrix scaled
    """
    (N, p) = X_train.shape
    if p > 1:
        scaler = StandardScaler()
        scaler.fit(X_train[:,1:])
        X_train = scaler.transform(X_train[:,1:])
        X_test = scaler.transform(X_test[:,1:])

        # Adding the intercept after the scaling, as the StandardScaler removes the 1s in the first column.
        intercept_train = np.ones((len(X_train),1))
        intercept_test = np.ones((len(X_test),1))
        X_train = np.concatenate((intercept_train,X_train),axis=1)
        X_test = np.concatenate((intercept_test,X_test),axis=1)
    else:
        X_train = X_train
        X_test = X_test
    return X_train, X_test


def PreProcess(x, y,z, test_size, n):
    """
    Input:
        x         : The meshgrid datapoints for x
        y         : The meshgrid datapoints for y
        z         : The FrankeFunction datapoints
        test_size : The test_size for testing training datasplit
        n         : The maximum number of polynomial degree, (x+y)^n, that the functions can fit.
    
    Ravels the x, y and z
    
    Output:
        X_train   : Training Design matrix
        X_test    : Testing Design Matrix
        z_train   : Train Data
        z_test    : Test Data
    """
    z = np.ravel(z)
    # Creating design matrix for the maximum polynomial degree. 
    X = create_X(x,y,n)

    # Test Train splitting of data
    X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

    # Scaling the train and test set
    X_train, X_test = Scaling(X_train, X_test)
    return X_train, X_test, z_train, z_test 
    

def MSE(y,ypred):
    """
    Input:
        y      :  The true data
        ypred  :  The predicted data
    Computes the MSE error
    Output:
        MSE : Mean Squared Error
    """
    MSE = np.mean((y-ypred)**2)
    return MSE