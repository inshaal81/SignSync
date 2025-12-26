import numpy as np
import pandas as pd

    # BASIC FUNCS.
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    A = np.maximum(0,Z)
    assert(A.shape == Z.shape)
    cache = Z 
    
    return A, cache

def softmax(Z):
    # Softmax activation for multi-class classification 
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Numerical stability
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    cache = Z
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) # converting dz to a correct object
    dZ[Z <= 0] = 0 # When z <= 0, should set dz to 0
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache): 
    Z = cache
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    assert (dZ.shape == Z.shape)
    
    return dZ


    # DATA 
def load_data(): 
    # Load ASL alphabet dataset from CSV files.
    train_df = pd.read_csv('data/datasets/sign_mnist_train.csv')
    test_df = pd.read_csv('data/datasets/sign_mnist_test.csv')
    
    # Separate features and labels
    train_x = train_df.drop('label', axis=1).values.T # train set features 
    train_y_raw = train_df['label'].values  # assign raw labels first
    
    test_x = test_df.drop('label', axis=1).values.T # test set features 
    test_y_raw = test_df['label'].values  # assign raw labels first
    
    # The dataset has labels 0-25, but we only want 0-23
    # Remove samples with labels 9 (J) and 25 (Z) if they exist
    train_mask = (train_y_raw != 9) & (train_y_raw != 25)
    train_x = train_x[:, train_mask]
    train_y_raw = train_y_raw[train_mask]
    
    test_mask = (test_y_raw != 9) & (test_y_raw != 25)
    test_x = test_x[:, test_mask]
    test_y_raw = test_y_raw[test_mask]

    # Remap labels: 0-8 stay same, 10-24 become 9-23
    def remap_labels(labels):
        remapped = labels.copy()
        remapped[labels > 9] -= 1  # Shift down labels after J
        return remapped
    
    train_y = remap_labels(train_y_raw).reshape(1, -1)
    test_y = remap_labels(test_y_raw).reshape(1, -1)

    # Normalize pixel values
    train_x = train_x / 255.0
    test_x = test_x / 255.0
    
    # Class names (A-Z minus J and Z)
    classes = np.array([chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']])
    
    return train_x, train_y, test_x, test_y, classes

def initialize_parameters(n_x, n_h, n_y): ## edit this
    np.random.seed(1)
    
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    
    return parameters     

def initialize_parameters_deep(layer_dims):
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) 
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
    return parameters


    # FORWARD
def linear_forward(A, W, b):
    Z = W.dot(A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    elif activation == "softmax":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)

    else:
        print("Error! Please make sure you have passed the value correctly in the \"activation\" parameter")
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2 # number of layers in the neural network
    
    # Implement [LINEAR -> RELU]*(L-1)
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    
    # Implement LINEAR -> SOFTMAX 
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation="softmax")
    caches.append(cache)
            
    return AL, caches

def dropout_forward(A, keep_prob):
    """
    Apply dropout during forward propagation.
    
    Parameters:
    A: activations from previous layer
    keep_prob: probability of keeping a neuron (0.5 = 50% dropout)
    
    Returns:
    A: activations with dropout applied
    mask: dropout mask (for backprop)
    """
    mask = np.random.rand(*A.shape) < keep_prob
    A = A * mask
    A = A / keep_prob  # Inverted dropout - scale up to maintain expected value
    
    return A, mask


    # COST
def compute_cost(AL, Y, parameters=None, lambd=0.0):
    # Categorical cross-entropy cost for multi-class classification.
    # AL: shape (24, m) - predictions for 24 classes
    # Y: shape (1, m) - true labels (integers 0-23)

    m = Y.shape[1]
    # Convert Y to one-hot encoding
    Y_one_hot = np.zeros((AL.shape[0], m))
    Y_one_hot[Y.astype(int), np.arange(m)] = 1
    # Categorical cross-entropy
    cost = -np.sum(Y_one_hot * np.log(AL + 1e-8)) / m
    
    # L2 regularization
    if lambd > 0 and parameters is not None:
        L = len(parameters) // 2
        L2_cost = 0
        for l in range(1, L+1):
            L2_cost += np.sum(np.square(parameters['W' + str(l)]))
        L2_cost = (lambd / (2 * m)) * L2_cost
        cost = cost + L2_cost
    cost = np.squeeze(cost)
    
    return cost


    # BACKWARD
def linear_backward(dZ, cache, lambd=0.0):
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T) + (lambd/m) * W  # Add L2 gradient
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation, lambd=0.0):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd=lambd)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd=lambd)
        
    else:
        print("Error! Please make sure you have passed the value correctly in the \"activation\" parameter")
    
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) # number of layers
    m = AL.shape[1]

    # Convert Y to one-hot
    Y_one_hot = np.zeros_like(AL)
    Y_one_hot[Y.astype(int), np.arange(m)] = 1    
    
    dAL = AL - Y_one_hot # Softmax gradient 
    # Lth layer (SIGMOID -> LINEAR) gradients
    current_cache = caches[L-1]
    linear_cache, _ = current_cache
    dA_prev_temp, dW_temp, db_temp = linear_backward(dAL, linear_cache)
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def dropout_backward(dA, mask, keep_prob):
    """
    Apply dropout during backpropagation.
    
    Parameters:
    dA: gradient from next layer
    mask: dropout mask from forward pass
    keep_prob: same as forward
    
    Returns:
    dA: gradient with dropout applied
    """
    dA = dA * mask
    dA = dA / keep_prob
    
    return dA


    # UPDATE PARAMS.
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2 # layers in the neural network
    # Update rule for each parameter 
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

