import numpy as np
import pickle
from .utils import *

# L-layer Deep Neural Network for multi-class classification.
class DeepNeuralNetwork:

    def __init__(self, layer_dims, learning_rate=0.009, dropout_rates=None, lambd=0.0):
        self.layer_dims = layer_dims
        self.learning_rate = learning_rate
        self.parameters = initialize_parameters_deep(layer_dims)
        self.costs = []
        self.lambd = lambd
        
        # Dropout rates per layer (None = no dropout)
        if dropout_rates is None:
            # Default: progressive dropout by depth
            self.dropout_rates = [0.0] + [0.2, 0.4, 0.5][:len(layer_dims)-2] + [0.0]
        else:
            self.dropout_rates = dropout_rates


    def train(self, X, Y, num_iterations=2500, print_cost=True):
        dropout_caches = []

        for i in range(num_iterations):
        # Forward propagation WITH DROPOUT
            caches = []
            dropout_caches = []
            A = X
            L = len(self.parameters) // 2
            
            # Hidden layers with dropout
            for l in range(1, L):
                A_prev = A
                A, cache = linear_activation_forward(
                    A_prev, 
                    self.parameters['W' + str(l)], 
                    self.parameters['b' + str(l)], 
                    activation="relu"
                )
                caches.append(cache)
                
                # Apply dropout AFTER activation
                if self.dropout_rates[l] > 0:
                    A, mask = dropout_forward(A, keep_prob=1-self.dropout_rates[l])
                    dropout_caches.append((l, mask, 1-self.dropout_rates[l]))
                else:
                    dropout_caches.append(None)
            
            # Output layer (NO dropout)
            AL, cache = linear_activation_forward(
                A, 
                self.parameters['W' + str(L)], 
                self.parameters['b' + str(L)], 
                activation="softmax"
            )
            caches.append(cache)
            
            # Compute cost
            cost = compute_cost(AL, Y, self.parameters, lambd=self.lambd)
            
            # Backward propagation WITH DROPOUT
            grads = {}
            m = AL.shape[1]
            
            # Convert Y to one-hot
            Y_one_hot = np.zeros_like(AL)
            Y_one_hot[Y.astype(int), np.arange(m)] = 1
            
            # Output layer gradient
            dAL = AL - Y_one_hot
            current_cache = caches[L-1]
            linear_cache, _ = current_cache
            dA_prev_temp, dW_temp, db_temp = linear_backward(dAL, linear_cache, lambd=self.lambd)
            grads["dA" + str(L-1)] = dA_prev_temp
            grads["dW" + str(L)] = dW_temp
            grads["db" + str(L)] = db_temp
            
            # Hidden layers with dropout
            for l in reversed(range(L-1)):
                # Apply dropout to gradient if it was used in forward pass
                if dropout_caches[l] is not None:
                    layer_idx, mask, keep_prob = dropout_caches[l]
                    grads["dA" + str(l + 1)] = dropout_backward(
                        grads["dA" + str(l + 1)], mask, keep_prob
                    )
                
                current_cache = caches[l]
                dA_prev_temp, dW_temp, db_temp = linear_activation_backward(
                    grads["dA" + str(l + 1)], current_cache, activation="relu", lambd=self.lambd
                )
                grads["dA" + str(l)] = dA_prev_temp
                grads["dW" + str(l + 1)] = dW_temp
                grads["db" + str(l + 1)] = db_temp
            
            # Update parameters
            self.parameters = update_parameters(
                self.parameters, grads, self.learning_rate
            )
            
            # Record cost
            if i % 100 == 0:
                self.costs.append(cost)
                if print_cost:
                    print(f"Cost after iteration {i}: {cost:.6f}")
    
        return self.parameters, self.costs
 

    def predict(self, X):
        AL, _ = L_model_forward(X, self.parameters)
        predictions = np.argmax(AL, axis=0).reshape(1, -1)
        return predictions
    

    def evaluate(self, X, Y):
        predictions = self.predict(X)
        accuracy = np.mean(predictions == Y)
        return accuracy
    

    def save(self, filepath):
        # Save model parameters to file.
        with open(filepath, 'wb') as f:
            pickle.dump({
                'parameters': self.parameters,
                'layer_dims': self.layer_dims,
                'learning_rate': self.learning_rate,
                'costs': self.costs
            }, f)
        print(f"Model saved to {filepath}")
    

    @classmethod
    def load(cls, filepath):
        # Load model from file.
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(data['layer_dims'], data['learning_rate'])
        model.parameters = data['parameters']
        model.costs = data['costs']
        print(f"Model loaded from {filepath}")
        return model
    
