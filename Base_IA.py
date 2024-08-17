import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from utilities import load_data
import tqdm

# Generation des donnees
X_train, y_train, X_test, y_test = load_data()

# Fonctions du modèle logistique
def Initialiser(n0, n1, n2):
    W1 = np.random.randn(n1, n0) * 0.01
    b1 = np.zeros((n1, 1))
    W2 = np.random.randn(n2, n1) * 0.01
    b2 = np.zeros((n2, 1))

    Params = {
        'W1': W1,
        'W2': W2,
        'b1': b1,
        'b2': b2,
    }
    return Params

def ForwardPropagation(X, Params):
    W1, W2, b1, b2 = Params['W1'], Params['W2'], Params['b1'], Params['b2']
    
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)  # Tanh activation function
    Z2 = np.dot(W2, A1) + b2
    A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid activation function
    
    Activations = {
        'A1': A1,
        'A2': A2,
    }    
    return Activations

def LogLoss(Activations, y):
    A2 = Activations['A2']
    epsilon = 1e-15  # Pour éviter les log(0)
    return -1 / y.shape[1] * np.sum(y * np.log(A2 + epsilon) + (1 - y) * np.log(1 - A2 + epsilon))

def BackPropagation(Activations, Params, X, y):
    A1 = Activations['A1']
    A2 = Activations['A2']
    W2 = Params['W2']

    m = y.shape[1]
    dZ2 = A2 - y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))  # Derivative of tanh
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

    Gradients = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2,
    }
    return Gradients

def Update(Params, Gradients, learning_rate):
    Params['W1'] -= learning_rate * Gradients['dW1']
    Params['b1'] -= learning_rate * Gradients['db1']
    Params['W2'] -= learning_rate * Gradients['dW2']
    Params['b2'] -= learning_rate * Gradients['db2']
    return Params

def Predict(X, Params):
    Activations = ForwardPropagation(X, Params)
    A2 = Activations['A2']
    return (A2 >= 0.5).astype(int)  # Use 0.5 as the threshold

def NeuronNetwork(X_train, y_train, X_test, y_test, n1, n_iter=10000, learning_rate=0.01):
    n0 = X_train.shape[0]
    n2 = y_train.shape[0]
    Params = Initialiser(n0, n1, n2)

    TrainLoss = []
    TrainAcc = []
    TestLoss = []
    TestAcc = []

    for i in tqdm.tqdm(range(n_iter)):
        Activations_train = ForwardPropagation(X_train, Params)
        Gradients = BackPropagation(Activations_train, Params, X_train, y_train)
        Params = Update(Params, Gradients, learning_rate)

        if i % 10 == 0:
            TrainLoss.append(LogLoss(Activations_train, y_train)) 
            y_pred_train = Predict(X_train, Params)
            TrainAcc.append(accuracy_score(y_train.flatten(), y_pred_train.flatten()))
            
            Activations_test = ForwardPropagation(X_test, Params)
            TestLoss.append(LogLoss(Activations_test, y_test)) 
            y_pred_test = Predict(X_test, Params)
            TestAcc.append(accuracy_score(y_test.flatten(), y_pred_test.flatten()))

    print("Bonne prédiction à " + str(accuracy_score(y_test.flatten(), y_pred_test.flatten()) * 100) + '%')

    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.plot(TrainLoss, label='Train Loss')
    plt.plot(TestLoss, label='Test Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(TrainAcc, label='Train Accuracy')    
    plt.plot(TestAcc, label='Test Accuracy')
    plt.legend()
    plt.show()
        
    return Params

# Normalisation des données
X_train = X_train.T
X_train_reshape = X_train.reshape(-1, X_train.shape[-1]) / X_train.max()
X_test = X_test.T
X_test_reshape = X_test.reshape(-1, X_test.shape[-1]) / X_train.max()

y_train = y_train.T
y_test = y_test.T

m_train = 1000
m_test = 200
X_test_reshape = X_test_reshape[:, :m_test]
y_test = y_test[:, :m_test]
X_train_reshape = X_train_reshape[:, :m_train]
y_train = y_train[:, :m_train]

print(X_train_reshape.shape)
print(y_train.shape)
print(X_test_reshape.shape)
print(y_test.shape)

Params = NeuronNetwork(X_train_reshape, y_train, X_test_reshape, y_test, n1=16, n_iter=50000, learning_rate=0.0005)
