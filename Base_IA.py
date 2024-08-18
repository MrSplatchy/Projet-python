import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from utilities import load_data
import tqdm

# Generation des donnees
X_train, y_train, X_test, y_test = load_data()

# Fonctions du modèle logistique
def Initialiser(Dims):
    Params = {}
    C = len(Dims)

    for c in range(1, C):
        Params['W' +  str(c)] = np.random.randn(Dims[c], Dims[c - 1])
        Params['b' +  str(c)] = np.random.randn(Dims[c], 1)

    return Params

def ForwardPropagation(X, Params):
    Activations = {'A0': X}
    C = len(Params) // 2

    for c in range(1, C + 1):  
        Z = np.dot(Params['W' + str(c)], Activations['A' + str(c - 1)]) + Params['b' + str(c)] 
        Activations['A' + str(c)] = 1 / (1 + np.exp(-Z))  # Sigmoid activation function

    return Activations

def LogLoss(A, y):
    epsilon = 1e-15  # Pour éviter les log(0)
    m = y.shape[1]
    return -1 / m * np.sum(y * np.log(A + epsilon) + (1 - y) * np.log(1 - A + epsilon))

def BackPropagation(Activations, Params, y):
    m = y.shape[1]
    C = len(Params) // 2
    Gradients = {}

    dZ = Activations['A' + str(C)] - y

    for c in reversed(range(1, C + 1)):
        Gradients['dW' + str(c)] = 1 / m * np.dot(dZ, Activations['A' + str(c-1)].T)
        Gradients['db' + str(c)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        
        if c > 1:
            dZ = np.dot(Params['W' + str(c)].T, dZ) * Activations['A' + str(c-1)] * (1 - Activations['A' + str(c-1)])

    return Gradients

def Update(Params, Gradients, learning_rate):
    C = len(Params) // 2

    for c in range(1, C + 1):
        Params['W' + str(c)] -= learning_rate * Gradients['dW' + str(c)]
        Params['b' + str(c)] -= learning_rate * Gradients['db' + str(c)]
    return Params

def Predict(X, Params):
    Activations = ForwardPropagation(X, Params)
    A = Activations['A' + str(len(Activations) // 2)]
    return (A >= 0.5).astype(int)  # Use 0.5 as the threshold

def NeuronNetwork(X_train, y_train, X_test, y_test, hidden_layers=(32, 32, 32), n_iter=10000, learning_rate=0.01):
    Dims = [X_train.shape[0]] + list(hidden_layers) + [y_train.shape[0]]
    Params = Initialiser(Dims)

    TrainLoss, TrainAcc, TestLoss, TestAcc = [], [], [], []

    for i in tqdm.tqdm(range(n_iter)):
        Activations_train = ForwardPropagation(X_train, Params)
        Gradients = BackPropagation(Activations_train, Params, y_train)
        Params = Update(Params, Gradients, learning_rate)

        if i % 10 == 0:
            C = len(Params) // 2
            A_final_train = Activations_train['A' + str(C)]
            TrainLoss.append(LogLoss(A_final_train, y_train)) 
            y_pred_train = Predict(X_train, Params)
            TrainAcc.append(accuracy_score(y_train.flatten(), y_pred_train.flatten()))  # Flatten to ensure matching shapes
            
            Activations_test = ForwardPropagation(X_test, Params)
            A_final_test = Activations_test['A' + str(C)]
            TestLoss.append(LogLoss(A_final_test, y_test)) 
            y_pred_test = Predict(X_test, Params)
            TestAcc.append(accuracy_score(y_test.flatten(), y_pred_test.flatten()))  # Flatten to ensure matching shapes

    print(f"Bonne prédiction à {accuracy_score(y_test.flatten(), y_pred_test.flatten()) * 100:.2f}%")

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

# Normalisation et reshape des données
X_train = X_train.T
X_train = X_train.reshape(X_train.shape[0], -1)  # Reshape en 2D
X_train = X_train / np.max(X_train)

X_test = X_test.T
X_test = X_test.reshape(X_test.shape[0], -1)  # Reshape en 2D
X_test = X_test / np.max(X_train)  # Utiliser le max de X_train pour normaliser X_test

y_train = y_train.T
y_train = y_train.reshape(1, -1)  # Assurer que y est un vecteur ligne

y_test = y_test.T
y_test = y_test.reshape(1, -1)  # Assurer que y est un vecteur ligne

# Limiter le nombre d'exemples si nécessaire
m_train = min(1000, X_train.shape[1])
m_test = min(200, X_test.shape[1])

X_train = X_train[:, :m_train]
y_train = y_train[:, :m_train]
X_test = X_test[:, :m_test]
y_test = y_test[:, :m_test]

print("Formes après prétraitement:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

Params = NeuronNetwork(X_train, y_train, X_test, y_test, n_iter=50000, learning_rate=0.0005)
