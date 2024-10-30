from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris 
import matplotlib.pyplot as plt
import numpy as np

from PrayagPiya_MPL import NeuralNetwork
from tenserflow_MPL import MLP
from mlxextender_MPL import fit

def plotting(loss, accuracy, function_name):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss, label='Loss')
    plt.plot(accuracy, label='Accuracy')
    plt.title('Loss / Accuracy over epochs ' + function_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    plt.show()

def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    #hot encode the y values 
    encoder = OneHotEncoder(sparse_output=False)
    y_new = encoder.fit_transform(y.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(X, y_new, test_size=0.2, random_state=42)

    print('Result from my NN')
    nn = NeuralNetwork(4, 3, 5, 2)
    nn.fit(X_train, y_train, epochs=10000)
    plotting(nn.losses, nn.accuracies, "My Nueral Network")
    y_test_pred = nn.predict(X_test)

    
    print("\n")
    print(f"Test Accuracy : NN {nn.accuracy(y_test, encoder.transform(y_test_pred.reshape(-1, 1))):.4f}")

main()