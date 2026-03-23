import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np

# -----------------------------
# Classical CNN
# -----------------------------
class ClassicalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16*7*7, 6)
        self.fc2 = nn.Linear(6, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# -----------------------------
# Hybrid QNN
# -----------------------------
n_qubits = 6
n_layers = 4

dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))

    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RY(weights[l, i], wires=i)

        for i in range(n_qubits-1):
            qml.CNOT(wires=[i, i+1])

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits)}
quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)


class HybridQNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)

        self.fc = nn.Linear(16*7*7, n_qubits)
        self.quantum = quantum_layer
        self.classifier = nn.Linear(n_qubits, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = torch.tanh(x) * np.pi
        x = self.quantum(x)

        return self.classifier(x)