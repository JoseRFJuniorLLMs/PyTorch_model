import torch
import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 canal de entrada, 32 canais de saída
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 canais de entrada, 64 canais de saída
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Camada totalmente conectada (entrada: 64*7*7, saída: 128)
        self.fc2 = nn.Linear(128, 10)  # Camada totalmente conectada (entrada: 128, saída: 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # Convolução + ReLU
        x = torch.max_pool2d(x, 2)     # Pooling
        x = torch.relu(self.conv2(x))  # Convolução + ReLU
        x = torch.max_pool2d(x, 2)     # Pooling
        x = x.view(-1, 64 * 7 * 7)    # Achatar para a camada totalmente conectada
        x = torch.relu(self.fc1(x))   # Camada totalmente conectada + ReLU
        x = self.fc2(x)               # Camada totalmente conectada (saída)
        return x

# Criando o modelo
model = SimpleCNN()

# Definindo a função de perda e o otimizador
criterion = nn.CrossEntropyLoss()  # Perda de entropia cruzada para classificação
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Otimizador Adam

# Dados de exemplo (imagens 1x28x28 e rótulos)
inputs = torch.randn(5, 1, 28, 28)  # 5 amostras, 1 canal, 28x28 pixels
targets = torch.randint(0, 10, (5,))  # 5 rótulos para 10 classes

# Treinamento do modelo
model.train()
for epoch in range(10):  # Número de épocas
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    print(f'Época [{epoch+1}/10], Perda: {loss.item()}')
