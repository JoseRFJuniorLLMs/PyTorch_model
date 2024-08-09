"""
Resumo dos Passos:
Gerar Dados Sintéticos: Crie amostras e rótulos binários.
Transformar Dados: Converta para tensores PyTorch.
Definir o Modelo: Crie uma rede neural para a tarefa.
Configurar Treinamento: Defina perda e otimizador.
Treinar o Modelo: Execute o loop de treinamento.
Testar o Modelo: Avalie a precisão ou o desempenho.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Função para gerar dados sintéticos
def generate_data(n_samples=100, n_features=10):
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (np.sum(X, axis=1) > 0).astype(np.float32)  # Rótulo binário baseado na soma das características
    return torch.tensor(X), torch.tensor(y)

# Gerando dados sintéticos
X, y = generate_data()

# Definindo o modelo simples
class BinaryNN(nn.Module):
    def __init__(self):
        super(BinaryNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Criando o modelo
model = BinaryNN()

# Definindo a função de perda e o otimizador
criterion = nn.BCELoss()  # Função de perda para classificação binária
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Treinamento do modelo
model.train()
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X).squeeze()
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()

    print(f'Época [{epoch+1}/10], Perda: {loss.item()}')

# Testar o modelo (usando os mesmos dados sintéticos)
model.eval()  # Coloca o modelo em modo de avaliação
with torch.no_grad():
    outputs = model(X).squeeze()
    predicted = outputs.round()  # Arredonda as previsões para 0 ou 1
    accuracy = (predicted == y).float().mean()
    print(f'Precisão: {accuracy.item()}')
