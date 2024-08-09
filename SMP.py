import torch
import torch.nn as nn
import torch.optim as optim

# Definindo o modelo
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # Camada totalmente conectada (10 entradas, 50 saídas)
        self.fc2 = nn.Linear(50, 1)   # Camada totalmente conectada (50 entradas, 1 saída)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Função de ativação ReLU
        x = self.fc2(x)
        return x

# Criando o modelo
model = SimpleNN()

# Definindo a função de perda e o otimizador
criterion = nn.MSELoss()  # Erro quadrático médio
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Otimizador Stochastic Gradient Descent

# Dados de exemplo
inputs = torch.randn(5, 10)  # 5 amostras, cada uma com 10 características
targets = torch.randn(5, 1)  # 5 amostras, cada uma com 1 valor alvo

# Treinamento do modelo
model.train()  # Coloca o modelo em modo de treinamento
for epoch in range(100):  # Número de épocas
    optimizer.zero_grad()  # Zera os gradientes dos parâmetros
    outputs = model(inputs)  # Faz a previsão
    loss = criterion(outputs, targets)  # Calcula a perda
    loss.backward()  # Calcula os gradientes
    optimizer.step()  # Atualiza os parâmetros

    if (epoch + 1) % 10 == 0:
        print(f'Época [{epoch+1}/100], Perda: {loss.item()}')
