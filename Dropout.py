import torch
import torch.nn as nn
import torch.optim as optim

class DropoutNN(nn.Module):
    def __init__(self):
        super(DropoutNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.dropout = nn.Dropout(0.5)  # 50% de dropout
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Criando o modelo
model = DropoutNN()

# Definindo a função de perda e o otimizador
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dados de exemplo
inputs = torch.randn(5, 10)  # 5 amostras, cada uma com 10 características
targets = torch.randn(5, 1)  # 5 amostras, cada uma com 1 valor alvo

# Treinamento do modelo
model.train()
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    print(f'Época [{epoch+1}/10], Perda: {loss.item()}')
