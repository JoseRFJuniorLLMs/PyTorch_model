import torch
import torch.nn as nn
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size)  # Estado oculto inicial
        c0 = torch.zeros(1, x.size(0), hidden_size)  # Estado da célula inicial
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Usa a saída do último passo de tempo
        return out

# Parâmetros
input_size = 10
hidden_size = 20
output_size = 1

# Criando o modelo
model = LSTMModel(input_size, hidden_size, output_size)

# Definindo a função de perda e o otimizador
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Dados de exemplo
inputs = torch.randn(5, 7, input_size)  # 5 amostras, 7 passos de tempo, 10 características
targets = torch.randn(5, output_size)   # 5 amostras, 1 valor alvo

# Treinamento do modelo
model.train()
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    print(f'Época [{epoch+1}/10], Perda: {loss.item()}')
