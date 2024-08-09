import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# Defina o modelo
class Modelo(nn.Module):
    def __init__(self):
        super(Modelo, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Verifique se GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Crie o modelo e mova-o para o dispositivo
modelo = Modelo().to(device)

# Crie os dados de entrada e os rótulos
inputs = torch.randn(1000, 784)
labels = torch.randint(0, 10, (1000,))

# Crie o DataLoader
dataset = TensorDataset(inputs, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Defina a função de custo e otimização
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(modelo.parameters(), lr=0.001)

# Treine o modelo
num_epochs = 10
for epoch in range(num_epochs):
    modelo.train()
    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = modelo(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validação
    modelo.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = modelo(x)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}")
    print(f"Val Loss: {val_loss/len(val_loader):.4f}")
    print(f"Val Accuracy: {100 * correct / total:.2f}%")
    print("-----------------------------")

print("Treinamento concluído!")