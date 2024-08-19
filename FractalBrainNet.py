import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Definindo um bloco fractal simples
class FractalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth, dropout_prob=0.0):
        super(FractalBlock, self).__init__()
        self.depth = depth
        self.dropout_prob = dropout_prob

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        
        if depth > 1:
            self.sub_block1 = FractalBlock(out_channels, out_channels, depth - 1, dropout_prob)
            self.sub_block2 = FractalBlock(out_channels, out_channels, depth - 1, dropout_prob)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        if self.depth > 1:
            x1 = self.sub_block1(x)
            x2 = self.sub_block2(x)
            x = x1 + x2
        x = self.relu(self.conv2(x))
        if self.dropout_prob > 0.0:
            x = self.dropout(x)
        return x

# Definindo a rede neural usando blocos fractais
class FractalBrainNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, depth=3, dropout_prob=0.0):
        super(FractalBrainNet, self).__init__()
        self.fractal_block = FractalBlock(in_channels, 32, depth, dropout_prob)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.fractal_block(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Configuração de treinamento
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')

# Configuração de teste
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.CrossEntropyLoss(reduction='sum')(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')
    return accuracy

# Preparando o dataset e o treinamento
def main():
    # Configurações básicas
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 10

    # Transformação e carregamento do dataset MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transform),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('../data', train=False, transform=transform),
                             batch_size=batch_size, shuffle=False)

    # Inicializando a rede neural e otimizador
    model = FractalBrainNet(depth=3, dropout_prob=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Treinamento e teste da rede
    best_accuracy = 0
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        accuracy = test(model, device, test_loader)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "fractalbrainnet_mnist.pt")

    print(f'Melhor precisão no teste: {best_accuracy:.2f}%')

if __name__ == '__main__':
    main()
