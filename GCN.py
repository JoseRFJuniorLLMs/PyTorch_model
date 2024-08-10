import torch
import torch.nn.functional as F
from torch_geometric.data import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np

# Configurações
RANDOM_SEED = 42
NUM_EPOCHS = 200
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
HIDDEN_CHANNELS = 16
DROPOUT_RATE = 0.5

torch.manual_seed(RANDOM_SEED)

# Carregar o conjunto de dados Cora
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Dividir os dados em treinamento, validação e teste
data = train_test_split_edges(data, val_ratio=0.1, test_ratio=0.1)

# Definir o modelo GCN
class GCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, dropout_rate):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Inicializar o modelo, a função de perda e o otimizador
model = GCN(num_features=dataset.num_features, num_classes=dataset.num_classes,
            hidden_channels=HIDDEN_CHANNELS, dropout_rate=DROPOUT_RATE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
criterion = torch.nn.CrossEntropyLoss()

# Funções de treinamento e avaliação
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred[mask] == data.y[mask]
        return correct.sum().item() / mask.sum().item()

# Métricas adicionais
def calculate_metrics(mask):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).cpu().numpy()
        true = data.y[mask].cpu().numpy()
        return {
            'accuracy': accuracy_score(true, pred[mask.cpu().numpy()]),
            'precision': precision_score(true, pred[mask.cpu().numpy()], average='macro'),
            'recall': recall_score(true, pred[mask.cpu().numpy()], average='macro'),
            'f1': f1_score(true, pred[mask.cpu().numpy()], average='macro')
        }

# Listas para armazenar métricas
train_losses = []
val_accuracies = []

# Loop de treinamento
best_val_acc = 0
for epoch in range(NUM_EPOCHS):
    loss = train()
    train_losses.append(loss)
    val_acc = evaluate(data.val_mask)
    val_accuracies.append(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f}, Val Accuracy: {val_acc:.4f}')

print('Treinamento concluído.')

# Carregar o melhor modelo
model.load_state_dict(torch.load('best_model.pth'))

# Avaliar o modelo nos conjuntos de treinamento, validação e teste
train_metrics = calculate_metrics(data.train_mask)
val_metrics = calculate_metrics(data.val_mask)
test_metrics = calculate_metrics(data.test_mask)

print("\nMétricas finais:")
for split, metrics in zip(['Treinamento', 'Validação', 'Teste'], [train_metrics, val_metrics, test_metrics]):
    print(f"\n{split}:")
    for metric, value in metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")

# Plotar curvas de aprendizado
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Perda de Treinamento')
plt.xlabel('Época')
plt.ylabel('Perda')

plt.subplot(1, 2, 2)
plt.plot(val_accuracies)
plt.title('Acurácia de Validação')
plt.xlabel('Época')
plt.ylabel('Acurácia')

plt.tight_layout()
plt.savefig('learning_curves.png')
plt.close()

print("\nCurvas de aprendizado salvas em 'learning_curves.png'")