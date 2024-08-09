import torch
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv

# Defina as dimensões do grafo e do valor futuro a ser previsto
num_nodes = 5
num_features = 2
output_dim = 1

# Crie um Grafo com num_nodes nodos, cada um com num_features características
graph = Data(x=torch.randn(num_nodes, num_features))

# Crie um GNN para processar o grafo e prever o valor futuro
class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GraphConv(64, 128)
        self.conv2 = GraphConv(128, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        return x

# Crie um modelo de GNN e treine-o
model = GNN()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(graph)
    loss = criterion(output, graph.x[:,-1]) # Use only the last value as target
    loss.backward()
    optimizer.step()

print('Previsão:', model(graph).detach().numpy())