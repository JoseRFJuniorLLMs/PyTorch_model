import torch
import torch.nn as nn

# Defina uma classe NeuralNet que implementa uma rede neural simples
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Crie um módulo de entrada (Linear) com 2 entradas e 3 saídas
        self.fc1 = nn.Linear(3, 10, 1)
        # bias 1 = tensor([0.3147, 0.0000, 0.6478], grad_fn=<ReluBackward0>)
    # Defina uma função para realizar a saída da rede neural
    def forward(self, x):
        # Aplicar a função Linear ao input 'x' e retornar o resultado
        return torch.relu(self.fc1(x))

# Crie um objeto da classe NeuralNet
net = Net()

# Crie um tensor de exemplo com 2 entradas (é o que estamos esperando para entrada do módulo linear)
input_tensor = torch.tensor([0.1, 0.2, 0.3])

# Realize a saída usando o objeto 'net'
output = net(input_tensor)

print(output)