import torch
import torchvision
import torchvision.transforms as transforms

# Transformação para normalizar os dados
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Baixando o conjunto de dados MNIST
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

# Exemplo de loop de treinamento
for images, labels in trainloader:
    print(images.size(), labels)
    break
