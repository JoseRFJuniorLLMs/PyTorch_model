import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np

# Função para calcular a constante do caos usando o mapa logístico
def logistic_map(r, x, num_iterations=1):
    """
    Calcula o próximo valor no mapa logístico.
    
    Parâmetros:
    ----------
    r : float
        Parâmetro de controle do mapa logístico.
    x : float
        Valor atual.
    num_iterations : int
        Número de iterações a serem realizadas.
        
    Retorna
    -------
    float
        Valor resultante após o número de iterações.
    """
    for _ in range(num_iterations):
        x = r * x * (1 - x)
    return x

class DropConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False, dropout_prob=0.0):
        super(DropConvBlock, self).__init__()
        self.use_dropout = (dropout_prob != 0.0)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activ = nn.ReLU(inplace=True)
        if self.use_dropout:
            self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x

class FractalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super(FractalBlock, self).__init__()
        self.num_columns = num_columns
        self.loc_drop_prob = loc_drop_prob

        self.blocks = nn.Sequential()
        depth = 2 ** (num_columns - 1)
        for i in range(depth):
            level_block_i = nn.Sequential()
            for j in range(self.num_columns):
                column_step_j = 2 ** j
                if (i + 1) % column_step_j == 0:
                    in_channels_ij = in_channels if (i + 1 == column_step_j) else out_channels
                    level_block_i.add_module("subblock{}".format(j + 1), DropConvBlock(
                        in_channels=in_channels_ij,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        dropout_prob=dropout_prob))
            self.blocks.add_module("block{}".format(i + 1), level_block_i)

    @staticmethod
    def join_outs(raw_outs, glob_num_columns, num_columns, loc_drop_prob, training):
        curr_num_columns = len(raw_outs)
        out = torch.stack(raw_outs, dim=0)
        assert (out.size(0) == curr_num_columns)

        if training:
            batch_size = out.size(1)
            batch_mask = np.random.binomial(
                n=1,
                p=(1.0 - loc_drop_prob),
                size=(curr_num_columns, batch_size)).astype(np.float32)
            batch_mask = torch.from_numpy(batch_mask).to(out.device)
            assert (batch_mask.size(0) == curr_num_columns)
            assert (batch_mask.size(1) == batch_size)
            batch_mask = batch_mask.unsqueeze(2).unsqueeze(3)
            masked_out = out * batch_mask
            num_alive = batch_mask.sum(dim=0)
            num_alive[num_alive == 0.0] = 1.0
            out = masked_out.sum(dim=0) / num_alive
        else:
            out = out.mean(dim=0)

        return out

    def forward(self, x, glob_num_columns):
        outs = [x] * self.num_columns

        for level_block_i in self.blocks._modules.values():
            outs_i = []

            for j, block_ij in enumerate(level_block_i._modules.values()):
                input_i = outs[j]
                outs_i.append(block_ij(input_i))

            joined_out = FractalBlock.join_outs(
                raw_outs=outs_i[::-1],
                glob_num_columns=glob_num_columns,
                num_columns=self.num_columns,
                loc_drop_prob=self.loc_drop_prob,
                training=self.training)

            len_level_block_i = len(level_block_i._modules.values())
            for j in range(len_level_block_i):
                outs[j] = joined_out

        return outs[0]

class FractalUnit(nn.Module):
    def __init__(self, in_channels, out_channels, num_columns, loc_drop_prob, dropout_prob):
        super(FractalUnit, self).__init__()
        self.block = FractalBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            num_columns=num_columns,
            loc_drop_prob=loc_drop_prob,
            dropout_prob=dropout_prob)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, glob_num_columns):
        x = self.block(x, glob_num_columns=glob_num_columns)
        x = self.pool(x)
        return x

class FractalBrainNet(nn.Module):
    def __init__(self, channels, num_columns, dropout_probs, loc_drop_prob, glob_drop_ratio, in_channels=3, in_size=(32, 32), num_classes=10):
        super(FractalBrainNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        self.glob_drop_ratio = glob_drop_ratio
        self.num_columns = num_columns

        self.features = nn.Sequential()
        for i, out_channels in enumerate(channels):
            dropout_prob = dropout_probs[i]
            self.features.add_module("unit{}".format(i + 1), FractalUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                num_columns=num_columns,
                loc_drop_prob=loc_drop_prob,
                dropout_prob=dropout_prob))
            in_channels = out_channels

        self.output = nn.Linear(in_features=in_channels, out_features=num_classes)
        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        glob_batch_size = int(x.size(0) * self.glob_drop_ratio)
        glob_num_columns = np.random.randint(0, self.num_columns, size=(glob_batch_size,))

        x = self.features(x, glob_num_columns=glob_num_columns)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x

def apply_chaotic_dropout_prob(model, r=3.7, num_iterations=10):
    """
    Ajusta a taxa de dropout dos blocos fractais usando uma constante caótica.
    
    Parâmetros:
    ----------
    model : nn.Module
        O modelo neural a ser ajustado.
    r : float
        Valor da constante de caos para o mapa logístico.
    num_iterations : int
        Número de iterações para o cálculo do mapa logístico.
    """
    for name, module in model.named_modules():
        if isinstance(module, DropConvBlock):
            x = torch.tensor(0.5)  # valor inicial para o mapa logístico
            dropout_prob = logistic_map(r, x, num_iterations)
            module.dropout_prob = dropout_prob
            if module.dropout is not None:
                module.dropout.p = dropout_prob

# Exemplo de uso
def main():
    # Configuração do modelo
    channels = [64, 128, 256]
    num_columns = 3
    dropout_probs = [0.1, 0.2, 0.3]
    loc_drop_prob = 0.15
    glob_drop_ratio = 0.5
    num_classes = 10

    model = FractalBrainNet(
        channels=channels,
        num_columns=num_columns,
        dropout_probs=dropout_probs,
        loc_drop_prob=loc_drop_prob,
        glob_drop_ratio=glob_drop_ratio,
        num_classes=num_classes)

    # Aplicando o ajuste de taxa de dropout usando a constante de caos
    apply_chaotic_dropout_prob(model, r=3.7, num_iterations=10)

    # Teste com dados fictícios
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(f'Output shape: {y.shape}')

if __name__ == "__main__":
    main()
