# Ponderada - Semana 07
## Previsão de Preços de Abacate com RNN

Este projeto utiliza uma Rede Neural Recorrente (RNN), implementada com PyTorch, para prever os preços médios de abacates com base em dados históricos.

### Objetivo

Prever o preço futuro dos abacates a partir de uma sequência de preços passados, utilizando séries temporais e uma arquitetura de rede neural recorrente simples.

### Modelo

- Tipo de modelo: RNN (com `nn.RNN`)
- Biblioteca: PyTorch
- Função de perda: MSELoss
- Otimizador: Adam
- Normalização: `MinMaxScaler` (0 a 1)

### Estrutura dos Dados

O dataset utilizado é o `avocado.csv`, que contém informações de vendas de abacates nos EUA.

A principal variável usada é:

- `AveragePrice`: preço médio do abacate

### Pré-processamento

1. Carregamento do dataset com pandas
2. Normalização da coluna `AveragePrice` entre 0 e 1
3. Criação de janelas (sequências) para entrada na RNN
4. Conversão para tensores do PyTorch

### Arquitetura

```python
class AvocadoPriceRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out
```
### Treinamento

O treinamento da RNN foi realizado para prever o preço médio do abacate com base em séries temporais.

#### Configurações Principais

- **Épocas**: 100 (ou ajustável de acordo com a performance)
- **Critério de parada**: otimização da função de perda (MSELoss)
- **Divisão dos dados**: 80% para treino, 20% para teste
- **Batch size**: não utilizado explicitamente (treinamento em lote completo)
- **Otimização**: Adam com taxa de aprendizado padrão (`lr=0.001`)

#### Loop de Treinamento

```python
model = AvocadoPriceRNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    output = model(X_train)
    loss = criterion(output, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Época {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
```

### 📉 Avaliação

Após o treinamento, o modelo é avaliado com os dados de teste. A performance é comparada visualmente com os valores reais por meio de gráficos.

```python
model.eval()
with torch.no_grad():
    predicted = model(X_test)
    predicted = scaler.inverse_transform(predicted.numpy())
    actual = scaler.inverse_transform(y_test.numpy())
```

### Avaliação

Após o treinamento, o modelo é avaliado com os dados de teste. A performance é comparada visualmente com os valores reais por meio de gráficos.

```python
model.eval()
with torch.no_grad():
    predicted = model(X_test)
    predicted = scaler.inverse_transform(predicted.numpy())
    actual = scaler.inverse_transform(y_test.numpy())
```