import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

data = np.empty((1,216), dtype='float32')
with os.scandir('dataset') as entries:
    for entry in entries:
        with open('dataset/' + entry.name, 'rb') as f:
            np_arr = np.load(f)
            np_arr = np.reshape(np_arr, (1,216))
            data = np.append(data, np_arr, axis=0)

data = data[1:]

data_tensor = torch.tensor(data, dtype=torch.float32)
train_data = data_tensor
train_label = [1, 0, 0, 0]
test_data = data_tensor
test_label = [1, 0, 0, 0]

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def init_model(input_size, h1, h2, output_size, learning_rate):
    model = nn.Sequential(
        nn.Linear(input_size, h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, output_size),
        nn.Softmax(dim=1)
    )

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)

    return model, optimizer

def train_model(dataset, model, optimizer, epochs=1):
    model = model.to(device=device)

    for e in range(epochs):
        for t, (x, y) in enumerate(dataset):
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            scores = model(x)
            loss = F.mse_loss(scores, y)

            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()

            if t % 5 == 0:
                print("Iteration %d, loss = %.4f" % (t, loss.item()))
                print()

def organize_data(inputs, labels):
    assert inputs.shape[0] == labels.shape[0]

    dataset = []
    for i in range(inputs.shape[0]):
        dataset.append((torch.reshape(torch.from_numpy(inputs[i]), (1, 216)), torch.reshape(torch.from_numpy(labels[i]), (1, 5))))

    return dataset
