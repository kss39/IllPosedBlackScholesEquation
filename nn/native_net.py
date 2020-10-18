import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset

debug = False

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(14, 50)
        self.hidden2 = nn.Linear(50, 25)
        self.hidden3 = nn.Linear(25, 14)
        self.output = nn.Linear(14, 2)
        self.to(get_device())

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.output(x)
        return x


class OptionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.data.iloc[idx]
        if self.transform:
            row = self.transform(row)
        return row


def get_device():
    if torch.cuda.is_available() and not debug:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU
    return device


if __name__ == '__main__':
    if not len(sys.argv) == 3:
        print('Usage: python predict.py [folder] [size_limit]')
        sys.exit(-1)
    file = sys.argv[1]
    size = int(sys.argv[2])
    input_df = pd.read_csv(file, nrows=size).sample(frac=1).values
    size_train = int(min(size, len(input_df)) / 10 * 9)
    device = get_device()
    X = torch.from_numpy(input_df[0:size_train, 1:-2]).float().to(device)
    y = torch.from_numpy(input_df[0:size_train, -2:]).float().to(device)
    X_test = torch.from_numpy(input_df[size_train:, 1:-2]).float().to(device)
    y_test = torch.from_numpy(input_df[size_train:, -2:]).float().to(device)
    nnn = Net()
    optimizer = torch.optim.SGD(nnn.parameters(), lr=0.0001)
    loss_func = nn.MSELoss()

    for t in range(200000):
        y_pred = nnn(X)
        loss = loss_func(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 1000 == 0:
            print(f'Epoch {int(t/1000)}: Loss {loss}')

    result = nnn(X_test)
    test_loss = loss_func(result, y_test)

    print(f'Test loss is {test_loss};')
    print('For the final five test options,')
    print('Estimates and real prices are:')
    print('Estimation\t\t\t\tReal\t\t\t\tMinimizer')
    for i in range(1, 6):
        print(f'{result[-i].cpu().detach().numpy()}\t\t\t{y_test[-i].cpu().detach().numpy()}'
              f'\t\t\t{X_test[-i].cpu().detach().numpy()[-2:]}')
