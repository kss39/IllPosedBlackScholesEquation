import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset

debug = False

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(13, 50, True)
        self.hidden2 = nn.Linear(50, 25, True)
        self.hidden3 = nn.Linear(25, 15, True)
        self.output = nn.Linear(15, 1, True)
        self.to(get_device())

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.sigmoid(self.output(x))
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

def init_weights(m):
    print(m)

if __name__ == '__main__':
    if not len(sys.argv) == 3:
        print('Usage: python predict.py [folder] [size_limit]')
        sys.exit(-1)
    file = sys.argv[1]
    size = int(sys.argv[2])
    input_df = pd.read_csv(file, nrows=size).sample(frac=1).values
    size_train = int(min(size, len(input_df)) / 10 * 9)
    device = get_device()
    for i in range(len(input_df)):
        #label
        input_df[i][15] = 1 if input_df[i][15] >= input_df[i][4] else 0
        #normalization
        strike = input_df[i][1]
        u_ask = input_df[i][8] - strike
        u_bid = input_df[i][9] - strike
        total = 0.0
        for j in range(2, 8):
            total += input_df[i][j]
        av = total/6.0
        sigma = 0.0
        for j in range(2, 8):
            sigma += (input_df[i][j] - av) ** 2
        sigma = math.sqrt(sigma/5.0)
        for j in range(2, 8):
            input_df[i][j] = (input_df[i][j] - av)/sigma
        input_df[i][8] = (u_ask - av)/sigma
        input_df[i][9] = (u_bid - av)/sigma
        input_df[i][13] = (input_df[i][13] - av)/sigma
        input_df[i][14] = (input_df[i][14] - av)/sigma
        #print(input_df[i])

    X = torch.from_numpy(input_df[0:size_train, 2:-2]).float().to(device)
    #print(X)
    y = torch.from_numpy(input_df[0:size_train, -2:-1]).float().to(device)
    y.resize_((size_train))
    #print(y)
    #print(y.shape)
    #X_test = torch.from_numpy(input_df[size_train:, 2:-2]).float().to(device)
    #y_test = torch.from_numpy(input_df[size_train:, -2:-1]).float().to(device)
    nnn = Net()
    #nnn.apply(torch.nn.init.xavier_uniform_)
    #nnn.apply(init_weights)
    #print(nnn)
    optimizer = torch.optim.Adam(nnn.parameters(), lr=0.0001)
    loss_func = nn.BCELoss()

    for t in range(1000):
        y_pred = nnn(X)
        #print(y_pred)
        loss = loss_func(y_pred, y)

        if t % 100 == 0:
            print(f'Epoch {int(t)}: Loss {loss}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Final loss: {loss}')
    #print(y_pred)
    exit(0)
    result = nnn(X_test)
    test_loss = loss_func(result, y_test)

    print(f'Test loss is {test_loss};')
    print('For the final five test options,')
    print('Estimates and real prices are:')
    print('Estimation\t\t\t\tReal\t\t\t\tMinimizer')
    for i in range(1, 6):
        print(f'{result[-i].cpu().detach().numpy()}\t\t\t{y_test[-i].cpu().detach().numpy()}'
              f'\t\t\t{X_test[-i].cpu().detach().numpy()[-2:]}')
