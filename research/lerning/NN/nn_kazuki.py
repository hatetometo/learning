import torch
from torch._C import device
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def dataset_create(df):
    #Y = torch.FloatTensor(df['result'].values)
    #X = torch.LongTensor(df.drop('result', axis=1).values)
    data = torch.tensor(df.drop('result', axis = 1).values, dtype=torch.float)
    label = torch.tensor(df['result'].values)
    #X = [[float(x) for x in df['score_win']], [float(x) for x in df['score_lose']], [float(x) for x in df['num']]]
    #Y = [[int(x) for x in df['result']],]
    #X = torch.tensor(train_data, dtype=torch.float32, requires_grad=True)
    #Y = torch.tensor(train_label, dtype=torch.int64)
    return data, label

def mean_norm(df_input):
    return df_input.apply(lambda x: (x-x.mean())/ x.std(), axis=0)

df = pd.read_csv('data.csv', sep=',')
train, test = train_test_split(df, test_size=0.2, shuffle=True)
train_X, train_Y = dataset_create(train)
test_X, test_Y = dataset_create(test)
#train_Y = torch.transpose(train_Y, 0, 1)
#test_Y = torch.transpose(test_Y, 0, 1)
print(test_X.size())
print(test_Y.size())
print(train_X.size())
print(train_Y.size())
train_Dataset = torch.utils.data.TensorDataset(train_X, train_Y)
train_DataLoader = torch.utils.data.DataLoader(dataset=train_Dataset, batch_size = 32, shuffle = True)
test_Dataset = torch.utils.data.TensorDataset(test_X, test_Y)
test_DataLoader = torch.utils.data.DataLoader(dataset=test_Dataset, batch_size = 32, shuffle = False)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 60)
        self.fc2 = nn.Linear(60, 80)
        self.fc3 = nn.Linear(80, 2)
         
    def forward(self, x):
        x = F.relu(self.fc1(x)) # ReLU: max(x, 0)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

net = Net()

from matplotlib import pyplot as plt

# 損失の平均値と正解率をエポックごとに保持する配列
def run_plus(DataLoader, device):
    device = torch.device('cuda:0' if torch.cuda.is_available() else'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
    losses = []
    accs = []
    for epoch in range(500+1):
        loss_ = 0
        acc_ = 0
        for xx, yy in DataLoader:
            x = xx.to(device)
            y = yy.to(device)
            y_pred = net(x).to(device)

            loss = criterion(y_pred, y) # 推論値と正解データの誤差を計算
            loss_ += loss.item()

            acc_ += (y_pred.max(1)[1] == y).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print('epoch:{0}\tloss:{1}\taccuracy:{2}'.format(epoch, loss_ / len(DataLoader), acc_ / len(DataLoader)))
        losses.append(loss_ / len(DataLoader))
        accs.append(acc_ / len(DataLoader))

    return losses, accs

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# net = net.to(device)

# サブプロットを定義
fig = plt.figure()
g = fig.subplots(2)
epochs = torch.tensor(range(500+1)).tolist()

# 学習の実行
train_loss, train_acc = run_plus(train_DataLoader, device)
test_loss, test_acc = run_plus(test_DataLoader, device)

# 損失をプロット
g[0].plot(epochs, train_loss, label='train')
g[0].plot(epochs, test_loss, label='test')
g[0].set_xlabel('epoch')
g[0].set_ylabel('Loss')
g[0].set_title('Loss')
hans, labs = g[0].get_legend_handles_labels()
g[0].legend(handles=hans, labels=labs)

# 正解率をプロット/
g[1].plot(epochs, train_acc, label='train')
g[1].plot(epochs, test_acc, label='test')
g[1].set_xlabel('epoch')
g[1].set_ylabel('Accuracy')
g[1].set_title('Accuracy')
hans, labs = g[1].get_legend_handles_labels()
g[1].legend(handles=hans, labels=labs)

fig.tight_layout()
plt.show()