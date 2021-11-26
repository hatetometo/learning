from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # x(入力)のユニット数は9
        self.fc1 = nn.Linear(12, 10)
        # 隠れ層1のユニット数は10
        self.fc2 = nn.Linear(10, 10)
        # 隠れ層2のユニット数は10
        self.fc3 = nn.Linear(10, 9)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x),dim=1)
        return x

train = pd.read_csv('NN_testdata_2.csv', sep=',', encoding='utf-8')
test = pd.read_csv('testdata_2_in2019.csv', sep=',', encoding='utf-8')

X_train = train[['year','day','time','num','sup_x','sup_y','sup_s','sup_t','em_x','em_y','em_s','em_t']].astype(float)
Y_train = train['result'].values
X_test = test[['year','day','time','num','sup_x','sup_y','sup_s','sup_t','em_x','em_y','em_s','em_t']].values
Y_test = test['result'].values
X_train = X_train.values


x = torch.tensor(X_train, dtype=torch.float32)
y = torch.LongTensor(Y_train)

train_dataset = torch.utils.data.TensorDataset(x, y)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64)

# 学習モデルのインスタンスを作成
model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01)
# 損失関数の定義
criterion = nn.CrossEntropyLoss()

# for i, j in train_dataloader:
#     optimizer.zero_grad()
#     outputs = model(i)
#     print(outputs)

epoch = 50+1
for i in range(1, epoch):
    sum_loss = 0.0
    for x, y in train_dataloader:
        # 勾配の初期化
        optimizer.zero_grad()
        # 説明変数xをネットワークにかける
        output = model(x)
        # 損失関数の計算
        loss = criterion(output, y)
        # 勾配の計算
        loss.backward()
        # パラメタの更新
        optimizer.step()

        sum_loss += loss.item()
    print(f"epoch : {i}loss : {sum_loss/i}")

outputs = model(torch.tensor(X_test, dtype = torch.float))
_, predicted = torch.max(outputs.data, 1)
print(outputs)
print(outputs.shape)
# print(predicted)
y_predicted = predicted.numpy()
# print(y_predicted)
accuracy = 100 * np.sum(predicted.numpy() == Y_test) / len(y_predicted)
print('accuracy: {:.1f}%'.format(accuracy))
