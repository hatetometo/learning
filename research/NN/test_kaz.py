from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

# パラメータ情報
BATCH_SIZE = 64
MOMENTUM = 0.9
LEARNING_RATE = 0.03
WEIGHT_DECAY = 0.005
EPOCH = 50

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(12, 16)
        self.fc2 = nn.Linear(16, 10)
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


train_input = torch.tensor(X_train, dtype=torch.float32)
train_label = torch.LongTensor(Y_train)
test_input = torch.tensor(X_test, dtype=torch.float32)
test_label = torch.LongTensor(Y_test)

train_dataset = torch.utils.data.TensorDataset(train_input, train_label)
test_dataset = torch.utils.data.TensorDataset(test_input, test_label)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 学習モデルのインスタンスを作成
model = Net()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,  momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
# 損失関数の定義
criterion = nn.CrossEntropyLoss()

train_loss_value=[]      #trainingのlossを保持するlist
train_acc_value=[]       #trainingのaccuracyを保持するlist
test_loss_value=[]       #testのlossを保持するlist
test_acc_value=[]        #testのaccuracyを保持するlist

for epoch in range(EPOCH):
    print(epoch+1)
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

    sum_loss = 0.0          #lossの合計
    sum_correct = 0         #正解率の合計
    sum_total = 0           #dataの数の合計

    #train dataを使ってテストをする(パラメータ更新がないようになっている)
    for (x, y) in train_dataloader:
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        sum_loss += loss.item()                            #lossを足していく
        _, predicted = outputs.max(1)                      #出力の最大値の添字(予想位置)を取得
        sum_total += y.size(0)                        #labelの数を足していくことでデータの総和を取る
        sum_correct += (predicted == y).sum().item()  #予想位置と実際の正解を比べ,正解している数だけ足す
    print("train mean loss={}, accuracy={}"
            .format(sum_loss*BATCH_SIZE/len(train_dataloader.dataset), float(sum_correct/sum_total)))  #lossとaccuracy出力
    train_loss_value.append(sum_loss*BATCH_SIZE/len(train_dataloader.dataset))  #traindataのlossをグラフ描画のためにlistに保持
    train_acc_value.append(float(sum_correct/sum_total))   #traindataのaccuracyをグラフ描画のためにlistに保持

    sum_loss = 0.0
    sum_correct = 0
    sum_total = 0

    for (inputs, labels) in test_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        sum_loss += loss.item()
        _, predicted = outputs.max(1)
        sum_total += labels.size(0)
        sum_correct += (predicted == labels).sum().item()
    print("test  mean loss={}, accuracy={}"
            .format(sum_loss*BATCH_SIZE/len(test_dataloader.dataset), float(sum_correct/sum_total)))
    test_loss_value.append(sum_loss*BATCH_SIZE/len(test_dataloader.dataset))
    test_acc_value.append(float(sum_correct/sum_total))
