import numpy as np
from sklearn import svm

#csvファイルの読み込み
npArray = np.loadtxt("testdata_1.csv", delimiter = ",", dtype = "float")

# 説明変数の格納
x = npArray[:, 0:9]

#目的変数の格納
y = npArray[:, 9:10].ravel()

#学習手法にSVMを選択
model = svm.SVC()

#学習
model.fit(x,y)

#評価データ(ここは自分で好きな値を入力)
result = [[16710 ,1937,702,376,1729,1863,1247,379,243]]

#predict関数で、評価データの試合結果を予測
ans = model.predict(result)

if ans == 0:
    print("勝ちです")
if ans == 1:
    print("負けです")
if ans == 2:
    print('引き分けです')
