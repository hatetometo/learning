import numpy as np
import ActivationFunction as AF

# year,day,time,num,sup_x,sup_y,sup_s,sup_t,em_x,em_y,em_s,em_t,result

# 3層ニューラルネットワーク
class ThreeLayerNetwork:
    # コンストラクタ
    def __init__(self, inodes, hnodes, onodes, lr):
        # 各レイヤーのノード数
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes

        # 学習率
        self.lr = lr

        # 重みの初期化
        self.w_ih = np.random.normal(0.0, 1.0, (self.hnodes, self.inodes))
        self.w_ho = np.random.normal(0.0, 1.0, (self.onodes, self.hnodes))

        # 活性化関数
        self.af = AF.sigmoid
        self.daf = AF.derivative_sigmoid

    # 誤差逆伝搬
    def backprop(self, idata, tdata):
        # 縦ベクトルに変換
        o_i = np.array(idata, ndmin=2).T
        t = np.array(tdata, ndmin=2).T

        # 隠れ層
        x_h = np.dot(self.w_ih, o_i)
        o_h = self.af(x_h)

        # 出力層
        x_o = np.dot(self.w_ho, o_h)
        o_o = self.af(x_o)

        # 誤差計算
        e_o = (t - o_o)
        e_h = np.dot(self.w_ho.T, e_o)

        # 重みの更新
        self.w_ho += self.lr * np.dot((e_o * self.daf(o_o)), o_h.T)
        self.w_ih += self.lr * np.dot((e_h * self.daf(o_h)), o_i.T)


    # 順伝搬
    def feedforward(self, idata):
        # 入力のリストを縦ベクトルに変換
        o_i = np.array(idata, ndmin=2).T

        # 隠れ層
        x_h = np.dot(self.w_ih, o_i)
        o_h = self.af(x_h)

        # 出力層
        x_o = np.dot(self.w_ho, o_h)
        o_o = self.af(x_o)

        return o_o

if __name__=='__main__':
    # パラメータ
    inodes = 12
    hnodes = 10
    onodes = 10
    lr = 0.3

    # ニューラルネットワークの初期化
    nn = ThreeLayerNetwork(inodes, hnodes, onodes, lr)

    # トレーニングデータのロード
    training_data_file = open('NN_testdata_2.csv', 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # テストデータのロード
    test_data_file = open('testdata_2_in2019.csv')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # 学習
    epoch = 10
    for e in range(epoch):
        print('#epoch ', e)
        data_size = len(training_data_list)
        for i in range(data_size):
            if i % 1000 == 0:
                print('  train: {0:>5d} / {1:>5d}'.format(i, data_size))
            val = training_data_list[i].split(',')
            idata = (np.asfarray(val[1:]) / 255.0 * 0.99) + 0.01
            tdata = np.zeros(onodes) + 0.01
            tdata[int(val[0])] = 0.99
            nn.backprop(idata, tdata)
            pass
        pass

    # テスト
    scoreboard = []
    for record in test_data_list:
        val = record.split(',')
        idata = (np.asfarray(val[1:]) / 255.0 * 0.99) + 0.01
        tlabel = int(val[0])
        predict = nn.feedforward(idata)
        plabel = np.argmax(predict)
        scoreboard.append(tlabel == plabel)
        pass

    scoreboard_array = np.asarray(scoreboard)
    print('performance: ', scoreboard_array.sum() / scoreboard_array.size)
