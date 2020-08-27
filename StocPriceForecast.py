import datetime
import numpy as np
import matplotlib.pylab as plt
from chainer import Chain, Variable, cuda, optimizer, optimizers, serializers
import chainer.functions as F
import chainer.links as L
import pandas as pd


# モデルクラス定義
 
class LSTM(Chain):
    def __init__(self, in_size, hidden_size, out_size):
        # クラスの初期化
        # :param in_size: 入力層のサイズ
        # :param hidden_size: 隠れ層のサイズ
        # :param out_size: 出力層のサイズ
        super(LSTM, self).__init__(
            xh = L.Linear(in_size, hidden_size),
            hh = L.LSTM(hidden_size, hidden_size),
            hy = L.Linear(hidden_size, out_size)
        )
 
    def __call__(self, x, t=None, train=False):
        # 順伝播の計算を行う関数
        # :param x: 入力値
        # :param t: 正解の予測値
        # :param train: 学習かどうか
        # :return: 計算した損失 or 予測値
        x = Variable(x)
        if train:
            t = Variable(t)
        h = self.xh(x)
        h = self.hh(h)
        y = self.hy(h)
        if train:
            return F.mean_squared_error(y, t)
        else:
            return y.data
 
    def reset(self):
        # 勾配の初期化とメモリの初期化
        self.zerograds()
        self.hh.reset_state()



EPOCH_NUM = 1000
IN_SIZE=1
HIDDEN_SIZE = 60
OUT_SIZE=1
BATCH_ROW_SIZE = 10 # 分割した時系列をいくつミニバッチに取り込むか
BATCH_COL_SIZE = 60 # ミニバッチで分割する時系列数


# 学習
def Training(str="",weight_load=False):
    print("\nTraining\n")
    
    # 教師データ
    df = pd.read_csv('nikkei-225-index-historical-chart-data.csv',header=8)
    #mat = df.query('date.str.match("^2019-")', engine='python')
    mat = df.query('date.str.match('+str+')', engine='python')
    train_data_t = mat[' value'].values
    print(train_data_t)
    
    train_data = np.arange(len(train_data_t), dtype="float32");
    
    for i in range(len(train_data_t)-1):
        train_data[i] = train_data_t[i+1]-train_data_t[i]
    
    gain = np.max(train_data)-np.min(train_data)
    gain = gain/2
    
    train_data = train_data/gain
    
    print(train_data)
    # 入力データと教師データを作成
    train_x, train_t = [], []
    for i in range(len(train_data)-1):
        train_x.append(train_data[i])
        train_t.append(train_data[i+1])
    train_x = np.array(train_x, dtype="float32")
    train_t = np.array(train_t, dtype="float32")
    Num = len(train_x)
     
    # モデル定義
    model = LSTM(in_size=IN_SIZE, hidden_size=HIDDEN_SIZE, out_size=OUT_SIZE)
    optimizer = optimizers.Adam()
    optimizer.setup(model)
     
    if weight_load:
        serializers.load_npz("mymodel.npz", model)
    
    # 学習開始
    print("Train")
    st = datetime.datetime.now()
    for epoch in range(EPOCH_NUM):
     
        # ミニバッチ学習
        x, t = [], []
        #  ミニバッチ学習データとして、時系列全体から、BATCH_COL_SIZE分の時系列を抜き出したものを、BATCH_ROW_SIZE個用意する
        for i in range(BATCH_ROW_SIZE):
            # ランダムな箇所、ただしBATCH_COL_SIZE分だけ抜き取れる場所から選ぶ
            # (indexの末端がBATCH_COL_SIZEを超えない部分でリミットを掛ける)
            index = np.random.randint(0, Num-BATCH_COL_SIZE+1) 
            x.append(train_x[index:index+BATCH_COL_SIZE]) # BATCH_COL_SIZE分の時系列を取り出す
            t.append(train_t[index:index+BATCH_COL_SIZE])
        x = np.array(x, dtype="float32")
        t = np.array(t, dtype="float32")
        loss = 0
        total_loss = 0
        model.reset() # 勾配とメモリの初期化
        for i in range(BATCH_COL_SIZE): # 各時刻おきにBATCH_ROW_SIZEごと読み込んで損失を計算する
            x_ = np.array([x[j, i] for j in range(BATCH_ROW_SIZE)], dtype="float32")[:, np.newaxis] # 時刻iの入力値
            t_ = np.array([t[j, i] for j in range(BATCH_ROW_SIZE)], dtype="float32")[:, np.newaxis] # 時刻i+1の値（＝正解の予測値）
            loss += model(x=x_, t=t_, train=True) # 誤差合計
        loss.backward() # 誤差逆伝播
        loss.unchain_backward() 
        total_loss += loss.data
        optimizer.update()
        if (epoch+1) % 100 == 0:
            ed = datetime.datetime.now()
            print("epoch:\t{}\ttotal loss:\t{}\ttime:\t{}".format(epoch+1, total_loss, ed-st))
            st = datetime.datetime.now()
     
    serializers.save_npz("mymodel.npz", model) # npz形式で書き出し
     
# 予測能力評価
def Predict(str):
    print("\nPredict")
    # モデルの定義
    model = LSTM(in_size=IN_SIZE, hidden_size=HIDDEN_SIZE, out_size=OUT_SIZE)
    
    # 重みロード
    serializers.load_npz("mymodel.npz", model)
    
    # 予測元データロード
    df = pd.read_csv('nikkei-225-index-historical-chart-data.csv',header=8)
    #mat = df.query('date.str.match("^2018-((08)|(09)|(10)|(11))")', engine='python')
    #mat = df.query('date.str.match("^2019-((03)|(04)|(05)|(06))")', engine='python')
    #mat = df.query('date.str.match("^2020-((01)|(02)|(03)|(04))")', engine='python')
    mat = df.query('date.str.match('+str+')', engine='python')
    train_data_t = mat[' value'].values
    
    train_data = np.arange(len(train_data_t), dtype="float32");
    
    print(train_data_t)
    
    # 偏差算出(微分)
    for i in range(len(train_data_t)-1):
        train_data[i] = train_data_t[i+1]-train_data_t[i]
    
    # 正規化用ゲイン
    gain = np.max(train_data)-np.min(train_data)
    gain = gain/2
    
    train_data = train_data/gain	# ±1.0以内に
    Num = len(train_data)-1
    
    print(train_data)
    
    predict = np.empty(0) # 予測値格納用
    predict_size = 30     # 予測サイズ
    predata_size = len(train_data)-predict_size # 予測直前までのデータ数
    indata = train_data[1:predata_size] # 予測直前までのデータ
    for _ in range(predata_size):
        model.reset()
        for i in indata: # モデルに予測直前までの時系列を読み込ませる
            x = np.array([[i]], dtype="float32")
            y = model(x=x, train=False)
        predict = np.append(predict, y) # 最後の予測値を記録
        # モデルに読み込ませる予測直前時系列を予測値で更新する
        indata = np.delete(indata, 0)
        indata = np.append(indata, y)
    
    plt.plot(range(Num+1), train_data, color="red", label="t")
    plt.plot(range(predata_size, predata_size+predict_size-1), predict[0:predict_size-1], "--.", label="y")
    plt.legend(loc="upper left")
    plt.show()

    predict = predict * gain	# 元データと同じ割合で予測値を拡大
    ipredict = np.arange(len(predict)+1, dtype="float32")
    predict_tmp = train_data_t[predata_size]; # 初期値(積分定数)
    ipredict[0]=predict_tmp
    # 積分
    for i in range(len(predict)):
        predict_tmp = predict_tmp + predict[i]
        ipredict[i+1] = predict_tmp
    
    plt.plot(range(Num+1), train_data_t, color="red", label="t")
    plt.plot(range(predata_size, predata_size+predict_size-1), ipredict[0:predict_size-1], "--.", label="y")
    plt.show()

# 本当に将来を予測
def Predict2(str="",tail=90):
    print("\nPredict2")
    # モデルの定義
    model = LSTM(in_size=IN_SIZE, hidden_size=HIDDEN_SIZE, out_size=OUT_SIZE)
    
    # 重みロード
    serializers.load_npz("mymodel.npz", model)
    
    # 予測元データロード
    df = pd.read_csv('nikkei-225-index-historical-chart-data.csv',header=8)
    
    if not str:
        mat = df.tail(tail)
    else:
        mat = df.query('date.str.match('+str+')', engine='python')
    
    train_data_t = mat[' value'].values
    
    train_data = np.arange(len(train_data_t), dtype="float32");
    
    
    for i in range(len(train_data_t)-1):
        train_data[i] = train_data_t[i+1]-train_data_t[i]
    
    gain = np.max(train_data)-np.min(train_data)
    gain = gain/2
    
    train_data = train_data/gain
    Num = len(train_data)-1
    
    print(train_data)
    
    predict = np.empty(0) # 予測値格納用
    predict_size = 30     # 予測サイズ
    indata = train_data # 予測直前までの時系列
    for _ in range(predict_size):
        model.reset() # メモリを初期化
        for i in indata: # モデルに予測直前までのデータを読み込ませる
            x = np.array([[i]], dtype="float32")
            y = model(x=x, train=False)
        predict = np.append(predict, y) # 最後の予測値を記録
        # モデルに読み込ませる予測直前データを予測値で更新する
        indata = np.delete(indata, 0)
        indata = np.append(indata, y)
    
    plt.plot(range(Num+1), train_data, color="red", label="t")
    plt.plot(range(Num, Num+predict_size), predict[0:predict_size], "--.", label="y")
    plt.legend(loc="upper left")
    plt.show()

    predict = predict * gain
    ipredict = np.arange(len(predict)+1, dtype="float32")
    predict_tmp = train_data_t[Num];
    ipredict[0]=predict_tmp
    for i in range(len(predict)):
        predict_tmp = predict_tmp + predict[i]
        ipredict[i+1] = predict_tmp
    
    plt.plot(range(Num+1), train_data_t, color="red", label="t")
    plt.plot(range(Num, Num+predict_size), ipredict[0:predict_size], "--.", label="y")
    plt.show()

#Training(str="\"^(2019-)|(2020-)\"",weight_load=False)
#Training(str="\"^(2019-)|(2020-)\"",weight_load=True)
Predict(str="\"^2019-((01)|(02)|(03)|(04))\"")
Predict(str="\"^2019-((04)|(05)|(06)|(07))\"")
Predict(str="\"^2019-((07)|(08)|(09)|(10))\"")
Predict(str="\"^2019-((09)|(10)|(11)|(12))\"")
Predict(str="\"^2020-((01)|(02)|(03)|(04))\"")

#Predict2(str="\"^2020-((01)|(02)|(03))\"")
Predict2()
