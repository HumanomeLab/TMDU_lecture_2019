import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
import torch.utils.data as data
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import os
import copy
import argparse
import time
import re
import numpy as np
import pandas as pd


"""
デバイス名を取る。GPUが利用可能なら、GPUを利用する。
"""
def device_name():
    device = ""
    if torch.cuda.is_available(): # GPUが利用可能か確認
        device = 'cuda'
    else:
        device = 'cpu'
    return device


"""
exp_data.csvに対して、以下のことを実施する make_dataset 関数を作成する。

データの読み込み
クラスを以下のベクトル形式で指定する　（one-hot vector と呼ばれる）
クラスが0 (y==0) なら、[1, 0]
クラスが1 (y==1) なら、[0, 1]
"""

def make_dataset(filename):
    labels = []
    # データの読み込みと特徴量の選択
    dataset = pd.read_csv(filename, index_col="Name")
    df = dataset.drop("class", axis=1)
    # df の値を取り出し、型を変換する
    X = df.values.astype(np.float32)
    y = np.array(dataset["class"].values)
    # クラスの値を one-hot-vector に変換する
    for y_i in y:
        c = [0, 0]  # one-hot-vector
        if y_i == 0.0:
            c = [1, 0]
        else:
            c = [0, 1]
        labels.append(np.array(c))
    return X, labels

"""
各サンプル点の変数をsampleに、クラスをtargetsに保持する。
"""
class DatasetFolder(data.Dataset):
    def __init__(self, X, y):
        self.samples = X
        self.targets = y

    def __getitem__(self, index):
        # index番目の変数とクラスを返す
        sample = self.samples[index]
        target = self.targets[index]
        sample = torch.from_numpy(sample)  # numpyからPyTorchの型へ変換
        target = torch.from_numpy(target)  # numpyからPyTorchの型へ変換
        return sample, target

    def __len__(self):
        return len(self.samples)

"""
モデルを訓練する train_model を定義する。 この関数は、大きく２つのfor文でできている。

"for epoch in range()"の中は、1エポックの実行である。
各ループの最後に、更新したモデルがバリデーションデータの上で精度が良くなっているか確認し、
よくなっていればモデルを保存する。

"for inputs, labels in dataloaders..."
の中は１つのバッチの実行である。今のモデルを利用したときの目的関数（スライドのL）の値の計算、
勾配降下法を実施を行う。
"""
def train_model(device, dataloaders, dataset_sizes, 
                model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    # 途中経過でモデル保存するための初期化
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000.0
    # 時間計測用
    end = time.time()

    train_loss_list = []
    val_loss_list = []

    for epoch in range(num_epochs):
        print('Epoch:{}/{}'.format(epoch, num_epochs - 1), end="")

        # 各エポックで訓練+バリデーションを実行
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                labels = labels.float()
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # 訓練のときだけ履歴を保持する
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, classnums = torch.max(labels, 1)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, classnums)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 統計情報
                running_loss += loss.item() * inputs.size(0)
                running_corrects += float(torch.sum(preds == classnums))

            if phase == 'train':
                scheduler.step()

            # サンプル数で割って平均を求める
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('\t{} Loss: {:.4f} Acc: {:.4f} Time: {:.4f}'.format(phase, epoch_loss, epoch_acc, time.time()-end), end="")
            if phase == 'train':
                train_loss_list.append(epoch_loss)
            else:
                val_loss_list.append(epoch_loss)

            # 精度が改善したらモデルを保存する
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            end = time.time()

        print()

    time_elapsed = time.time() - since
    print()
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Val loss: {:.4f}'.format(best_loss))
    # 一番 validationデータで精度の良かったモデルを読み出し
    model.load_state_dict(best_model_wts)    

    return model, train_loss_list, val_loss_list

def calc_test_accuracy(device, dataloader, dataset_size, model, criterion):
    running_loss = 0.0
    running_corrects = 0
    model.train(False)
    y_pred = []
    y_true = []

    for inputs, labels in dataloader:
        labels = labels.float()
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, classnums = torch.max(labels, 1)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, classnums)

        # 統計情報
        running_loss += loss.item() * inputs.size(0)
        running_corrects += float(torch.sum(preds == classnums))
        # 精度の計算
        y_pred = y_pred + list(preds.cpu().numpy())
        y_true = y_true + list(classnums.cpu().numpy())

    # サンプル数で割って平均を求める
    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects / dataset_size
    print('On Test:\tLoss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return y_true, y_pred

def train_and_test(filename, model, params):
    batch_size = params.get("batch_size", 64)
    epochs = params.get("epochs", 5)
    lr = params.get("lr", 0.0001)
    momentum = params.get("momentum", 0.90)
    step_size = params.get("step_size", 5)
    gamma = params.get("gamma", 0.1)

    if torch.cuda.is_available(): # GPUが利用可能か確認
        device = 'cuda'
    else:
        device = 'cpu'

    print("Settings:")
    print("\tDevice:", device)
    print("\tBatch size:", batch_size)
    print("\tEpochs:", epochs)
    print("\tLearning rate:",  lr)
    print("\tMomentum(SGD):", momentum)
    print("\tStep size for LR:", step_size)
    print("\tGamma for LR:", gamma)
    print()

    # 全データの読み込み
    X, y = make_dataset(filename)
    # テストデータの分割
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size = 0.20)
    # 訓練データとValidationデータの分割
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size = 0.25)

    print("# of samples:")
    print("\tTraining: {:d}".format(len(X_train)))
    print("\tValidation: {:d}".format(len(X_val)))
    print("\tTest: {:d}".format(len(X_test)))
    print()


    # データ読み込み用の関数を定義
    feature_datasets = {
        'train':DatasetFolder(X_train, y_train),
        'val':DatasetFolder(X_val, y_val),
        'test': DatasetFolder(X_test, y_test)
    }

    # バッチサイズ分のデータを読み込む。
    # training はデータをシャッフルし、読み込む画像の順番をランダムにする。
    # 他はシャッフルの必要なし。
    workers=0
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            feature_datasets['train'],
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers),
        'val': torch.utils.data.DataLoader(
            feature_datasets['val'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers),
        'test': torch.utils.data.DataLoader(
            feature_datasets['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers)
    }

    # 訓練, 評価, テストの画像の大きさをカウントする。
    dataset_sizes = {x: len(feature_datasets[x]) for x in ['train', 'val', 'test']}

    # 損失関数、
    # パラメータの最適化方法、学習率の更新方法を定義。
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # 実際の学習を実施
    model, train_loss_list, val_loss_list = train_model(device, dataloaders, dataset_sizes, model, 
                                                        criterion, optimizer, exp_lr_scheduler, num_epochs=epochs)

    # テストデータでの精度を求める
    y_true, y_pred = calc_test_accuracy(device, dataloaders['test'], dataset_sizes['test'], model, criterion)
    print(pd.DataFrame(confusion_matrix(y_true, y_pred), columns=[0,1]))
    
    return model, train_loss_list, val_loss_list

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(906, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
	        
if __name__ == "__main__":
    params = {
        "epochs":10,
        "batch_size":64,
        "lr":0.0005,
        "momentum":0.95,
    }
    # ネットワークをGPUに送る
    device = "cpu"
    model = Net()
    model = model.to(device)

    final_model, train_loss, val_loss = train_and_test("data/exp_data.csv", model, params)

    model_file_name = "best_model.torch"
    torch.save(final_model.state_dict(), model_file_name)

    p1 = plt.plot(list(range(len(train_loss))), train_loss, linestyle="dashed")
    p2 = plt.plot(list(range(len(val_loss))), val_loss, linestyle="solid")
    plt.legend((p1[0], p2[0]), ("Training", "Validation"), loc=1)
