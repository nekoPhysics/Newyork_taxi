import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import math as m




# 読み込み数
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 500)
number = 100000
r = 6378.137


# データ読み込み
print("loading data")
df = pd.read_csv('train.csv', nrows=number)
# df = df.sample(n=number, random_state=777)

# 欠損値の除去
def cleanup(df):
    return df[(df.fare_amount > 0) & (df.passenger_count > 0) & (df.passenger_count < 7)
            & (df.pickup_longitude < -70) & (df.pickup_longitude > -80) & (df.pickup_latitude > 35) & (df.pickup_latitude < 45)
            & (df.dropoff_longitude < -70) & (df.dropoff_longitude > -80) & (df.dropoff_latitude > 35) & (df.dropoff_latitude < 45)]

print("cleaning data")
df = cleanup(df)

# 日時の情報を付加
def add_datetime_info(dataset):
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'],format="%Y-%m-%d %H:%M:%S UTC")
    
    dataset['hour'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['weekday'] = dataset.pickup_datetime.dt.weekday
    dataset['year'] = dataset.pickup_datetime.dt.year
    
    return dataset.drop(['pickup_datetime'],axis=1)


print("add datetime info")
df = add_datetime_info(df)


# 直線距離の情報を付加
def calc_distance(dataset):
    dataset['distance'] = r * m.acos((m.sin(dataset.pickup_latitude * m.pi/180) * m.sin(dataset.dropoff_latitude * m.pi/180))
                                         + (m.cos(dataset.pickup_latitude * m.pi/180) * m.cos(dataset.dropoff_latitude * m.pi/180) * m.cos((dataset.dropoff_longitude * m.pi/180) - (dataset.pickup_longitude * m.pi/180))))
    dataset.loc[(dataset['pickup_latitude'] == dataset['dropoff_latitude'])&(dataset['pickup_longitude'] == dataset['dropoff_longitude']), 'distance'] = 0
    
    return dataset


print("add distance info")
df = calc_distance(df)

# spliting
t = df['fare_amount']
x = df.drop('fare_amount', axis=1)

# convert to tensor
x = torch.tensor(x.values, dtype=torch.float32)
t = torch.tensor(t.values, dtype=torch.float32)
dataset = torch.utils.data.TensorDataset(x, t)

# 各データセットのサンプル数を決定
# train : val : test = 60% : 20% : 20%
n_train = int(len(dataset) * 0.6)
n_val = int(len(dataset) * 0.2)
n_test = len(dataset) - n_train - n_val

# ランダムに分割を行うため、シードを固定して再現性を確保
torch.manual_seed(0)

# データセットの分割
train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

      
# 学習データに対する処理
class TrainNet(pl.LightningModule):
    
    @pl.data_loader 
    def train_dataloader(self):
        return torch.utils.data.DataLoader(train, self.batch_size, shuffle=True)
    
    def training_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        results = {'loss': loss}
        return results
    
   
# 検証データに対する処理
class ValidationNet(pl.LightningModule):

    @pl.data_loader
    def val_dataloader(self):
        return torch.utils.data.DataLoader(val, self.batch_size)

    def validation_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        results = {'val_loss': loss}
        return results

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        results = {'val_loss': avg_loss}
        return results
    
 
# テストデータに対する処理
class TestNet(pl.LightningModule):

    @pl.data_loader
    def test_dataloader(self):
        return torch.utils.data.DataLoader(test, self.batch_size)

    def test_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        results = {'test_loss': loss}
        return results

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        results = {'test_loss': avg_loss}
        return results


# 学習データ、検証データ、テストデータへの処理を継承したクラス
class Net(TrainNet, ValidationNet, TestNet):
    
    def __init__(self, input_size=13, hidden_size=5, output_size=1, batch_size=10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.batch_size = batch_size

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
    
    # New: 平均ニ乗誤差
    def lossfun(self, y, t):
        return F.mse_loss(y, t)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)
    

# 再現性の確保
torch.manual_seed(0)

# インスタンス化
net = Net(input_size=18)
trainer = Trainer()

# 学習の実行
trainer.fit(net)

trainer.test()
print(trainer.callback_metrics)
