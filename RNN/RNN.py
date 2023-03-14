import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Embedding, LSTM, SpatialDropout1D
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
# import tensorflow as tf


BATCH_SIZE = 1


#数据集
path_train ='ecg_data.csv'
# 使用pandas读入，读取文件中所有数据
data_train = pd.read_csv(path_train,header=None)
#将dataframe转换成array
data_train=np.array(data_train).astype('float32')


# 去除最后一列
x=np.delete(data_train,[-1], axis=1)
#提取最后一列
y= data_train[:,[-1]]
# one-hot编码
encoder=LabelEncoder()
y= encoder.fit_transform(y)
y= np_utils.to_categorical(y)


#划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


#数据归一化，防止分配不均造成不良后果
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.fit_transform(x_test)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=188,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )

        self.out = nn.Linear(4, 1)

    def forward(self, x):
        r_out, (h_c, h_h) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)
loss_fun = nn.CrossEntropyLoss()


for step, (b_x, b_y) in enumerate(data_train):
    b_x = b_x.view(-1, 28, 28)

    r_out = rnn(b_x)
    loss = loss_fun(r_out, b_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step%50 == 0:
        test_out = rnn(x_test)
        y_predict = torch.max(test_out, 1)[1].data.numpy()
        accuracy = float((y_predict == y_test).astype(int).sum()) / float(y_test.size)
        print("loss=", loss.data.numpy(), " accuracy=%.2f" % accuracy)



