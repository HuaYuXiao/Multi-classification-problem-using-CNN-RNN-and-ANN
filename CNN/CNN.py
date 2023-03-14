import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


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


# 定义神经网络
def CNN():
    model =Sequential()
    model.add(Conv1D(16, 3, input_shape=(188, 1)))
    model.add(Conv1D(16, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# 训练分类器
model=KerasClassifier(build_fn=CNN)


#训练模型
model.fit(x_train, y_train)


# 预测
y_predict=model.predict(x_test)
y_predict= encoder.fit_transform(y_predict)
y_predict= np_utils.to_categorical(y_predict)
accuracy_score(y_test,y_predict)
print(classification_report(y_test,y_predict))
