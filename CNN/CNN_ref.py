import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils, plot_model
import matplotlib.pyplot as pl
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import itertools


# 载入数据
B_data = np.load('./data/data.npy')
B_label = np.load('./data/label.npy')
# B_data = np.transpose(B_data)
#print(B_data)
X = np.expand_dims(B_data[:, 0:128].astype(float), axis=2)
#print(X[1])
# print(X)
# print(X.shape)
# print('--------------')
Y = B_label
# 湿度分类编码为数字
encoder=LabelEncoder()
Y_encoded = encoder.fit_transform(Y)
Y_onehot = np_utils.to_categorical(Y_encoded)  # one-hot编码
# truelabel1 = Y_onehot.argmax(axis=-1)  # 将one-hot转化为label
# print('truelabel1\n')
# print(truelabel1)
# print(Y_onehot)
# print(Y_onehot.shape)
# print(Y_onehot[1])
# print(Y_onehot[898])


X_train, X_test, Y_train, Y_test = train_test_split(X, Y_onehot, test_size=0.5, random_state=0)


# print('X_train\n')
# print(X_train)
# print(X_train.shape)
# print('Y_train\n')
# print(Y_train)
# print(Y_train.shape)
# print('Y_test\n')
# print(Y_test)   # 对应标签onehot编码


# 定义神经网络
def baseline_model():
    model = Sequential()
    model.add(Conv1D(16, 3, input_shape=(128, 1)))
    model.add(Conv1D(16, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(6, activation='softmax'))
    plot_model(model, to_file='model_classifier.png', show_shapes=True)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# 训练分类器
estimator = KerasClassifier(build_fn=baseline_model, epochs=20, batch_size=1, verbose=1)  # 模型，轮数，每次数据批数，显示进度条
estimator.fit(X_train, Y_train)  # 训练模型


# 将其模型转换为json
model_json = estimator.model.to_json()
with open(r"model.json", 'w')as json_file:
    json_file.write(model_json)  # 权重不在json中,只保存网络结构
estimator.model.save_weights('model.h5')


# 加载模型用做预测
json_file = open(r"model.json", "r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# 分类准确率
print("The accuracy of the classification model:")
scores = loaded_model.evaluate(X_test, Y_test, verbose=0)
print('%s: %.2f%%' % (loaded_model.metrics_names[1], scores[1] * 100))

# 输出预测类别
predicted = loaded_model.predict(X)  # 返回对应概率值
print('predicted\n')
# print(predicted)
# predicted_label = loaded_model.predict_classes(X)  # 返回对应概率最高的标签
predicted_label = np.argmax(loaded_model.predict(X), axis=-1)
# print("\npredicted label: " + str(predicted_label))
# print(11111111111111111111111111111111111)
# # 显示混淆矩阵
# plot_confuse(estimator.model, X_test, Y_test)  # 模型，测试集，测试集标签
#
# # 可视化卷积层
# visual(estimator.model, X_train, 1)
