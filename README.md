# Multi-classification-problem-using-CNN-RNN-and-attention
SUSTech EE271 Artificial Intelligence and Machine Learning

This dataset comprises a total of 8528 recordings with 188 features (the first 188 columns of ecg_data.csv) extracted from single ECG signals. There are supposed to be 4 categories labeled 1 through 4 (shown in the column of ecg_data.csv), corresponding to “Normal”, “Atrial Fibrillation (AF)”, “Non-AF related abnormal heart rhythms”, and noisy recording”. The distribution of normal, AF, other rhythms and noisy data is largely imbalanced in the dataset.

## Requirement

1. Try to use a fully connected feedforward deep network, a CNN (could be any modern CNN
network), a RNN (could be any RNN such as Pyramid RNN, LSTM, GRU, Grid LSTM), and an
attention network to solve the above 4-class classification problem.
2. Consider the following performance metrics: F1-score for normal ($𝐹_{norm}$), AF ($𝐹_{af}$), other rhythms ($𝐹_{oth}$), and the final accuracy 𝐹 = $\frac{1}{3}(𝐹_{norm}+𝐹_{af}+𝐹_{oth})$. Compare and analyze the results obtained from the four approaches in (1). It is suggested a 5-fold cross validation is considered to observe the performance.
