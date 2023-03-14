# Multi-classification-problem-using-CNN-RNN-and-attention
SUSTech EE271 Artificial Intelligence and Machine Learning

Final Project Description:
This dataset comprises a total of 8528 recordings with 188 features (the first 188 columns of ecg_data.csv) extracted from single ECG signals. There are supposed to be 4 categories labeled 1 through 4 (shown in the column of ecg_data.csv), corresponding to “Normal”, “Atrial Fibrillation (AF)”, “Non-AF related abnormal heart rhythms”, and noisy recording”. The distribution of normal, AF, other rhythms and noisy data is largely imbalanced in the dataset.

1. Try to use a fully connected feedforward deep network, a CNN (could be any modern CNN
network), a RNN (could be any RNN such as Pyramid RNN, LSTM, GRU, Grid LSTM), and an
attention network to solve the above 4-class classification problem.
2. Consider the following performance metrics: F1-score for normal (𝐹 ), AF (𝐹 ), other !"#$ %&
rhythms (𝐹 ), and the final accuracy 𝐹 = 1(𝐹 + 𝐹 + 𝐹 ). Compare and analyze the "'( ) 3 !"#$ %& "'(
results obtained from the four approaches in (1). It is suggested a 5-fold cross validation is considered to observe the performance.

Requirement:
1. You should complete your project by yourself only, and then hand in your code as well as a project report before the deadline.
2. The midterm project takes 20 marks for the course (20%).
3. The deadline to submit your project report and code packages is 23:59PM of Dec. 31, 2022.
It is a firm deadline (Late submission will receive 0 mark).
4. When completing your course project, you are required to write a project report together
with the codes for the project. Base on the project report and the code package, the project
will be marked.
5. The project report should be written in English.
6. The project report should be presented in the IEEE conference paper style and suggest to
use LaTex if possible. Refer to the following link https://www.ieee.org/conferences/publishing/templates.html
for the LaTex Template (a LaTex template package is also included in the zipped file), or you can work in Overleaf (an online LaTex editor). The project report should contain the project title, authors, abstract, keywords, I. Introduction, II. Problem formulation, III. Method and algorithms, IV. Experiment results and analysis, V. Conclusion and future problems, and References.
7. Hand in a complete code package including the data set, code files with detailed description of dependencies, etc., so that the code can be checked and run on another computer without any problem.
8. The project and the codes should not be copied from others. Once it is noticed that the hand-in is copied from others including your classmates or online available work, you will receive 0 mark.
9. Mark criteria

evidence to show your effort in the project report.
1. Creativity (5 marks): You have to have your own idea and input to solve the problem
and analyze the results. Please be aware that the only way TA can understand your new inputs is from the project report. So please make sure that you have provided sufficient
2. Completeness (5 marks): The project should be a self-contained one. It should have a
clear problem formulation, followed by a complete solution and algorithm, as well as
experiment results and analysis, and concluding discussion.
3. Presentation of project report (5 marks): The report should be well organized and
clearly written. The problem under consideration and the developed solution (algorithm) should be clearly described. The simulation (experiment) results should be fully discussed and analyzed. Valuable conclusion should be provided. Any unclear points will get some marks off.
4. Presentation of the codes and codes comments (5 marks): The codes should meet a good Python coding style and easy for reading. Also, the codes should be accompanied with clear and detailed code comments. Any confusion in understanding the codes may lead to some marks off.
