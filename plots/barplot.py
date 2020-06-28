import matplotlib.pyplot as plt
import numpy as np




Algorithm = ['DecisionTree', 'GradientBoosting', 'KNN', 'KNN(Bagging)', 'SVM','NaiveBayes','NearestCentroid',
'RandomForest','BiLstm','CNN','CNN+Lstm']

Accuracy = [88.72,84.31, 88.78,88.93,94.49,90.08,73.48,90.08,93.73,95.01
,94.26]

xpos = np.arange(len(Accuracy))



plt.barh(xpos,Accuracy, label="Accuracy")
plt.yticks(xpos,Algorithm)
plt.ylabel("Algorithm")
plt.xlabel("Accuracy")
plt.title('Results')
plt.legend()



for index, value in enumerate(Accuracy):
    plt.text(value, index, str(value))



plt.show()





