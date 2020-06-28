from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import pandas as pd
import nltk

filePath="D:\\mtech4\\data\\combine3a.txt"
df = pd.read_csv(filePath, header = None, delimiter='\t')
rows,columns=df.shape
df.columns = ['Text','Label']




DataClass=[]
for i in range(0,rows):
	DataClass.append(1)

df['Class']=DataClass




filePath="D:\\mtech4\\data\\randomData\\combineMatrix.txt"
df2 = pd.read_csv(filePath, header = None, delimiter='\t')
rows,columns=df2.shape
df2.columns = ['Text','Tag']


DataClass=[]
for i in range(0,rows):
	DataClass.append(2)

df2['Class']=DataClass



##combine datasets to create dataframe




df3=pd.DataFrame()
df3['Text']=df['Text']
df3['Class']=df['Class']



df4=pd.DataFrame()
df4['Text']=df2['Text']
df4['Class']=df2['Class']





finaldf = pd.concat([df3, df4], ignore_index=True)
rows,columns=finaldf.shape



text=[]
label=[]
for i in finaldf.Text:
	text.append(i)
for i in finaldf.Class:
	if i==1:
		label.append(True)
	else:
		label.append(False)



X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=0.2, random_state=42)






from sklearn.svm import SVC
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SVC(kernel='poly',degree=8)),
                     ])
text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)

#print(metrics.classification_report(label_test, predicted))
accuracy=accuracy_score(y_test,predicted)
print('svm(Poly): {}'.format(accuracy))
y_pred_svm = text_clf.decision_function(X_test)



from sklearn.linear_model import LogisticRegression
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LogisticRegression()),
                     ])
text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)

#print(metrics.classification_report(label_test, predicted))
accuracy=accuracy_score(y_test,predicted)
print('LogisticRegression: {}'.format(accuracy))
y_pred_logistic = text_clf.decision_function(X_test)




for i in range(0,len(y_test)):
	print(y_test[i] , predicted[i])




from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
logistic_fpr, logistic_tpr, threshold = roc_curve(y_test, y_pred_logistic)
auc_logistic = auc(logistic_fpr, logistic_tpr)

svm_fpr, svm_tpr, threshold = roc_curve(y_test, y_pred_svm)
auc_svm = auc(svm_fpr, svm_tpr)

plt.figure(figsize=(5, 5), dpi=100)
plt.plot(svm_fpr, svm_tpr, linestyle='-', label='SVM (auc = %0.3f)' % auc_svm)
plt.plot(logistic_fpr, logistic_tpr, marker='.', label='Logistic (auc = %0.3f)' % auc_logistic)

plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')

plt.legend()

plt.show()
