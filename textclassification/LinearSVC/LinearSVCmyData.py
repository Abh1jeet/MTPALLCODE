from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import nltk
from sklearn.metrics import accuracy_score

filePath="D:\\mtech4\\data\\randomData\\combineSem16.txt"
df = pd.read_csv(filePath, header = None, delimiter='\t')
rows,columns=df.shape
#df.columns = ['TID', 'Text','Tag','Label']
df.columns = ['Text','Label']

#neu 0
#neg 1
#pos 2


text=[]
label=[]
for i in df.Text:
	text.append(i)
for i in df.Label:
	label.append(i)



X_train, X_test, y_train, y_test = train_test_split(text, label, test_size=0.33, random_state=42)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LinearSVC()),
                     ])

text_clf.fit(X_train, y_train)


predicted = text_clf.predict(X_test)

print(metrics.classification_report(y_test, predicted))

accuracy=accuracy_score(y_test,predicted)
print('svm(Linear): {}'.format(accuracy))