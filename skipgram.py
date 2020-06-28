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

filePath="D:\\mtech4\\data\\combine2.txt"
df = pd.read_csv(filePath, header = None, delimiter='\t')
rows,columns=df.shape
df.columns = ['TID', 'Text','Tag','Label']

#df.columns = ['Text','Label']




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


import nltk

from nltk.tokenize import word_tokenize,sent_tokenize
from nltk import ngrams,FreqDist
import matplotlib.pyplot as plt 




k=2 #sizeOFEmbedding
#finding embedding

import gensim 
from gensim.models import Word2Vec

  
   
data = []  
frequency={}


# tokenization(raw)

for i in text: 
    sentence=i.split()
    temp = []   
    #adding start token
    temp.insert(0,'<s>')
    # tokenize the sentence into words

    for word in sentence:
        temp.append(word.lower())
        if word.lower() not in frequency:
          frequency[word.lower()]=0

    #finding length
    tokenLen=len(temp)
    #adding stop token at the end of sentence
    temp.insert(tokenLen,'</s>')
    data.append(temp) 



#print(data)

# Create CBOW model 
model1 = gensim.models.Word2Vec(data, min_count = 1,  
                              size = k, window = 5) 
  
# Print results 

#print(model1.wv['the'])


vocab=[]
for key,value in frequency.items():
  vocab.append(key);

#adding start and stop token to vocabulary
vocab.append('<s>')
vocab.append('</s>')
# for i in vocab:
#   print(i)


Word2VecDic={}    #word2vecDic will contain word vector of each word in vocublury
for i in vocab:
  Word2VecDic[i]=model1.wv[i]
  #print(i,model1.wv[i])






##finding word vector of sentence


feature_text=[]
for sentence in text:
    ans=[]
    ans.append(0)
    ans.append(0)  
    for word in sentence.split():
        vector=Word2VecDic[word.lower()]
        ans[0]=ans[0]+vector[0]
        ans[1]=ans[1]+vector[1]
    feature_text.append(ans)


#print(feature_text)

X_train, X_test, y_train, y_test = train_test_split(feature_text, label, test_size=0.33, random_state=42)

from sklearn.svm import SVC
model_SVC = SVC(kernel = 'rbf', random_state = 4)
model_SVC.fit(X_train, y_train)

y_pred_svm_Mat = model_SVC.decision_function(X_test)


# from sklearn.linear_model import LogisticRegression
# model_logistic = LogisticRegression()
# model_logistic.fit(X_train, y_train)

# y_pred_logistic = model_logistic.decision_function(X_test)















filePath="D:\\mtech4\\data\\combine2.txt"
df = pd.read_csv(filePath, header = None, delimiter='\t')
rows,columns=df.shape
df.columns = ['TID', 'Text','Tag','Label']

#df.columns = ['Text','Label']




DataClass=[]
for i in range(0,rows):
  DataClass.append(1)

df['Class']=DataClass




filePath="D:\\mtech4\\data\\randomData\\combineSem13.txt"
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


import nltk

from nltk.tokenize import word_tokenize,sent_tokenize
from nltk import ngrams,FreqDist
import matplotlib.pyplot as plt 




k=2 #sizeOFEmbedding
#finding embedding

import gensim 
from gensim.models import Word2Vec

  
   
data = []  
frequency={}


# tokenization(raw)

for i in text: 
    sentence=i.split()
    temp = []   
    #adding start token
    temp.insert(0,'<s>')
    # tokenize the sentence into words

    for word in sentence:
        temp.append(word.lower())
        if word.lower() not in frequency:
          frequency[word.lower()]=0

    #finding length
    tokenLen=len(temp)
    #adding stop token at the end of sentence
    temp.insert(tokenLen,'</s>')
    data.append(temp) 



#print(data)

# Create CBOW model 
model1 = gensim.models.Word2Vec(data, min_count = 1,  
                              size = k, window = 5) 
  
# Print results 

#print(model1.wv['the'])


vocab=[]
for key,value in frequency.items():
  vocab.append(key);

#adding start and stop token to vocabulary
vocab.append('<s>')
vocab.append('</s>')
# for i in vocab:
#   print(i)


Word2VecDic={}    #word2vecDic will contain word vector of each word in vocublury
for i in vocab:
  Word2VecDic[i]=model1.wv[i]
  #print(i,model1.wv[i])






##finding word vector of sentence


feature_text=[]
for sentence in text:
    ans=[]
    ans.append(0)
    ans.append(0)  
    for word in sentence.split():
        vector=Word2VecDic[word.lower()]
        ans[0]=ans[0]+vector[0]
        ans[1]=ans[1]+vector[1]
    feature_text.append(ans)


#print(feature_text)

X_train_13, X_test_13, y_train_13, y_test_13 = train_test_split(feature_text, label, test_size=0.33, random_state=42)

from sklearn.svm import SVC
model_SVC = SVC(kernel = 'rbf', random_state = 4)
model_SVC.fit(X_train_13, y_train_13)

y_pred_svm_13 = model_SVC.decision_function(X_test_13)































filePath="D:\\mtech4\\data\\combine2.txt"
df = pd.read_csv(filePath, header = None, delimiter='\t')
rows,columns=df.shape
df.columns = ['TID', 'Text','Tag','Label']

#df.columns = ['Text','Label']




DataClass=[]
for i in range(0,rows):
  DataClass.append(1)

df['Class']=DataClass




filePath="D:\\mtech4\\data\\randomData\\combineSem16.txt"
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


import nltk

from nltk.tokenize import word_tokenize,sent_tokenize
from nltk import ngrams,FreqDist
import matplotlib.pyplot as plt 




k=2 #sizeOFEmbedding
#finding embedding

import gensim 
from gensim.models import Word2Vec

  
   
data = []  
frequency={}


# tokenization(raw)

for i in text: 
    sentence=i.split()
    temp = []   
    #adding start token
    temp.insert(0,'<s>')
    # tokenize the sentence into words

    for word in sentence:
        temp.append(word.lower())
        if word.lower() not in frequency:
          frequency[word.lower()]=0

    #finding length
    tokenLen=len(temp)
    #adding stop token at the end of sentence
    temp.insert(tokenLen,'</s>')
    data.append(temp) 



#print(data)

# Create CBOW model 
model1 = gensim.models.Word2Vec(data, min_count = 1,  
                              size = k, window = 5) 
  
# Print results 

#print(model1.wv['the'])


vocab=[]
for key,value in frequency.items():
  vocab.append(key);

#adding start and stop token to vocabulary
vocab.append('<s>')
vocab.append('</s>')
# for i in vocab:
#   print(i)


Word2VecDic={}    #word2vecDic will contain word vector of each word in vocublury
for i in vocab:
  Word2VecDic[i]=model1.wv[i]
  #print(i,model1.wv[i])






##finding word vector of sentence


feature_text=[]
for sentence in text:
    ans=[]
    ans.append(0)
    ans.append(0)  
    for word in sentence.split():
        vector=Word2VecDic[word.lower()]
        ans[0]=ans[0]+vector[0]
        ans[1]=ans[1]+vector[1]
    feature_text.append(ans)


#print(feature_text)

X_train_16, X_test_16, y_train_16, y_test_16 = train_test_split(feature_text, label, test_size=0.33, random_state=42)

from sklearn.svm import SVC
model_SVC = SVC(kernel = 'rbf', random_state = 4)
model_SVC.fit(X_train_16, y_train_16)

y_pred_svm_16 = model_SVC.decision_function(X_test_16)














































































from sklearn.metrics import roc_curve, auc

# logistic_fpr, logistic_tpr, threshold = roc_curve(y_test, y_pred_logistic)
# auc_logistic = auc(logistic_fpr, logistic_tpr)

svm_fpr, svm_tpr, threshold = roc_curve(y_test, y_pred_svm_Mat)
auc_svm = auc(svm_fpr, svm_tpr)

svm_fpr_13, svm_tpr_13, threshold_13 = roc_curve(y_test_13, y_pred_svm_13)
auc_svm_13 = auc(svm_fpr_13, svm_tpr_13)


svm_fpr_16, svm_tpr_16, threshold_16 = roc_curve(y_test_16, y_pred_svm_16)
auc_svm_16 = auc(svm_fpr_16, svm_tpr_16)



plt.figure(figsize=(5, 5), dpi=100)
plt.plot(svm_fpr, svm_tpr, linestyle='-', label='SVM Matrix (auc = %0.3f)' % auc_svm)
plt.plot(svm_fpr_13, svm_tpr_13, marker='.', label='SVM Sem13 (auc = %0.3f)' % auc_svm_13)
plt.plot(svm_fpr_16, svm_tpr_16, marker='.', label='SVM Sem16 (auc = %0.3f)' % auc_svm_16)


plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')

plt.legend()

plt.show()





