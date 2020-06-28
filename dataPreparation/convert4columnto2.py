import pandas as pd 
import numpy as np 
import re
import json
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

import nltk
filePath="D:\\mtech4\\data\\combine12Mmat.txt"
df=pd.read_csv(filePath,header=None,skiprows=0,delimiter="\t")
rows,columns=df.shape
df.columns = ['TID', 'Text','Tag','Label']
print(rows)
print(columns)

outputFile="D:\\mtech4\\data\\combine12Mmat2.txt"
target=open(outputFile,"w")


print(df['Text'][0] , df['Label'][0])


for i in range(0,rows):
	target.write(str(df['Text'][i]))
	target.write('\t')
	target.write(str(df['Label'][i]))
	target.write('\n')

