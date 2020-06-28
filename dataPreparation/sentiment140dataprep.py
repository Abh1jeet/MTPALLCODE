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

filePath="D:\\mtech4\\data\\sentiment140\\sentiment140train.csv"
df = pd.read_csv(filePath, header = None, delimiter=',')
rows,columns=df.shape
#df.columns = ['TID', 'Text','Tag','Label']
df.columns = ["Label","TID","Time","Query","User","Text"]


#0 negative
#2 neutral
#4 positive


text=[]
label=[]
for i in df.Text:
	text.append(i)


for i in df.Label:
	if i==2:
		print(i)

# for i in df.Label:
# 	if i=="0":
# 		label.append("negative")
# 	elif i=="2":
# 		label.append("neutral")
# 	else:
# 		label.append("positive")


# for i in label:
# 	print(i)

# # pos=0
# # neg=0
# # neu=0


# # for i in label:
# # 	if i=="positive":
# # 		pos=pos+1
# # 	elif i=="negative":
# # 		neg=neg+1
# # 	else:
# # 		neu=neu+1 

# # print(pos)
# # print(neg)
# # print(neu)


