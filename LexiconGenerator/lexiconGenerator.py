import os
import numpy as np 
import pandas as pd 


filePath="D:\\mtech4\\data\\randomData\\combineMatrix.txt"

df=pd.read_csv(filePath,header=None,skiprows=0,delimiter="\t")
rows,columns=df.shape
df.columns = ['Text','Label']
print(rows)
print(columns)

outputFile="D:\\mtech4\\data\\lexiconGenerator.txt"
target=open(outputFile,"w")


dictd={}

#0 negative
#1 neutral
#2 positive

for i in range(0,rows):
	sentence=df['Text'][i].split()
	label=df['Label'][i]
	hashtag=[]
	for word in sentence:
		if word[0]=='#':
			if word[-1]=="." or word[-1]==',' or word[-1]=='"' or word[-1]=='!' or word[-1]=='?'or word[-1]==':'or word[-1]=="'": 
				hashtag.append(word[1:-1].lower())
			else:
				hashtag.append(word[1:].lower())

	for word in hashtag:
		if word in dictd:
			if label=="positive":
				dictd[word][2]=dictd[word][2]+1
			if label=="negative":
				dictd[word][0]=dictd[word][0]+1
			if label=="neutral":
				dictd[word][1]=dictd[word][1]+1
		else:
			if label=="positive":
				a=[]
				a.append(0)
				a.append(0)
				a.append(1)
				dictd[word]=a

			if label=="negative":
				a=[]
				a.append(1)
				a.append(0)
				a.append(0)
				dictd[word]=a

			if label=="neutral":
				a=[]
				a.append(0)
				a.append(1)
				a.append(0)
				dictd[word]=a



count=0
for key, value in dictd.items():
	print(key ,value)
	count=count+1

print(count)