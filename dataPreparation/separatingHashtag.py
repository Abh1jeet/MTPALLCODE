import pandas as pd
filePath="D:\\mtech4\\data\\randomData\\combineMatrix.txt"
df=pd.read_csv(filePath,header=None,skiprows=0,delimiter="\t")
rows,columns=df.shape
df.columns = ['Text','Label']
print(rows)
print(columns)


# #finding hashtag present in each sentence

# count=0
# allhashtags=[]
# allhashtagsDict={}
# for i in range(0,rows):
# 	sentence=df['Text'][i].split()
# 	label=df['Label'][i]
# 	hashtags=[]
# 	for word in sentence:
# 		if word[0]=="#":
# 			hashtags.append(word)
# 			allhashtags.append(word[1:].lower())
# 			allhashtagsDict[word[1:].lower()]=label

# 	#print(hashtags)
# 	#print("##################")

# allhashtags=set(allhashtags)
# allhashtags=list(allhashtags)
# #print(allhashtags)
# #print(len(allhashtags))





##extracting all the hashtags we have
filePath2="D:\\mtech4\\data\\senti_emo_words_final.txt"
df2=pd.read_csv(filePath2,header=None,skiprows=0,delimiter="\t")
rows2,columns2=df2.shape
df2.columns = ['Hashtag','Label']


#creating dictionary of hashtags
hashtagDict={}
for i in range(0,rows2):
	hashtagDict[df2['Hashtag'][i]]=df2['Label'][i]




# hashtagDB=df2['Hashtag']
# count=0
# count2=0
# for word in hashtagDB:
# 	#print(word)	
# 	if word in allhashtags:
# 		count=count+1
# 		if hashtagDict[word]==allhashtagsDict[word]:
# 			print(word , hashtagDict[word])
# 			count2=count2+1
# print(count)
# print(count2)









from collections import OrderedDict

countC=0
countW=0
countA=0
countP=0
final={}
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

	result=[]
	for tag in hashtag:
		if tag in hashtagDict:
			result.append(hashtagDict[tag])
			#print(tag ,hashtagDict[tag])
			ans=[]
			ans.append(hashtagDict[tag])
			ans.append(label)
			final[tag]=ans
			countP=countP+1
			if label==hashtagDict[tag]:
				countC=countC+1
			else:
				countW=countW+1
		else:
			#print(tag ,"not found")
			#ans=[]
			#ans.append("not")
			#ans.append(label)
			#final[tag]=ans
			countA=countA+1
	#print(label ,hashtag,result)




outputFile="D:\\mtech4\\data\\temp.txt"
target=open(outputFile,"w")

for key in sorted(final.keys()):
	#print(key , final[key] )
	target.write(str(key))
	target.write('\t')
	target.write(str(final[key][0]))
	target.write('\t')
	target.write(str(final[key][1]))
	target.write('\n')




print("count of hashtag present: " , countP)
print("count of hashtag absent: " ,countA)
print("count of wrong pridictions: ",countW)
print("count of correct pridictions: ",countC)