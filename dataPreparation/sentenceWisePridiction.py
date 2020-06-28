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

c=0
n=0
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
	countP=0
	countN=0
	countNe=0
	for tag in hashtag:
		if tag in hashtagDict:
			result.append(hashtagDict[tag])
			#print(tag ,hashtagDict[tag])
			ans=[]
			ans.append(hashtagDict[tag])
			ans.append(label)
			final[tag]=ans
			if hashtagDict[tag]=="positive":
				countP=countP+1
			elif hashtagDict[tag]=="negative":
				countN=countN+1
			else:
				countNe=countNe+1
		

	if len(hashtag)>0 and len(result)>0 :
		#print(df['Text'][i],label ,hashtag,result)
		#print(label ,hashtag,result)
		pridict="neutral"
		if countP>=countN and countP>=countNe:
			#print(label ,hashtag,result,"positive")
			pridict="positive"
		elif countN>=countP and countN>=countNe:
			#print(label,hashtag,result,"negative")
			pridict="negative"
		else: 
			#print(label,hashtag,result,"neutral")
			pridict="neutral"

		
		if label==pridict:	
			c=c+1
		else:
			n=n+1
			print(label,hashtag,result,pridict)

print(c,n)





