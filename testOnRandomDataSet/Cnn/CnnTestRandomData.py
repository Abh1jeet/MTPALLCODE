import numpy as np
import pandas as pd
import nltk
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
# multi-class classification with Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GlobalAveragePooling1D ,GlobalMaxPooling1D
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation ,Bidirectional
from keras.layers.embeddings import Embedding

from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline





############################################################ Data Preprocessing  ###########################################################



filePath="D:\\mtech4\\data\\combine3a.txt"
df = pd.read_csv(filePath, header = None, delimiter='\t')
rows_train,columns_train=df.shape
#df.columns = ['TID', 'Text','Tag','Label']
df.columns = ['Text','Label']
#neu 0
#neg 1
#pos 2


text_train=[]
label_train=[]
for i in df.Text:
	text_train.append(i)
for i in df.Label:
	label_train.append(i)

print(len(text_train))
print(len(label_train))


filePath="D:\\mtech4\\data\\randomData\\combineMatrix.txt"
df = pd.read_csv(filePath, header = None, delimiter='\t')
rows_test,columns_test=df.shape
df.columns = ['Text','Label']

#neu 0
#neg 1
#pos 2


text_test=[]
label_test=[]
for i in df.Text:
	text_test.append(i)
for i in df.Label:
	label_test.append(i)

print(len(text_test))
print(len(label_test))


##################################################################################################################################################



#################################################################### Train Test Preparation ######################################################

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

#total text
text_total= text_train + text_test 
label_total=label_train + label_test  
#pad sequences
max_length=max([len(s.split()) for s in text_total])


tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(text_total)

#define vocablury size
vocab_size=len(tokenizer_obj.word_index)+1


print("max length :" , max_length)
print("vocab size :" , vocab_size)



##generating tokens
text_token=tokenizer_obj.texts_to_sequences(text_total)

##adding padding 
text_token_withpad=pad_sequences(text_token,maxlen=max_length,padding='post')



#give number to all the different classes
encoder = LabelEncoder()
encoder.fit(label_total)
encoded_y = encoder.transform(label_total)

#give one hot encoding to each label 
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_y)



X_train=[]
X_test=[]
y_train=[]
y_test=[]


for i in range(0,rows_train):
	X_train.append(text_token_withpad[i])
	y_train.append(dummy_y[i])
for i in range(rows_train,rows_train + rows_test):
	X_test.append(text_token_withpad[i])
	y_test.append(dummy_y[i])

 
print("train : " , len(X_train))
print("test : " , len(X_test))


X_train=np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)


j=0
for i in y_train:
	print(i)
	if j>5:
		break
	j=j+1
        

#######building model#########
EMBEDDING_DIM=100
filters = 250
kernel_size = 3
hidden_dims = 250



# main model

model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(vocab_size,EMBEDDING_DIM,input_length=max_length))### size of vocab ,embedding dimension , length of sentence
model.add(Dropout(0.2))
# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,kernel_size,padding='valid',activation='relu',strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())
# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(3, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


print(model.summary())















#training
history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=2, batch_size=30)

#testing
y_pred = model.predict(X_test)
for i in y_pred:
	print(i)

pred = list()
for i in range(len(y_pred)):
    pred.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))

for i in range(0,len(test)):
	print(pred[i] , test[i])

from sklearn.metrics import accuracy_score
a = accuracy_score(pred,test)
print('Accuracy is:', a*100)