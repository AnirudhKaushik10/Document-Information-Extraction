import pickle
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.constraints import max_norm


#Load the pickle file
dbfile = open('C:/Users/Acer/Desktop/df_train_api.pk', 'rb')
df=pickle.load(dbfile)

#Get stop words defined in NLTK library
stop_words = set(stopwords.words('english'))

# A list which will return the list of words after removing stop words
key_words=[]        
for x in df['groups']:
    #Convert to Lower Case
    x=x.lower() 
    # Remove Punctuations using Regex
    x = re.sub(r'[^a-zA-Z0-9\s]','', x)
    # Split the string on spaces
    x = x.split(' ')
    y=[]
    for word in x:
        # Removing all stops words and 1 letter words
        if word not in stop_words and len(word)>1:
            y.append(word)
    key_words.append(y)


from keras.preprocessing.text import Tokenizer
t=Tokenizer()
# Tokenize the Pre-processed Keywords and Get the counts of each word
t.fit_on_texts(key_words)
words_frequency=t.word_counts

# Get a sorted list of keywords based on the frequency of occurence
sorted_words=[]
for key,value in words_frequency.items():
    # Only Take words occuring more than 25 times in the dataset
    if value>25:
        sorted_words.append([key,value])

#Sorting the list
sorted_words=sorted(sorted_words,key=lambda l:l[1], reverse=True)

key_words=[]
for words,val in sorted_words:
    key_words.append(words)

i=-1
#Feature Vector
X=[]
#Label Vector
Y=[]
for x,y,lb in zip(df['groups'],df['coords'],df['label']):
    #x=x.split(" ")
    #for z in x:]
    x=x.lower()
    x = x.split(' ')
    # Filtering out words in dataset if they are either present in keywords list or don't have a label of 0
    for word in x:
        if word in key_words or lb!=0:
            X.append([])
            i=i+1
            X[i].append(word)
            for z in y:
                X[i].append(z)
            Y.append([lb])

#Convert to numpy array
X=np.asarray(X)
Y=np.asarray(Y)

#Convert the processed words of Feature vector into unique integers
label_encoder=LabelEncoder()
X[:,0]=label_encoder.fit_transform(X[:,0])

#Converting Feature Vector into Categorical values
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()

#Converting Labels into Categorical Values
onehotencoder_1=OneHotEncoder(categorical_features=[0])
Y=onehotencoder.fit_transform(Y).toarray()

#Normalize and Scale the Feature Vector
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

#Split the Dataset into Training and Validation Set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=0)

#Input Dimensions
dim=len(X_train[0])
# A 3 layer simple Artificial Neural Network
class model:
    def __init__(self, name, address=None):
        classifier=Sequential()
        #Layer1
        classifier.add(Dense(output_dim=512,init='uniform',activation='relu',input_dim=dim))
        classifier.add(Dropout(0.2))
        #Layer2
        classifier.add(Dense(output_dim=512,init='uniform',activation='relu'))
        classifier.add(Dropout(0.5))
        #Layer3
        classifier.add(Dense(output_dim=6,init='uniform',activation='softmax'))
        #Optimizer
        sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
        classifier.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
        
        #Fit on Training Dataset and Test on Validation Dataset
        classifier.fit(X_train,Y_train,batch_size=512,nb_epoch=50,validation_data=(X_test,Y_test))
        
        # Prediction for each label on validation data
        Y_pred=classifier.predict(X_test)
        #Selecting the the predicted label with the highest confidence percentage 
        y_test_non_category = [ np.argmax(t) for t in Y_test ]
        y_predict_non_category = [ np.argmax(t) for t in Y_pred ]
        
        #Make the confusion matrix of the predicted data
        from sklearn.metrics import confusion_matrix
        conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)
        
model('1')
