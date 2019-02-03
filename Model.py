#Importing the important libraries
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.utils import np_utils
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import json

#importing the dataset and visualising it
train_review=pd.read_csv('C:\\Users\\AMIT VIKRAM TRIPATHI\\Desktop\\assignment_\\train.csv', sep='~')
print(type(train_review))
train_review.head(10)

#droping out the unnecessary info
train_review.drop(['User_ID','Browser_Used','Device_Used'],axis=1,inplace=True)

#Data preprocessing
def preprocessing(x):
    dataset = x
    reviews = []
    ratings = []

    # Enconding Categorical Data
    encoder = LabelEncoder()
    dataset['Is_Response'] = encoder.fit_transform(dataset['Is_Response'])
    cLen = len(dataset['Description'])

    for i in range(0,cLen):
        review = dataset['Description'][i]
        reviews.append(review)
        rating = dataset["Is_Response"][i]
        ratings.append(rating)
    ratings = np.array(ratings)
    return reviews,ratings


def text_array(text):
    return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

x=train_review
[reviews,ratings] = preprocessing(x)

#creating a dictionary of words with their indices
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(reviews)
dictionary = tokenizer.word_index

#saving the dictionary on local machine
with open('dictionary.json','w') as dictionary_file:
    json.dump(dictionary,dictionary_file)

#replacement of word with wordIndices
totalWordIndices = []
for num,text in enumerate(reviews):
    wordIndices = text_array(text)
    totalWordIndices.append(wordIndices)

#using one hot encoding
totalWordIndices = np.array(totalWordIndices)
x_train = tokenizer.sequences_to_matrix(totalWordIndices, mode='binary')
ratings = keras.utils.to_categorical(ratings,num_classes=2)

#Creating Dense Neural Network Model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

#Training and validation of dataset
#training
batch_size = 32
model.compile(loss='categorical_crossentropy', optimizer= 'sgd', metrics=['accuracy'])
cifar_train = model.fit(x_train, ratings,
                        batch_size=batch_size,
                        epochs=5,
                        verbose=1,
                        validation_split=0.1,
                        shuffle=True)

#Save model to local machine
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')
