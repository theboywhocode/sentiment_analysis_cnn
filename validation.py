#importing important libraries
import json
import numpy as np
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
from keras.models import model_from_json
import pandas as pd

#Conversion of text in array format
def text_array(text):
    words = kpt.text_to_word_sequence(text)
    wordIndices = []
    for word in words:
        if word in dictionary:
            wordIndices.append(dictionary[word])
    return wordIndices

#Loading the dictionary from local machine
labels = ['Good','Bad']
with open('C:\\Users\\AMIT VIKRAM TRIPATHI\\Desktop\\Assignment\\dictionary.json', 'r') as dictionary_file:
    dictionary = json.load(dictionary_file)

#Loading pretrained model
json_file = open('C:\\Users\\AMIT VIKRAM TRIPATHI\\Desktop\\Assignment\\model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('C:\\Users\\AMIT VIKRAM TRIPATHI\\Desktop\\Assignment\\model.h5')

testset = pd.read_csv('C:\\Users\\AMIT VIKRAM TRIPATHI\\Desktop\\Assignment\\test.csv', sep='~')
cLen = len(testset['Description'])
tokenizer = Tokenizer(num_words=10000)

#Prediction of Review in test.csv
y_pred = []
for i in range(0,cLen):
    review = testset['Description'][i]
    testArr = text_array(review)
    input = tokenizer.sequences_to_matrix([testArr], mode='binary')
    pred = model.predict(input)
    y_pred.append(labels[np.argmax(pred)])

#SubmissionModel
raw_data = {'User_ID': testset['User_ID'],
        'Is_Response': y_pred}
validation = pd.DataFrame(raw_data, columns = ['User_ID', 'Is_Response'])
validation.to_csv('submissionModel.csv', sep='~',index=False)
