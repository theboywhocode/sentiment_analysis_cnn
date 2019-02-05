### Overview 
This project is about the sentiment analysis of the reviews using the deep neural network. Using the model we can easily classify any review as the 
as Good or Bad. The whole project can be divided into the 5 parts:

**1. Data Preparation:**
- Training data is in form of .csv, pandas(pd) is used to read the data from the csv file. 
- The ‘Is_Response’ have two values namely ‘Good’ & ‘Bad’ which is further encoded into the integer values 0 and 1.

**2. Feature Extraction:**
- For the feature extraction, I have used bag of words model.
- While training the test dataset all the sentences are divided into subparts and stored in a dictionary.
- This dictionary contained the indices of the words in the form of tuples.

**3. Building the Analysis Model:**
- Deep neural network is used to extract the feature and classify the data provided.
- Sequential model is used with the 4 dense layer. Initially I have used ‘ReLU’ as the activation function and at the last layer ‘Softmax’ is used.
- Top to down approach in number of dense layer neurons helped in extracting better features. At last layer 2 neurons are used for the prediction purpose.
 
**4. Training of Model:**
- While training the model I have used cross entropy as the loss function.
- I have trained the model on AMD R4 Redon (Graphics) and it took me 25 min to run the 5 epoch.
- Stochastic Gradient Descent is used as the learning algorithm which supports the forward as well as backward propagation.

**5. Model Validation:**
- In this part a script is written which can be used impoart the validation data and generate the result over it. 

To know more details please visit the [blog](https://appliedmachinelearning.blog/2017/12/21/predict-the-happiness-on-tripadvisor-reviews-using-dense-neural-network-with-keras-hackerearth-challenge/) by Abhijeet Singh
