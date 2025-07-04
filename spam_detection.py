import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB  #classify the data
import streamlit as st

data = pd.read_csv(r"C:\Users\adgar\OneDrive\Documents\Projects\Email Spam detection\spam.csv")

#print(data.head()) #display names of first 5 data mails from dataset
#print(data.shape) #displays no. of rows, columns

data.drop_duplicates(inplace=True) #removes duplicates inplace is used when we make changes in original datasets

#print(data.shape) #displays no. of rows duplicates
#print(data.isnull().sum()) #if null values are present

data['Category'] = data['Category'].replace(['ham','spam'],['Not Spam','Spam']) #changes name of ham to not spam and in camel case
#print(data.head())
mess = data['Message']  #input dataset
category = data['Category']  #output dataset
(mess_train, mess_test, category_train, category_test) = train_test_split(mess, category, test_size=0.2)
cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

#creating model
model = MultinomialNB()
model.fit(features, category_train)

#Testing model
features_test = cv.transform(mess_test)
#print(model.score(features_test, category_test))

#predict data
def predict(message):
    input_message = cv.transform([message]).toarray()
    result=model.predict(input_message)
    return result
#output = predict('Congratulations you won a lottery')
#print(output)

st.header('Spam Dtection')
input_mess = st.text_input('Enter a  Message Here.')
if st.button('Validate'):
    output = predict(input_mess)
    st.markdown(output)