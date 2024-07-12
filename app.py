import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt')

def text_transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    
    # removing special chars
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]                 # cloning because the list is mutable so after clearing y text will be cleared
    
    y.clear()
    # removing punctuations and stopwords
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    #stemming
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y) # return the list in the form of string


tfidf = pickle.load(open('vectorizer.pkl' , 'rb'))
model = pickle.load(open('model.pkl' , 'rb'))

st.title(" SMS Email Spam Calassifier")

txt_input = st.text_area(label="Enter your message: ")

if st.button("Predict"):
    # preprocess
    txt_transform = text_transform(txt_input)

    # vectorize
    vector_txt = tfidf.transform([txt_transform])

    #predict

    result = model.predict(vector_txt)[0]

    if result == 1:
        st.header("Spam")
    else: 
        st.header("Not Spam")
        


