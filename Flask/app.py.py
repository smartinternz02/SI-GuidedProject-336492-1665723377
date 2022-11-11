#Importing the Libraries

#flask is use for run the web application.
import flask
#request is use for accessing file which was uploaded by the user on our application.
from flask import Flask, request,render_template 
#from flask_cors import CORS

#Python pickle module is used for serializing
# and de-serializing a Python object structure.
import pickle

#OS module in python provides functions for interacting with the operating system
import os

#Newspaper is used for extracting and parsing newspaper articles.
#For extracting all the useful text from a website.
#from newspaper import Article

#URLlib is use for the urlopen function and is able to fetch URLs.
#This module helps to define functions and classes to open URLs 
import urllib

#Loading Flask and assigning the model variable
app = Flask(__name__)
#CORS(app)
app=flask.Flask(__name__,template_folder='templates')

with open(r"C:/Users/v33an/Downloads/FAKE NEWS/Flask/Fake_news.pkl", 'rb') as handle:
    model = pickle.load(handle)


@app.route('/') #default route
def main():
    return render_template('main.html')

#Receiving the input url from the user and using Web Scrapping to extract the news content

#Route for prediction

@app.route('/predict',methods=['GET','POST'])


def predict():
	#Contains the incoming request data as string in case.	
    url =request.get_data(as_text=True)[5:]
	
	#The URL parsing functions focus on splitting a URL string into its components, 
	#or on combining URL components into a URL string.
    url = urllib.parse.unquote(url)
	
	#A new article come from Url and convert onto string
    article = Article(str(url))	
	
	#To download the article 
    article.download()
	
	#To parse the article 
    article.parse()
	
	#To perform natural language processing ie..nlp
    article.nlp()
	#To extract summary 
    news = article.summary
    print(type(news))

    #Passing the news article to the model and returing whether it is Fake or Real
    pred = model.predict([news])
    print(pred)
    return render_template('main.html', prediction_text='The news is "{}"'.format(pred[0]))
    
if __name__=="__main__":
    app.run(debug=False)


    