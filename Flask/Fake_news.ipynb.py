#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
from sklearn.pipeline import Pipeline


# In[4]:


df = pd.read_csv('C:/Users/v33an/Downloads/FAKE NEWS/Dataset/dataset.csv')
df.head()


# In[5]:


# Create a series to store the labels: y
# y = df.label


# In[6]:


#cleaned file containing the text and label
X = df['text']  # independent variable
y = df['label'] #dependent variable


# In[7]:


# Create training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.33, random_state=53)


# In[8]:


# Initialize a CountVectorizer object: count_vectorizer
count_vectorizer = CountVectorizer(stop_words='english')


# In[9]:


# Transform the training data using only the 'text' column values: count_train
count_train = count_vectorizer.fit_transform(X_train)


# In[10]:


# Transform the test data using only the 'text' column values: count_test
count_test = count_vectorizer.transform(X_test)


# In[12]:


# Print the first 10 features of the count_vectorizer
print(count_vectorizer.get_feature_names()[:10])


# In[13]:


# Initialize a TfidfVectorizer object: tfidf_vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)


# In[14]:


# Transform the training data: tfidf_train
tfidf_train = tfidf_vectorizer.fit_transform(X_train)


# In[15]:


# transform the test data: tfidf_test
tfidf_test = tfidf_vectorizer.transform(X_test)


# In[16]:


# Print the first 10 features
print(tfidf_vectorizer.get_feature_names()[:10])


# In[17]:


# Print the first 5 vectors of the tfidf training data
print(tfidf_train.A[:5])


# In[18]:


count_df = pd.DataFrame(count_train.A, columns=count_vectorizer.get_feature_names())


# In[21]:


# Create the TfidfVectorizer DataFrame: tfidf_df
tfidf_df = pd.DataFrame(tfidf_train.A, columns=tfidf_vectorizer.get_feature_names())


# In[22]:


# Print the head of count_df
print(count_df.head())


# In[23]:


# Print the head of tfidf_df
print(tfidf_df.head())


# In[24]:


# Calculate the difference in columns: difference
difference = set(count_df.columns) - set(tfidf_df.columns)
print(difference)


# In[25]:


# Check whether the DataFrame are equal
print(count_df.equals(tfidf_df))


# In[26]:


# Instantiate a Multinomial Naive Bayes classifier: nb_classifier
nb_classifier = MultinomialNB()


# In[27]:


# Fit the classifier to the training data
nb_classifier.fit(count_train, y_train)


# In[28]:


# Create the predicted tags: pred
pred = nb_classifier.predict(count_test)


# In[29]:


# Calculate the accuracy score: score
score = accuracy_score(y_test, pred)
print(score)


# In[30]:


# Calculate the confusion matrix: cm
cm =confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)


# In[31]:


#Training and testing the "fake news" model with TfidfVectorizer
nb_classifier = MultinomialNB()


# In[32]:


# Fit the classifier to the training data
nb_classifier.fit(tfidf_train, y_train)


# In[33]:


# Create the predicted tags: pred
pred = nb_classifier.predict(tfidf_test)


# In[34]:


# Calculate the accuracy score: score
score = accuracy_score(y_test, pred)
print(score)


# In[35]:


# Calculate the confusion matrix: cm
cm = confusion_matrix(y_test, pred, labels=['FAKE', 'REAL'])
print(cm)


# In[36]:


#Improving the model to test a few different alpha levels using the Tfidf vectors,
# to determine.if there is a better performing combination


# In[37]:


alphas = np.arange(0, 1, 0.1)


# In[38]:


# Define train_and_predict()
def train_and_predict(alpha):
    # Instantiate the classifier: nb_classifier
    nb_classifier = MultinomialNB(alpha=alpha)
    
    # Fit to the training data
    nb_classifier.fit(tfidf_train, y_train)
    
    # Predict the labels: pred
    pred = nb_classifier.predict(tfidf_test)
    
    # Compute accuracy: score
    score = accuracy_score(y_test, pred)
    return score


# In[39]:


# Iterate over the alphas and print the corresponding score
for alpha in alphas:
    print('Alpha: ', alpha)
    print('Score: ', train_and_predict(alpha))
    print()


# In[40]:


class_labels = nb_classifier.classes_


# In[41]:


# Extract the features: feature_names
feature_names = tfidf_vectorizer.get_feature_names()


# In[42]:


# Zip the feature names together with the coefficient array 
# and sort by weights: feat_with_weights
feat_with_weights = sorted(zip(nb_classifier.coef_[0], feature_names))


# In[43]:


# Print the first class label and the top 20 feat_with_weights entries
print(class_labels[0], feat_with_weights[:20])


# In[44]:


# Print the second class label and the bottom 20 feat_with_weights entries
print(class_labels[1], feat_with_weights[-20:])


# In[45]:


#Creating a pipeline that first creates bag of words(after applying stopwords) & then applies Multinomial Naive Bayes model
pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                     ('nbmodel', MultinomialNB())])


# In[46]:


#Training our data
pipeline.fit(X_train, y_train)


# In[47]:


#Predicting the label for the test data
pred = pipeline.predict(X_test)
print(confusion_matrix(y_test, pred))


# In[48]:


#Serialising the file
#pickle.dump(nb_classifier,open('fake_news.pkl','wb'))
#Serialising the file
with open('model.pkl', 'wb') as handle:
    pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:




