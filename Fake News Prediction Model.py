#!/usr/bin/env python
# coding: utf-8

# # Problem Statement : Problem Statement :
# 
# # Fake News Classification with The Help Of Natural Language Fake News Classification with The Help Of Natural Language Processing Technique.

#  - Fake news detection is a hot topic in the field of natural language processing. We consume news through several mediums throughout the day in our daily routine, but sometimes it becomes difficult to decide which one is fake and which one is authentic. Our job is to create a model which predicts whether a given news is real or fake.

# In[ ]:





# In[1]:


import numpy as np
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


news_df = pd.read_csv('train.csv')


# In[3]:


news_df.head()


# # About the Dataset:
# 
# id: unique id for a news article                                     
# title: the title of a news article                                  
# author: author of the news article                                    
# text: the text of the article; could be incomplete                                        
# label: a label that marks whether the news article is real or fake:                              
#     1: Fake news                                                 
#     0: real News                                        

# # 1 Preprocessing 

# In[4]:


news_df.isnull().sum()


# In[5]:


news_df.shape


# In[6]:


news_df = news_df.fillna(' ')


# In[7]:


news_df.isnull().sum()


# In[8]:


news_df['content'] = news_df['author']+' '+news_df['title']


# In[9]:


news_df


# # separating the data & label

# In[10]:


X = news_df.drop('label',axis=1)
y = news_df['label']


# In[11]:


print(X)


# # Stemming:
# 
# Stemming is the process of reducing a word to its Root word
# 
# example: hung         hanged        hanging ======hang

# # Steps:
# lower case                 
# splitting                             
# removing stopwords                              
# stemming                                   

# In[12]:


ps = PorterStemmer()
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[13]:


news_df['content'] = news_df['content'].apply(stemming)


# In[14]:


news_df['content']


# # separating the data and label
# 

# In[15]:


X = news_df['content'].values
y = news_df['label'].values


# # converting the textual data to numerical data

# In[16]:


vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)


# In[17]:


print(X)


# # Splitting the dataset to training & test data

# In[18]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state=2)


# In[25]:


X_train.shape


# In[ ]:





# # Training the Model: Logistic Regression

# In[19]:


model = LogisticRegression()
model.fit(X_train,Y_train)


# In[20]:


# on training set
train_y_pred = model.predict(X_train)
print(accuracy_score(train_y_pred,Y_train))


# In[21]:


# on testing set
testing_y_pred = model.predict(X_test)
print(accuracy_score(testing_y_pred,Y_test))


# # Detection System

# In[22]:


input_data = X_test[10]
prediction = model.predict(input_data)


# In[23]:


if prediction[0] == 0:
    print('The News Is Real')
else:
    print('The News is Fake')


# In[24]:


news_df['content'][2]


# In[ ]:




