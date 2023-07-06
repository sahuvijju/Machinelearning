#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


# In[2]:


credits_df = pd.read_csv("credits.csv")
movies_df = pd.read_csv("movies.csv")


# In[5]:


pd.set_option('display.max_columns',None)#full dataset show both dataset
pd.set_option('display.max_rows',None)


# In[6]:


credits_df.head()


# In[7]:


credits_df.tail()


# In[8]:


movies_df.head()


# In[9]:


movies_df.tail()


# In[10]:


movies_df = movies_df.merge(credits_df, on = 'title')


# In[11]:


movies_df.shape


# In[12]:


movies_df.head()


# In[13]:


print(movies_df.info())


# In[14]:


movies_df = movies_df[['id', 'title','overview','genres', 'keywords', 'cast', 'crew']]


# In[15]:


movies_df.head()


# In[16]:


movies_df.info()


# In[17]:


movies_df.isnull().sum()#returns missing value in dataset


# In[18]:


movies_df.dropna(inplace = True)#remove all missing values in original dataset


# In[19]:


movies_df.duplicated().sum()


# In[20]:


movies_df.iloc[0].genres#iloc funtion retrive anyperticular value in using index  value


# In[21]:


import ast  #abstract statement tree


# In[22]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[23]:


movies_df['genres'] = movies_df['genres'].apply(convert)
movies_df['keywords'] = movies_df['keywords'].apply(convert)
movies_df.head()


# In[24]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter !=3:
            L.append(i['name'])
            counter +=1
        else:
            break
        return L
    


# In[25]:


movies_df


# In[26]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
    return L


# In[27]:


movies_df['overview'][0]


# In[28]:


movies_df['overview'] = movies_df['overview'].apply(lambda x:x.split())


# In[29]:


movies_df


# In[30]:


movies_df['gerners']= movies_df['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['keywords']= movies_df['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['cast']= movies_df['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies_df['crew']= movies_df['genres'].apply(lambda x:[i.replace(" ","") for i in x])


# In[31]:


movies_df.head()


# In[32]:


movies_df['tags'] = movies_df['overview']+movies_df['genres']+movies_df['keywords']+movies_df['cast']+movies_df['crew']


# In[33]:


movies_df


# In[34]:


new_df = movies_df[['id','title','tags']]


# In[35]:


new_df


# In[36]:


new_df['tags'] = new_df['tags'].apply(lambda x:''.join(x))


# In[37]:


new_df


# In[38]:


new_df['tags'][0]


# In[39]:


new_df['tags']=new_df['tags'].apply(lambda X:X.lower())


# In[40]:


new_df.head()


# In[41]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')


# In[42]:


cv.fit_transform(new_df['tags']).toarray().shape


# In[43]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[44]:


vectors[0]


# In[45]:


import nltk


# In[46]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[47]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[48]:


new_df['tags']= new_df['tags'].apply(stem)


# In[49]:


from sklearn.metrics.pairwise import cosine_similarity


# In[50]:


cosine_similarity(vectors)


# In[51]:


cosine_similarity(vectors).shape


# In[52]:


similarity = cosine_similarity(vectors)


# In[53]:


similarity[0]


# In[54]:


similarity[0].shape


# In[55]:


sorted(list(enumerate(similarity[0])), reverse= True, key= lambda x:x[1])[1:6]


# In[56]:


def recommend(movie):
  
    
    movie_index = new_df[new_df['title']==movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse = True , key = lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[57]:


recommend('Harry Potter and the Half-Blood Prince')


# In[58]:


recommend('Iron Man')


# In[59]:


recommend('Liar Liar')


# In[60]:


recommend('Spider-Man')


# In[61]:


recommend('The Avengers')


# In[62]:


recommend('Captain America: Civil War')


# In[74]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

movies_df = pd.read_csv('movies.csv', low_memory=False)
# Preprocess data
movies_df['overview'] = movies_df['overview'].fillna('')  # Fill missing values
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_df['overview'])
# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
# Get movie recommendations
def get_recommendations(movie_title, cosine_sim, metadata):
    idx = metadata[metadata['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return metadata['title'].iloc[movie_indices]

movie_title = input("enter movie name :")

recommendations = get_recommendations(movie_title, cosine_sim, movies_df)

# Plotting the recommendations
plt.barh(recommendations, range(10), color='skyblue')
plt.xlabel("Similarity Score")
plt.ylabel("Movie Title")
plt.title(f"Top 10 Recommendations for '{movie_title}'")
plt.gca().invert_yaxis()

# Displaying the plot
plt.show()


# In[72]:


import matplotlib.pyplot as plt


genres = ["Action", "Comedy", "Drama", "Thriller", "Sci-Fi"]
recommendation_counts = [5008, 1058, 1808, 14570, 18798]

# Plotting the genre distribution
plt.bar(genres, recommendation_counts, color='skyblue')
plt.xlabel("Genres")
plt.ylabel("Recommendation Counts")
plt.title("Genre Distribution of Recommended Movies")

# Displaying the plot
plt.show()


# In[73]:


import matplotlib.pyplot as plt
# Sample data
genres = ["Action", "Comedy", "Drama", "Thriller", "Sci-Fi"]
genre_counts=[20,25,65,78,64]
# Plotting the genre distribution
plt.pie(genre_counts, labels=genres, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title("Movie Genre Distribution")

plt.show()


# In[75]:


recommendation_scores = [9, 4, 7, 4, 4, 5, 5, 2, 4, 5]
movie_genres = ['Action', 'Comedy', 'religious', 'Horror', 'Action', 'Comedy', 'religious', 'Horror', 'Action', 'Comedy']
data = {'Recommendation Score': recommendation_scores, 'Genre': movie_genres}
df = pd.DataFrame(data)
plt.figure(figsize=(10, 6))
sns.violinplot(x='Genre', y='Recommendation Score', data=df)
plt.xlabel('Movie Genre')
plt.ylabel('Recommendation Score')
plt.title('Movie Recommendation Scores by Genre')
plt.show()


# In[ ]:




