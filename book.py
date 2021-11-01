# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 19:45:06 2021

@author: Shubham
"""

import pandas as pd


book = pd.read_csv("D:\\Data Science study\\assignment\\Sent\\10 Recommendation system\\book.csv", encoding = "ISO-8859-1")

book.shape

book.columns

# Names of the columns are not usable so we are going to change them.

book.columns = ['srno', 'user_id', 'book_title', 'book_rating']

from sklearn.feature_extraction.text import TfidfVectorizer

#Checking whether there are any empty strings

book["book_title"].isnull().sum() 

# Lets continue

tfidf = TfidfVectorizer(stop_words="english")    #taking stop words from tfid vectorizer 

# Preparing the Tfidf matrix by fitting and transforming

tfidf_matrix_book = tfidf.fit_transform(book.book_title)   #Transform a count matrix to a normalized tf or tf-idf representation

tfidf_matrix_book.shape

# We are going to use the cosine simillarity matrix for calcualating the simillarity score

# book_rating_column = book.loc[:,'book_rating']
#######################################################################################
import numpy as np
empty_mat = np.zeros(book_rating_column.shape[1],book_rating_column.shape[1])









#######################################################################################################


#rating = book_rating_column.values
#rating.shape
#rating = rating.reshape(1, -1)

from sklearn.metrics.pairwise import linear_kernel

cosine_sim_matrix_book = linear_kernel(tfidf_matrix_book,tfidf_matrix_book)

# creating a mapping of book name to index number 

book_index = pd.Series(book.index,index=book['book_title']).drop_duplicates()

book_index["Jane Doe"]

def get_book_recommendations(Name,topN):
	# Getting the movie index using it's title
	book_id = book_index[Name]

	#Getting the pairwise simillarity score for all the books with that book
	cosine_score = list(enumerate(cosine_sim_matrix_book[book_id]))
	
	# Sorting cosine_simillarity scores based on the scores
	cosine_score = sorted(cosine_score,key=lambda x:x[1],reverse = True)

	#get the score of the top 10 books
	cosine_score_10 = cosine_score[0:topN+1]

	#Getting the book index
	book_idx = [i[0] for i in cosine_score_10]
	book_score = [i[1] for i in cosine_score_10]

	#simillar movies and scores
	
	simillar_books = pd.DataFrame(columns=["name", "score"])
	simillar_books["name"] = book.loc[book_idx,"name"]
	simillar_books["score"] = book_score
	simillar_book.reset_index(inplace=True)
	simillar_books.drop(["index"],axis=1, inplace = True)
	print(simillar_books)

get_book_recommendations("Night Watch", topN=10)
