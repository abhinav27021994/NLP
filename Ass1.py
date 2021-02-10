# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 22:40:55 2021

@author: Abhinav
"""
import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import sent_tokenize

#ELIMINATION OF STOPWORDS IN MOVIE REVIEW
movie_reviews.fileids()
review=movie_reviews.raw('pos/cv708_28729.txt ')
print(review)

stop_words=stopwords.words("english")
print(stop_words)

filtered_words=[]


filtered_words=[word for word in word_tokenize(review) if word not in stop_words]
print('Words except stop_words:\n',filtered_words)
print('Length of words excluding stop words: ',len(filtered_words))


#APPLY PUNKTSENTENCE TOKENIZER in movie review corpus

train_review=movie_reviews.raw('neg/cv989_17297.txt')
print(train_review)
test_review=movie_reviews.raw('neg/cv957_9059.txt')
print(test_review)

review_tokenize=PunktSentenceTokenizer(train_review)
tokenized_review=review_tokenize.tokenize(test_review)
tokenized_review

def process():
    for val in tokenized_review:
        words=word_tokenize(val)
        tag=nltk.pos_tag(words)
        print(tag)
process()
