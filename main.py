import csv
import random
import re
import string
import nltk
import pandas as pd
import gensim
from gensim.models import Word2Vec, KeyedVectors
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')
nltk.download('wordnet')


def read_data():
    # read the first CSV file into an array
    with open('./original_dataset/d_tweets.csv', newline='', encoding= 'utf-8') as csvfile:
        reader = csv.reader(csvfile)
        file1_data = [row[6] for row in reader]
    # add the label column to the first array
    #file1_data[0].append('label')
    file1_data.pop(0)
    label = []
    for i in range(1, len(file1_data)):
        label.append(1) # depressed

    print(file1_data[0])
    # read the second CSV file into an array
    with open('./original_dataset/non_d_tweets.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        file2_data = [row[6] for row in reader]

    # add the label column to the second array
    file2_data.pop(0)
    for i in range(1, len(file2_data)):
        label.append(0) # not depressed

    # combine the two arrays into one
    train_set = file1_data + file2_data
    #random.shuffle(train_set)
    combined = list(zip(train_set, label))
    random.shuffle(combined)
    train_set, label = zip(*combined)
    return train_set, label

def clean(train_set):
    new_train_set = []
    for tweet in train_set:
        tweet = re.sub(r'[^\w\s]', '', tweet) # remove anything that is not a space \s or a word \w
        tweet = tweet.split(' ') # split pe spatiu
        stop_words = stopwords.words('english')
        new_tweet = []
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        vectorizer = TfidfVectorizer()
        for word in tweet:
            if word not in stop_words:
                new_tweet.append(lemmatizer.lemmatize(word))
        new_train_set.append(new_tweet)
    return new_train_set

def text_vectorization(train_set):
    model = gensim.models.Word2Vec(window=5, min_count=2, workers=4, sg=0)
    model.build_vocab(train_set, progress_per = 1000)
    model.train(train_set, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("./responses.model")
    vectorized_train_set = []
    for tweet in train_set:
        vectorized_tweet = [model.wv[word] for word in tweet if word in model.wv.key_to_index]
        vectorized_train_set.append(vectorized_tweet)
    
    return vectorized_train_set


if __name__ == '__main__':
    train_set, label = read_data()
    train_set = clean(train_set)
    vectorized_train_set = text_vectorization(train_set)

