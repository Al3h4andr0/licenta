import csv
import random
import re
import string
import nltk
import pandas as pd
import gensim
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.layers import Embedding
from keras.preprocessing import sequence
from keras.preprocessing import sequence
from gensim.models import Word2Vec, KeyedVectors
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
nltk.download('stopwords')
nltk.download('wordnet')

embedding_dim = 0
vocab_size = 0
maximum_length = 0


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
    model = gensim.models.Word2Vec(window=5, min_count=2, workers=4, sg=0, max_vocab_size=138)
    model.build_vocab(train_set, progress_per=1000)
    model.train(train_set, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("./responses.model")
    vectorized_train_set = []
    words_used = []
    max_len = 0
    for tweet in train_set:
        vectorized_tweet = []
        for word in tweet:
            if word.lower() in model.wv.key_to_index:
                vectorized_tweet.append(model.wv[word.lower()])
                if word.lower() not in words_used:
                    words_used.append(word.lower())
            else:
                vectorized_tweet.append([0]*model.vector_size)
        if not vectorized_tweet:
            vectorized_tweet.append([0]*model.vector_size)
        if len(vectorized_tweet) > max_len : 
            max_len = len(vectorized_tweet)
        vectorized_train_set.append(vectorized_tweet)
    global embedding_dim, maximum_length, vocab_size
    embedding_dim = max_len
    maximum_length = max_len
    print("MAAAAAAAAAAAAAAX LENNNNNNNNNNNN: ", max_len)
    vocab_size = len(words_used)
    padded_train_set = []
    for tweet in vectorized_train_set:
        padded_tweet = tweet + [[0] * model.vector_size]*(max_len - len(tweet))
        padded_train_set.append(padded_tweet)
    return padded_train_set

def predict(X_train, y_train, X_test):
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def LSTMnn(X_train, X_test, y_train, y_test, vocab_size, embedding_dim, max_len):
    model = Sequential()
    print(vocab_size, " ++++++++++ ", embedding_dim)
    model.add(Embedding(input_dim=vocab_size, output_dim = embedding_dim, input_length = 138))
    model.add(LSTM(units = 128, dropout = 0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 10, batch_size=32)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("LSTM Neural network accuracy : ", test_acc)


if __name__ == '__main__':
    train_set, label = read_data()
    train_set = clean(train_set)
    vectorized_train_set = text_vectorization(train_set)
    X_train, X_test, y_train, y_test = train_test_split(vectorized_train_set, label, test_size=0.2, random_state=42)
    X_train = [np.array(x).flatten() for x in X_train]
    X_test = [np.array(x).flatten() for x in X_test]
    #linear regression
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_predict = predict(X_train, y_train, X_test)
    print(classification_report(y_test, y_predict))
    #lstm nn
    # X_train = np.reshape(X_train, (-1, 138))
    # X_test = np.reshape(X_test, (-1, 138))

    lstmnn = LSTMnn(X_train, X_test, y_train, y_test, vocab_size, embedding_dim, maximum_length)


