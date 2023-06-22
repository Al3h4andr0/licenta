import csv
import random
import re
import nltk
import gensim
import numpy as np
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW
from torch.utils.data import TensorDataset
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
import torch

nltk.download('stopwords')
nltk.download('wordnet')

embedding_dim = 0
vocab_size = 0
maximum_length = 0


def pad_sequences(sequences, max_len, value=0):
    padded_sequences = []
    for seq in sequences:
        # Calculate how much to pad
        padding_needed = max_len - len(seq)
        if padding_needed > 0:
            # Pad the sequence with zeros at the end
            seq = np.pad(seq, ((0, padding_needed), (0, 0)), 'constant', constant_values=value)
        else:
            # Or truncate the sequence if it's longer than max_len
            seq = seq[:max_len]
        padded_sequences.append(seq)
    return np.array(padded_sequences)


def read_data():
    # read the first CSV file into an array
    with open('./original_dataset/d_tweets.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        file1_data = [row[6] for row in reader]
    # add the label column to the first array
    file1_data.pop(0)
    label = []
    for i in range(1, len(file1_data)):
        label.append(1)  # depressed

    # read the second CSV file into an array
    with open('./original_dataset/non_d_tweets.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        file2_data = [row[6] for row in reader]

    # add the label column to the second array
    file2_data.pop(0)
    for i in range(1, len(file2_data)):
        label.append(0)  # not depressed

    # combine the two arrays into one
    train_set = file1_data + file2_data
    # random.shuffle(train_set)
    combined = list(zip(train_set, label))
    random.shuffle(combined)
    train_set, label = zip(*combined)
    combined = list(zip(train_set, label))
    random.shuffle(combined)
    train_set, label = zip(*combined)
    return list(train_set), list(label)


def clean(train_set):
    new_train_set = []
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english')

    for tweet in train_set:
        # Don't remove emojis and other characters as they may hold sentiment
        tweet = re.sub(r'http\S+|www.\S+', '', tweet)  # remove urls
        tweet = tweet.split(' ')

        new_tweet = []
        for word in tweet:
            if word not in stop_words:
                new_tweet.append(lemmatizer.lemmatize(word))

        new_train_set.append(new_tweet)
    return new_train_set


def text_vectorization_word_2_vec(train_set):
    model = gensim.models.Word2Vec(window=5, min_count=2, workers=4)
    model.build_vocab(train_set, progress_per=1000)
    model.train(train_set, total_examples=model.corpus_count, epochs=model.epochs)
    model.save("./responses.model")

    vectorized_train_set = []
    max_len = 0
    for tweet in train_set:
        vectorized_tweet = []
        for word in tweet:
            if word.lower() in model.wv.key_to_index:
                vectorized_tweet.append(model.wv[word.lower()])
            else:
                # Instead of zero vector, add a random vector or predefined unknown vector
                vectorized_tweet.append(np.random.rand(model.vector_size))

        if len(vectorized_tweet) > max_len:
            max_len = len(vectorized_tweet)
        vectorized_train_set.append(vectorized_tweet)

    padded_train_set = pad_sequences(vectorized_train_set, max_len)

    return padded_train_set, max_len, len(model.wv.key_to_index)


def text_vectorization_tf_idf(train_set):
    # Flattening the lists
    flat_train_set = [' '.join(tweet) for tweet in train_set]

    # Initialize the tf-idf vectorizer
    vectorizer = TfidfVectorizer()

    # Vectorize the training set
    tfidf_train_set = vectorizer.fit_transform(flat_train_set)

    return tfidf_train_set.toarray(), len(vectorizer.get_feature_names_out())


def predict(X_train, y_train, X_test):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred


# Define a function to tokenize the data and return TensorDatasets for the model
def tokenize_data_BERT(train_set, label, max_len=128):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Join the words in the tweet back into a single sentence
    sentences = [' '.join(tweet) for tweet in train_set]

    # Tokenize all the sentences and map the tokens to their word IDs
    input_ids = []
    attention_masks = []

    # For every sentence
    for sent in sentences:
        # Encode sentence to get its ids, mask and token type ids
        # Truncation and padding will be done automatically in `encode_plus`
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )

        # Add the encoded sentence to the list
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding)
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(label)

    # Create TensorDataset
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return dataset


def train_model_BERT(dataset, model, epochs=4, batch_size=32):
    # Split the dataset into training and validation datasets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create dataloaders for training and validation datasets
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=batch_size
    )

    validation_dataloader = DataLoader(
        val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=batch_size
    )

    # Use AdamW optimizer (AdamW is the version of Adam used in huggingface's transformers)
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

    # Create a learning rate scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Start training loop
    for epoch_i in range(0, epochs):
        print(f'======== Epoch {epoch_i + 1} / {epochs} ========')

        # Measure how long the training epoch takes
        total_train_loss = 0

        # Put the model into training mode
        model.train()

        for step, batch in enumerate(train_dataloader):
            # Unpack batch
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)

            # Clear any previously calculated gradients before performing a backward pass
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch)
            outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks, labels=labels)

            # Accumulate the training loss over all the batches so that we can calculate the average loss at the end
            total_train_loss += outputs.loss.item()

            # Perform a backward pass to calculate the gradients
            outputs.loss.backward()

            # Clip the norm of the gradients to 1.0 (to prevent "exploding gradients")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update the learning rate
            scheduler.step()

        # Calculate the average loss over all the batches
        avg_train_loss = total_train_loss / len(train_dataloader)

        print(f'Train loss: {avg_train_loss}')

        # ========================================
        #               Validation
        # ========================================

        print("\nRunning Validation...")

        # Put the model in evaluation mode
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask, labels=labels)

            total_eval_loss += outputs.loss.item()

            # Calculate the accuracy for this batch of test sentences, and accumulate it over all batches
            total_eval_accuracy += flat_accuracy(outputs.logits, labels)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        print(f'Validation Accuracy: {avg_val_accuracy}')
        print(f'Validation loss: {avg_val_loss}')

    print("\nTraining complete!")

    return model


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return torch.sum((pred_flat == labels_flat).float()) / len(labels_flat)


if __name__ == "__main__":
    # Read and clean the data
    train_set, label = read_data()
    train_set = clean(train_set)

    # Vectorize the text using Word2Vec and TF-IDF
    vectorized_train_set_word2vec, max_len_word2vec, vocab_size_word2vec = text_vectorization_word_2_vec(train_set)
    vectorized_train_set_tf_idf, vocab_size_tf_idf = text_vectorization_tf_idf(train_set)

    # Split the data into train and test sets
    X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(
        vectorized_train_set_word2vec, label, test_size=0.2, random_state=42
    )

    X_train_tf_idf, X_test_tf_idf, y_train_tf_idf, y_test_tf_idf = train_test_split(
        vectorized_train_set_tf_idf, label, test_size=0.2, random_state=42
    )

    # Flatten the Word2Vec vectors
    X_train_word2vec = [np.array(x).flatten() for x in X_train_word2vec]
    X_test_word2vec = [np.array(x).flatten() for x in X_test_word2vec]

    # Convert lists to numpy arrays
    X_train_word2vec = np.array(X_train_word2vec)
    y_train_word2vec = np.array(y_train_word2vec)
    X_test_word2vec = np.array(X_test_word2vec)
    y_test_word2vec = np.array(y_test_word2vec)

    # Train the model and make predictions using Word2Vec
    y_predict_word2vec = predict(X_train_word2vec, y_train_word2vec, X_test_word2vec)
    print("Classification report for Word2Vec:")
    print(classification_report(y_test_word2vec, y_predict_word2vec))

    # Train the model and make predictions using TF-IDF
    y_predict_tf_idf = predict(X_train_tf_idf, y_train_tf_idf, X_test_tf_idf)
    print("Classification report for TF-IDF:")
    print(classification_report(y_test_tf_idf, y_predict_tf_idf))

    # Tokenize data for BERT
    dataset = tokenize_data_BERT(train_set, label)

    # Initialize BERT model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )

    # Specify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Train BERT model
    trained_model = train_model_BERT(dataset, model)
