import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import string
import nltk
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Embedding, LSTM,SpatialDropout1D, Dropout, Bidirectional, Conv2D
from keras.utils.np_utils import to_categorical
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')
punctuations = list(string.punctuation)
stopword = list(stopwords.words('english'))

def preprocessingTraditionalLANG(d):
    tokens = word_tokenize(d)
    final_token = []
    x = WordNetLemmatizer()
    for token in tokens:
        if token not in punctuations:
            if token =='USER' or token == 'URL' or token.lower in stopword:
                #final_token.append(token)
                continue
            else:
                final_token.append(x.lemmatize(token.lower()))

    st = ' '.join(final_token)
    return st


def label_encoder(d):
    if 'N' in d: return 0
    else: return 1

def main():
    # X_test = pd.read_csv('/content/drive/MyDrive/CSE440 Project/testset-levela.tsv', sep = '\t')
    # y_test = pd.read_csv('/content/drive/MyDrive/CSE440 Project/labels-levela.csv')
    # y_test.OFF = y_test.OFF.apply(label_encoder)
    # X_test['tweet'] = X_test.tweet.apply(preprocessingTraditionalLANG)
    
    # X_train = pd.read_csv('/content/drive/MyDrive/CSE440 Project/X_train.csv')
    # X_valid = pd.read_csv('/content/drive/MyDrive/CSE440 Project/X_valid.csv')
    # y_train = pd.read_csv('/content/drive/MyDrive/CSE440 Project/y_train.csv')
    # y_valid = pd.read_csv('/content/drive/MyDrive/CSE440 Project/y_valid.csv')

    df = pd.read_csv('/content/drive/MyDrive/CSE440 Project/olid-training-v1.0.tsv', sep='\t')
    df.subtask_a = df.subtask_a.apply(label_encoder)
    y = df.subtask_a
    # Took this code snippet from https://github.com/shabbirg89/Youtube-2021/blob/main/Multiclass_Text_Classifier_LSTM.ipynb
    maxWord = 10000
    maxSeqLength = 120
    EMBEDDING_DIM = 300
    punctuations = string.punctuation

    tokenizer = Tokenizer(num_words=maxWord, filters=punctuations, lower=True)
    tokenizer.fit_on_texts(df['tweet'].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X = tokenizer.texts_to_sequences(df['tweet'].values)
    X = pad_sequences(X, maxlen=maxSeqLength)
    print('Shape of data tensor:', X.shape)
    y = pd.get_dummies(y).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state = 42)
    print(X_train.shape,y_train.shape)
    print(X_test.shape,y_test.shape)


    # Own Architeucture 
    with tpu_strategy.scope():
        model = Sequential()
        model.add(Embedding(maxWord, EMBEDDING_DIM, input_length=X.shape[1]))
        model.add(Bidirectional(LSTM(300, dropout=0.2, recurrent_dropout=0.2)))
        model.add(Dense(512, activation='softmax'))
        model.add(Dense(256, activation='softmax'))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='softmax'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='softmax'))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    epochs = 150
    batch_size = 64*tpu_strategy.num_replicas_in_sync
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=2, min_delta=0.0001)])
    accr = model.evaluate(X_test,y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    return model

if __name__ == '__main__':
    main()