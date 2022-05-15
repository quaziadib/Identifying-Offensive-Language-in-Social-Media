import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


df = pd.read_csv('src\X_train_trans.csv')
X = df['tweet']
y = pd.read_csv('src\y_train.csv')
X_train, X_test, y_train, y_test = pd.read_csv('src\X_train_trans.csv'), pd.read_csv('src\X_valid_trans.csv'), pd.read_csv('src\y_train.csv'), pd.read_csv('src\y_valid.csv')

def createModel():
    model = keras.Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    model.add(layers.Conv1D(256, 5, activation='relu'))
    model.add(layers.MaxPooling1D())
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='softmax'))
    return model

def f1_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

puncList = ["।", "”", "“", "’"]
x = "".join(puncList)
filterString = x + '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n০১২৩৪৫৬৭৮৯'
tokenizer = Tokenizer(filters=filterString, lower=False,oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index)+1
maxlen = 800
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    
embedding_dim = 100
model = createModel()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1_metric])
history = model.fit(X_train, y_train,epochs=1, batch_size=16)
model = keras.Model(inputs = model.inputs, outputs = [model.layers[-2].output])


    
X_train = model.predict(X_train)
X_test = model.predict(X_test)


X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
X_train.to_csv('XTrain.csv', index=False)
X_test.to_csv('XTest.csv', index=False)
y_train.to_csv('yTrain.csv', index=False)
y_test.to_csv('yTest.csv', index=False)
