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
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report
# import pickle



physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


X_train, X_test, y_train, y_test = pd.read_csv('/content/drive/MyDrive/CSE440 Project/Preprocessing_Trans/X_train_trans.csv'), pd.read_csv('/content/drive/MyDrive/CSE440 Project/Preprocessing_Trans/X_valid_trans.csv'), pd.read_csv('/content/drive/MyDrive/CSE440 Project/Preprocessing_Trans/y_train.csv'), pd.read_csv('/content/drive/MyDrive/CSE440 Project/Preprocessing_Trans/y_valid.csv')
X_train = X_train.tweet.values
X_test = X_test.tweet.values

y_train = y_train.subtask_a.values
y_test = y_test.subtask_a.values

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
history = model.fit(X_train, y_train,epochs=10, batch_size=64)
model = keras.Model(inputs = model.inputs, outputs = [model.layers[-2].output])


    
X_train = model.predict(X_train)
X_test = model.predict(X_test)


X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)


print('-'*10)
print('SVC')
svc = LinearSVC()
svc.fit(X_train, y_train)
preds = svc.predict(X_test)
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(8, 6))
plt.title('Confusion Matrix', size=16)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.show()
report = classification_report(y_test, preds, output_dict=True)
true = report['1']
fake = report['0']
overall = {"Accuracy": accuracy_score(y_test, preds), "recall": recall_score(y_test, preds),
               "f1-score": f1_score(y_test, preds), "precision": precision_score(y_test, preds) }

# model_path = "saved_models/svc_det_only.pkl"
# pickle.dump(svc, open(model_path, 'wb'))
    
print("True:", true)
print("Fake:", fake)
print("Overall:", overall)

# X_train = pd.read_csv('XTrain.csv')
# X_test = pd.read_csv('XTest.csv')
# y_train = pd.read_csv('ytrain.csv')
# y_test = pd.read_csv('ytest.csv')


print('-'*10)
print('RandomForestClassifier')
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
preds = rfc.predict(X_test)
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(8, 6))
plt.title('Confusion Matrix', size=16)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.show()
report = classification_report(y_test, preds, output_dict=True)
true = report['1']
fake = report['0']
overall = {"Accuracy": accuracy_score(y_test, preds), "recall": recall_score(y_test, preds),
               "f1-score": f1_score(y_test, preds), "precision": precision_score(y_test, preds) }

# # model_path = "saved_models/random_forest_det_only.pkl"
# # pickle.dump(rfc, open(model_path, 'wb'))

print("True:", true)
print("Fake:", fake)
print("Overall:", overall)
      
    
print('-'*10)
print('DecisionTreeClassifier')
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
preds = dtc.predict(X_test)
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(8, 6))
plt.title('Confusion Matrix', size=16)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.show()
report = classification_report(y_test, preds, output_dict=True)
true = report['1']
fake = report['0']
overall = {"Accuracy": accuracy_score(y_test, preds), "recall": recall_score(y_test, preds),
               "f1-score": f1_score(y_test, preds), "precision": precision_score(y_test, preds) }

# # model_path = "saved_models/decision_tree_det_only.pkl"
# # pickle.dump(dtc, open(model_path, 'wb'))
    
print("True:", true)
print("Fake:", fake)
print("Overall:", overall)

  
print('-'*10)
print('AdaBoostClassifier')
abc = AdaBoostClassifier()
abc.fit(X_train, y_train)
preds = abc.predict(X_test)
cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(8, 6))
plt.title('Confusion Matrix', size=16)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.show()
report = classification_report(y_test, preds, output_dict=True)
true = report['1']
fake = report['0']
overall = {"Accuracy": accuracy_score(y_test, preds), "recall": recall_score(y_test, preds),
               "f1-score": f1_score(y_test, preds), "precision": precision_score(y_test, preds) }

# model_path = "saved_models/ada_boosting_det_only.pkl"
# pickle.dump(abc, open(model_path, 'wb'))

print("True:", true)
print("Fake:", fake)
print("Overall:", overall)
  

    

print('-'*10)
print('KNN')
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
preds = knn.predict(X_test)
prec = precision_score(y_test, preds)
f1score = f1_score(y_test, preds)

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(8, 6))
plt.title('Confusion Matrix', size=16)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.show()
report = classification_report(y_test, preds, output_dict=True)
true = report['1']
fake = report['0']

overall = {"Accuracy": accuracy_score(y_test, preds), "recall": recall_score(y_test, preds),
               "f1-score": f1_score(y_test, preds), "precision": precision_score(y_test, preds) }

# model_path = "saved_models/knn_det_only.pkl"
# pickle.dump(knn, open(model_path, 'wb'))
    
print("True:", true)
print("Fake:", fake)
print("Overall:", overall)
