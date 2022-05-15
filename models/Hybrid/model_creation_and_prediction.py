import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, classification_report
import pickle


X_train = pd.read_csv('XTrain.csv')
X_test = pd.read_csv('XTest.csv')
y_train = pd.read_csv('ytrain.csv')
y_test = pd.read_csv('ytest.csv')


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

model_path = "saved_models/random_forest_det_only.pkl"
pickle.dump(rfc, open(model_path, 'wb'))

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

model_path = "saved_models/decision_tree_det_only.pkl"
pickle.dump(dtc, open(model_path, 'wb'))
    
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

model_path = "saved_models/ada_boosting_det_only.pkl"
pickle.dump(abc, open(model_path, 'wb'))

print("True:", true)
print("Fake:", fake)
print("Overall:", overall)
  
print('-'*10)
print('SVC')
svc = SVC()
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

model_path = "saved_models/svc_det_only.pkl"
pickle.dump(svc, open(model_path, 'wb'))
    
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

model_path = "saved_models/knn_det_only.pkl"
pickle.dump(knn, open(model_path, 'wb'))
    
print("True:", true)
print("Fake:", fake)
print("Overall:", overall)
