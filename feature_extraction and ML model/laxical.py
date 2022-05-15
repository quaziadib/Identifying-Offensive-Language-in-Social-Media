from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import pandas as pd
from sklearn.svm import LinearSVC


def data_loader(X_train, y_train, X_valid, y_valid, n1, n2, tp = 'tfidf', feature_type=None):

    print(feature_type)
    if tp == 'tfidf':
        tfidf = TfidfVectorizer(ngram_range=(n1, n2), stop_words='english')
        x_train = tfidf.fit_transform(X_train.loc[:, 'tweet'].values)
        x_valid = tfidf.transform(X_valid.loc[:, 'tweet'].values)

        x_train = pd.DataFrame.sparse.from_spmatrix(x_train, columns = tfidf.get_feature_names_out())
        x_valid = pd.DataFrame.sparse.from_spmatrix(x_valid, columns = tfidf.get_feature_names_out())
        
        model = LinearSVC(max_iter=10000)
        model.fit(x_train, y_train.subtask_a)
        pred = model.predict(x_valid)
        cf = confusion_matrix(y_valid.subtask_a, pred)
        f1 = f1_score(y_valid.subtask_a, pred)
        print("Confusion Matrix")
        print(cf)
        print(f"F1 Score: {f1}")
        print(classification_report(y_valid.subtask_a, pred))

    elif tp=='cv':
        cv = CountVectorizer(ngram_range=(n1, n2), stop_words='english')
        x_train = cv.fit_transform(X_train.loc[:, 'tweet'].values)
        #x_test = cv.transform(x_test)
        x_valid = cv.transform(X_valid.loc[:, 'tweet'].values)

        x_train = pd.DataFrame.sparse.from_spmatrix(x_train, columns = cv.get_feature_names_out())
        x_valid = pd.DataFrame.sparse.from_spmatrix(x_valid, columns = cv.get_feature_names_out())

        
        model = LinearSVC(max_iter=10000)
        model.fit(x_train, y_train.subtask_a)
        pred = model.predict(x_valid)
        cf = confusion_matrix(y_valid.subtask_a, pred)
        f1 = f1_score(y_valid.subtask_a, pred)
        print("Confusion Matrix")
        print(cf)
        print(f"F1 Score: {f1}")
        print(classification_report(y_valid.subtask_a, pred))
        #x_test.to_csv(f'/content/drive/MyDrive/CSE440 Project/X_test_cv_{n1}-{n2}_gram.csv')
    

def u_tfidf(X_train, y_train, X_valid, y_valid):
    data_loader(X_train, y_train, X_valid, y_valid, 1,1,'tfidf', 'u_tfidf')

def bi_tfidf(X_train, y_train, X_valid, y_valid):
    data_loader(X_train, y_train, X_valid, y_valid, 2,2, 'tfidf', 'bi_tfidf')

def tri_tfidf(X_train, y_train, X_valid, y_valid):
    data_loader(X_train, y_train, X_valid, y_valid, 3,3, 'tfidf', 'tri_tfidf')

def u_bi_tfidf(X_train, y_train, X_valid, y_valid):
    data_loader(X_train, y_train, X_valid, y_valid, 1,2, 'tfidf', 'u_bi_tfidf')

def bi_tri_tfidf(X_train, y_train, X_valid, y_valid):
    data_loader(X_train, y_train, X_valid, y_valid, 2,3, 'tfidf', 'bi_tri_tfidf')

def u_bi_tri_tfidf(X_train, y_train, X_valid, y_valid):
    data_loader(X_train, y_train, X_valid, y_valid, 1,3, 'tfidf', 'u_bi_tri_tfidf')


def u_cv(X_train, y_train, X_valid, y_valid):
    data_loader(X_train, y_train, X_valid, y_valid, 1,1, 'cv', 'u_cv')

def bi_cv(X_train, y_train, X_valid, y_valid):
    data_loader(X_train, y_train, X_valid, y_valid, 2,2, 'cv', 'bi_cv')

def tri_cv(X_train, y_train, X_valid, y_valid):
    data_loader(X_train, y_train, X_valid, y_valid, 3,3, 'cv', 'tri_cv')

def u_bi_cv(X_train, y_train, X_valid, y_valid):
    data_loader(X_train, y_train, X_valid, y_valid, 1,2, 'cv', 'u_bi_cv')

def bi_tri_cv(X_train, y_train, X_valid, y_valid):
    data_loader(X_train, y_train, X_valid, y_valid, 2, 3, 'cv', 'bi_tri_cv')

def u_bi_tri_cv(X_train, y_train, X_valid, y_valid):
    data_loader(X_train, y_train, X_valid, y_valid, 1,3, 'cv', 'u_bi_tri_cv')


def main():
    X_train_trad = pd.read_csv('/content/drive/MyDrive/CSE440 Project/Preprocessing_Trad/X_train_trad.csv')
    X_valid_trad = pd.read_csv('/content/drive/MyDrive/CSE440 Project/Preprocessing_Trad/X_valid_trad.csv')
    y_train = pd.read_csv('/content/drive/MyDrive/CSE440 Project/Preprocessing_Trad/y_train.csv')
    y_valid = pd.read_csv('/content/drive/MyDrive/CSE440 Project/Preprocessing_Trad/y_valid.csv')
    
    u_tfidf(X_train_trad, y_train, X_valid_trad, y_valid)
    bi_tfidf(X_train_trad, y_train, X_valid_trad, y_valid)
    tri_tfidf(X_train_trad, y_train, X_valid_trad, y_valid)
    u_bi_tfidf(X_train_trad, y_train, X_valid_trad, y_valid)
    bi_tri_tfidf(X_train_trad, y_train, X_valid_trad, y_valid)
    u_bi_tri_tfidf(X_train_trad, y_train, X_valid_trad, y_valid)

    u_cv(X_train_trad, y_train, X_valid_trad, y_valid)
    bi_cv(X_train_trad, y_train, X_valid_trad, y_valid)
    tri_cv(X_train_trad, y_train, X_valid_trad, y_valid)
    u_bi_cv(X_train_trad, y_train, X_valid_trad, y_valid)
    bi_tri_cv(X_train_trad, y_train, X_valid_trad, y_valid)
    u_bi_tri_cv(X_train_trad, y_train, X_valid_trad, y_valid)

if __name__ == '__main__':
    main()