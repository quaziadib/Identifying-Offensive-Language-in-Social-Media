# TODO: File Namming and path Fix
# TODO: File import Fix

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd

def data_loader(X_train, X_valid, n1, n2, tp = 'tfidf'):
    #x_train = X_train['tweet'].values
    #x_valid = X_valid['tweet'].values
    #x_test = X_test['tweet'].values

    if tp == 'tfidf':
        tfidf = TfidfVectorizer(ngram_range=(n1, n2))
        x_train = tfidf.fit_transform(X_train.loc[:, 'tweet'].values)
        #x_test = tfidf.transform(x_test)
        x_valid = tfidf.transform(X_valid.loc[:, 'tweet'].values)

        print(tp, n1, n2, 'train', 'start')
        x_train = pd.DataFrame.sparse.from_spmatrix(x_train, columns = tfidf.get_feature_names_out())
        print(tp, n1, n2, 'train', 'mid')
        x_train.to_csv(f'dataset/OLID/feature_vectors/TF-IDF/X_train_tfidf_{n1}-{n2}_gram.csv')
        print(tp, n1, n2, 'train', 'done')
       
        print(tp, n1, n2, 'valid', 'start')
        x_valid = pd.DataFrame.sparse.from_spmatrix(x_valid, columns = tfidf.get_feature_names_out())
        print(tp, n1, n2, 'valid', 'mid')
        x_valid.to_csv(f'dataset/OLID/feature_vectors/TF-IDF/X_valid_tfidf_{n1}-{n2}_gram.csv')
        print(tp, n1, n2, 'valid', 'done')
        #x_test.to_csv(f'/content/drive/MyDrive/CSE440 Project/X_test_tfidf_{n1}-{n2}_gram.csv')
    elif tp=='cv':
        cv = CountVectorizer(ngram_range=(n1, n2))
        x_train = cv.fit_transform(X_train.loc[:, 'tweet'].values)
        #x_test = cv.transform(x_test)
        x_valid = cv.transform(X_valid.loc[:, 'tweet'].values)
        
        print(tp, n1, n2, 'train', 'start')
        x_train = pd.DataFrame.sparse.from_spmatrix(x_train, columns = cv.get_feature_names_out())
        print(tp, n1, n2, 'train', 'mid')
        x_train.to_csv(f'dataset/OLID/feature_vectors/CV/X_train_cv_{n1}-{n2}_gram.csv')
        print(tp, n1, n2, 'train', 'done')

        print(tp, n1, n2, 'valid', 'start')
        x_valid = pd.DataFrame.sparse.from_spmatrix(x_valid, columns = cv.get_feature_names_out())
        print(tp, n1, n2, 'valid', 'mid')
        x_valid.to_csv(f'dataset/OLID/feature_vectors/CV/X_valid_cv_{n1}-{n2}_gram.csv')
        print(tp, n1, n2, 'valid', 'done')
        #x_test.to_csv(f'/content/drive/MyDrive/CSE440 Project/X_test_cv_{n1}-{n2}_gram.csv')
    

def u_tfidf(X_train, X_valid):
    data_loader(X_train, X_valid, 1,1)

def bi_tfidf(X_train, X_valid):
    data_loader(X_train, X_valid, 2,2)

def tri_tfidf(X_train, X_valid):
    data_loader(X_train, X_valid, 3,3)

def u_bi_tfidf(X_train, X_valid):
    data_loader(X_train, X_valid, 1,2)

def bi_tri_tfidf(X_train, X_valid):
    data_loader(X_train, X_valid, 2,3)

def u_bi_tri_tfidf(X_train, X_valid):
    data_loader(X_train, X_valid, 1,3)


def u_cv(X_train, X_valid):
    data_loader(X_train, X_valid, 1,1, 'cv')

def bi_cv(X_train, X_valid):
    data_loader(X_train, X_valid, 2,2, 'cv')

def tri_cv(X_train, X_valid):
    data_loader(X_train, X_valid, 3,3, 'cv')

def u_bi_cv(X_train, X_valid):
    data_loader(X_train, X_valid, 1,2, 'cv')

def bi_tri_cv(X_train, X_valid):
    data_loader(X_train, X_valid, 2, 3, 'cv')

def u_bi_tri_cv(X_train, X_valid):
    data_loader(X_train, X_valid, 1,3, 'cv')


def main():
    X_train_trad = pd.read_csv('dataset\OLID\Preprocessing\Preprocessing_Trad\X_train_trad.csv')
    X_valid_trad = pd.read_csv('dataset\OLID\Preprocessing\Preprocessing_Trad\X_valid_trad.csv')
    u_tfidf(X_train_trad, X_valid_trad)
    bi_tfidf(X_train_trad, X_valid_trad)
    tri_tfidf(X_train_trad, X_valid_trad)
    u_bi_tfidf(X_train_trad, X_valid_trad)
    bi_tri_tfidf(X_train_trad, X_valid_trad)
    u_bi_tri_tfidf(X_train_trad, X_valid_trad)

    u_cv(X_train_trad, X_valid_trad)
    bi_cv(X_train_trad, X_valid_trad)
    tri_cv(X_train_trad, X_valid_trad)
    u_bi_cv(X_train_trad, X_valid_trad)
    bi_tri_cv(X_train_trad, X_valid_trad)
    u_bi_tri_cv(X_train_trad, X_valid_trad)

if __name__ == '__main__':
    main()