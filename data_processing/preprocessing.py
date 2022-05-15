import string
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
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

def preprocessing_TRANSFORMER(d):
    tokens = word_tokenize(d)
    st = '[CLS] ' + ' '.join(tokens) + ' [SEP]'
    return st

def label_encoder(d):
    if 'N' in d: return 0
    else: return 1

def main():
    X_train = pd.read_csv('/content/drive/MyDrive/CSE440 Project/X_train.csv')
    X_valid = pd.read_csv('/content/drive/MyDrive/CSE440 Project/X_valid.csv')
    y_train = pd.read_csv('/content/drive/MyDrive/CSE440 Project/y_train.csv')
    y_valid = pd.read_csv('/content/drive/MyDrive/CSE440 Project/y_valid.csv')


    
    y_train.subtask_a = y_train.subtask_a.apply(label_encoder)
    y_valid.subtask_a = y_valid.subtask_a.apply(label_encoder)
    y_train.to_csv('/content/drive/MyDrive/CSE440 Project/Preprocessing_Trad/y_train.csv', index=False)
    y_valid.to_csv('/content/drive/MyDrive/CSE440 Project/Preprocessing_Trans/y_valid.csv', index=False)
    y_train.to_csv('/content/drive/MyDrive/CSE440 Project/Preprocessing_Trans/y_train.csv', index=False)
    y_valid.to_csv('/content/drive/MyDrive/CSE440 Project/Preprocessing_Trad/y_valid.csv', index=False)

    X_train['tweet'] = X_train.tweet.apply(preprocessingTraditionalLANG)
    X_valid['tweet'] = X_valid.tweet.apply(preprocessingTraditionalLANG)
    X_train.to_csv('/content/drive/MyDrive/CSE440 Project/Preprocessing_Trad/X_train_trad.csv', index=False)
    X_valid.to_csv('/content/drive/MyDrive/CSE440 Project/Preprocessing_Trad/X_valid_trad.csv', index=False)
    

    X_train = pd.read_csv('/content/drive/MyDrive/CSE440 Project/X_train.csv')
    X_valid = pd.read_csv('/content/drive/MyDrive/CSE440 Project/X_valid.csv')
    X_train['tweet'] = X_train.tweet.apply(preprocessing_TRANSFORMER)
    X_valid['tweet'] = X_valid.tweet.apply(preprocessing_TRANSFORMER)
    X_train.to_csv('/content/drive/MyDrive/CSE440 Project/Preprocessing_Trans/X_train_trans.csv', index=False)
    X_valid.to_csv('/content/drive/MyDrive/CSE440 Project/Preprocessing_Trans/X_valid_trans.csv', index=False)

if __name__ == '__main__':
    main()
