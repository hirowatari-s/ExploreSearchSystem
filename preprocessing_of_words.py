import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
import numpy as np

def preprocessing_tag_and_stopwords(s):
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    need_tag = ['JJ','JJR','JJS','NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']
    stopWords = set(stopwords.words('english'))

    pos = []
    for i in range(len(s)):
        morph = nltk.word_tokenize(s[i])
        l = nltk.pos_tag(morph)
        l_new = []
        for i in l:
            if i[0] not in stopWords and i[1] in need_tag:
                l_new.append(i[0]+' ')
        pos.append(''.join(l_new))
    return pos

def make_bow(df):
    s = df['snippet']
    pos = preprocessing_tag_and_stopwords(s)

    vectorizer = CountVectorizer(max_df=50, min_df=3)
    X = vectorizer.fit_transform(pos)
    word_label = np.array(vectorizer.get_feature_names())
    # Tfidf
    tfidf_transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
    tfidf = tfidf_transformer.fit_transform(X.toarray())

    features = np.array(tfidf.todense())
    return features, word_label

if __name__ == '__main__':
    # Load file
    csv_df = pd.read_csv("~/Downloads/Unsupervised Kernel Regression.csv")
    features, word_label = make_bow(csv_df)
