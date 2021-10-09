import nltk
import pandas as pd
from nltk.corpus import stopwords

def preprocessing_tag_and_stopwords(s):
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    need_tag = ['JJ','JJR','JJS','NN','NNS','NNP','NNPS','VB','VBD','VBG','VBN','VBP','VBZ']
    stopWords = set(stopwords.words('english'))
    print(stopWords)

    pos = []
    for i in range(len(s)):
        morph = nltk.word_tokenize(s[i])
        l = nltk.pos_tag(morph)
        l_new = []
        for i in l:
            if i[0] not in stopWords and i[1] in need_tag:
                l_new.append(i[0])
        pos.append(l_new)
    return pos

if __name__ == '__main__':
    # Load file
    csv_df = pd.read_csv("~/Downloads/Unsupervised Kernel Regression.csv")
    s = csv_df['snippet']
    pos = preprocessing_tag_and_stopwords(s)
