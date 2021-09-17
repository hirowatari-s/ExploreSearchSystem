from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import MeCab
import pandas as pd
import numpy as np
import mojimoji
import os
import urllib.request
import re

# 助詞いらないよ
def wakati(text):
    tagger = MeCab.Tagger('')
    tagger.parse('')
    node = tagger.parseToNode(text)
    word_list = []
    while node:
        pos = node.feature.split(",")[0]
        if pos in ["名詞", "動詞", "形容詞"]:   # 対象とする品詞
            word = node.surface
            word_list.append(word)
        node = node.next
    return " ".join(word_list)

# 数字はどれも一緒
def normalize_number(text):
    # 連続した数字を0で置換
    replaced_text = re.sub(r'\d+', '0', text)
    return replaced_text

# Stop wordのため
def download_stopwords(path):
    url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    if os.path.exists(path):
        print('File already exists.')
    else:
        print('Downloading...')
        # Download the file from `url` and save it locally under `file_name`:
        urllib.request.urlretrieve(url, path)

def create_stopwords(file_path):
    stop_words = []
    for w in open(file_path, "r"):
        w = w.replace('\n','')
        if len(w) > 0:
          stop_words.append(w)
    return stop_words


def make_bow(df):
    # Drop Miss Data
    cb1 = (df['snippet'] == 'キャッシュ' )
    cb2 = (df['snippet'] == '類似ページ' )
    condition_bool = (cb1 | cb2)
    df.drop(df[ condition_bool == True ].index, inplace=True)

    # Get out Noise
    df["target"] = df['snippet'].apply(wakati)
    # 半角統一
    df["target"] = df["target"].apply(mojimoji.zen_to_han)
    df["target"] = df["target"].apply(normalize_number)
    # stop word
    path = "stop_words.txt"
    download_stopwords(path)
    stop_words = create_stopwords(path)

    # Execute
    cv = CountVectorizer(stop_words=stop_words, token_pattern=u'(?u)\\b\\w+\\b', max_df=50, min_df=3)
    feature_sparse = cv.fit_transform(df["target"])
    # Tfidf
    tfidf_transformer = TfidfTransformer(norm='l2', sublinear_tf=True)
    tfidf = tfidf_transformer.fit_transform(feature_sparse)


    features = np.array(tfidf.todense())
    labels = df['site_name'].tolist()
    return features, labels, df


if __name__ == '__main__':
    # Load file
    keyword = "ペット"
    df = pd.read_csv(keyword + '.csv')
    features, labels, df = make_bow(df)

    print(features)
    print(features.shape)
    feature_file = 'data/tmp/'+keyword+'.npy'
    label_file = 'data/tmp/'+keyword+'_label.npy'
    np.save(feature_file, features)
    np.save(label_file, labels)
    df.to_csv(keyword + '.csv')
